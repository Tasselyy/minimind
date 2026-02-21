import os
import sys

# 设置包名为 trainer，为了相对导入
__package__ = "trainer"
# 将上一级目录添加到 sys.path 中，以便能够导入 model, dataset, trainer 等模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略警告信息
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将模型输出的 logits 转换为对应 token 的对数概率 (log probabilities)。
    
    参数:
        logits: 模型的原始输出，形状为 (batch_size, seq_len, vocab_size)
        labels: 真实的标签序列，形状为 (batch_size, seq_len)
    
    返回:
        log_probs_per_token: 每个 token 对应的对数概率，形状为 (batch_size, seq_len)
    """
    # 1. 使用 log_softmax 将 logits (batch_size, seq_len, vocab_size) 转换为对数概率
    # dim=2 表示在 vocab_size 维度上进行 softmax 计算
    log_probs = F.log_softmax(logits, dim=2)
    
    # 2. 从词表大小的分布中，提取出实际 label 对应的 token 的对数概率
    # labels.unsqueeze(2) 将 labels 形状变为 (batch_size, seq_len, 1)
    # torch.gather 根据 index 从 log_probs 中提取对应位置的值
    # 最后 squeeze(-1) 将形状恢复为 (batch_size, seq_len)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO (Direct Preference Optimization) 损失函数。
    DPO 的核心思想是不需要额外的奖励模型，直接使用语言模型本身来优化偏好。
    
    参数:
        ref_log_probs: 参考模型 (Reference Model) 给出的对数概率，形状 (batch_size, seq_len)
        policy_log_probs: 策略模型 (Policy Model，即当前训练模型) 给出的对数概率，形状 (batch_size, seq_len)
        mask: 有效 token 的掩码，形状 (batch_size, seq_len)，1表示有效，0表示 padding
        beta: 缩放系数 (KL散度惩罚项的系数)，用于控制策略模型偏离参考模型的程度
    """
    # 计算每个样本的实际长度（有效 token 数量），使用 clamp_min(1e-8) 防止除以 0
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    
    # 计算序列级别的平均对数概率：
    # 将每个 token 的概率乘以 mask（过滤掉 padding），在 seq_len 维度求和，然后除以有效长度
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # DPO 数据构建时，前半部分是 chosen (被偏好的/高质量的回答)，后半部分是 rejected (被拒绝的/低质量的回答)
    batch_size = ref_log_probs.shape[0]
    
    # 将 chosen 和 rejected 数据的对数概率分开
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # pi_logratios: 策略模型对 chosen 和 rejected 回答的概率对数比值差
    # 策略模型越偏好 chosen，这个值越大
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    
    # ref_logratios: 参考模型对 chosen 和 rejected 回答的概率对数比值差
    # 这个值作为基准 (baseline)
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # logits: 两者之差，表示当前策略模型相比参考模型，对 chosen 偏好的增加量
    logits = pi_logratios - ref_logratios
    
    # 损失函数: -log(sigmoid(beta * logits))
    # 我们希望 logits 越大越好（即策略模型更偏好 chosen），因此经过 sigmoid 后取 -log 作为损失去最小化
    loss = -F.logsigmoid(beta * logits)
    
    # 返回整个 batch 的平均损失
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    执行一个 Epoch 的训练循环。
    """
    start_time = time.time()
    
    # 遍历 DataLoader 获取每一个 batch，start 参数用于从断点恢复时正确计算 step
    for step, batch in enumerate(loader, start=start_step + 1):
        # 1. 获取输入数据并移动到对应设备 (GPU/CPU)
        # x_chosen: 用户输入 + 被偏好的回答 (输入序列)
        # x_rejected: 用户输入 + 被拒绝的回答 (输入序列)
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        # y_chosen/y_rejected: 对应的标签序列 (通常与 x 错位移了一位，用于预测下一个词)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        # 对应的 mask 掩码
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        
        # 将 chosen 和 rejected 沿着 batch 维度拼接 (dim=0)
        # 这样一次前向传播就能同时计算 chosen 和 rejected
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 2. 动态调整学习率 (学习率预热/衰减)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 3. 前向传播和损失计算
        with autocast_ctx: # 使用混合精度上下文，加速计算并减少显存占用
            # 3.1 参考模型前向传播 (不计算梯度)
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # 将 logits 转为每个 token 的对数概率
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # 3.2 策略模型（正在训练的模型）前向传播
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # 3.3 计算 DPO 损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # 总损失 = DPO损失 + 辅助损失 (例如 MoE 架构中的负载均衡损失)
            loss = dpo_loss_val + outputs.aux_loss
            # 如果使用了梯度累积，需要对 loss 进行缩放
            loss = loss / args.accumulation_steps

        # 4. 反向传播
        # 使用 scaler 缩放 loss 防止 FP16 下的梯度下溢
        scaler.scale(loss).backward()

        # 5. 梯度累积和参数更新
        # 当达到累积步数时，才执行一次参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 恢复梯度缩放
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新模型参数
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子
            scaler.update()
            # 清空梯度，set_to_none=True 可以稍微节省显存并提升速度
            optimizer.zero_grad(set_to_none=True)

        # 6. 打印日志和记录 Wandb
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 恢复 loss 为真实的数值（乘回累积步数）
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算剩余时间评估 (ETA)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印日志到控制台
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            # 同步数据到 wandb 平台（如果开启）
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 7. 保存模型 Checkpoint
        # 满足保存间隔，且当前进程是主进程（分布式训练时避免重复保存）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval() # 切换为评估模式
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 构建模型权重保存路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 处理 DDP 模型封装，获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model) # 处理 torch.compile 的情况
            state_dict = raw_model.state_dict()
            
            # 将参数转换为 FP16 并移至 CPU 后保存，节省存储空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存额外的训练状态 (如 optimizer, scaler等)，用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train() # 切回训练模式
            del state_dict # 删除字典释放内存

        # 8. 释放局部变量，防止显存泄漏
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境，获取当前进程的 local_rank
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子，保证可复现。多进程环境下不同进程种子略有不同避免数据读取重复
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果开启了断点续训，尝试从 checkpoints 目录加载之前的训练状态
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 初始化自动混合精度上下文 (AMP, Automatic Mixed Precision)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb可视化 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 这里使用了 swanlab 作为 wandb 的替代，提供类似的日志记录功能
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        # 初始化实验记录
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型和参考模型 ==========
    # 初始化要训练的策略模型 (Policy Model)
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        # 使用 torch.compile 进行图计算优化加速
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 初始化参考模型 (Reference Model)
    # DPO 算法需要一个固定的参考模型来作为基准
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval() # 设为评估模式
    ref_model.requires_grad_(False) # 冻结参考模型，不计算梯度
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # 初始化 DPO 数据集，处理偏好数据对
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果是分布式训练，使用 DistributedSampler 分配数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 初始化梯度缩放器（针对 FP16 防止下溢）和 AdamW 优化器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从检查点 (checkpoint) 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重、优化器状态、缩放器状态以及 epoch/step 进度
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 使用 DDP 封装模型 (分布式数据并行) ==========
    if dist.is_initialized():
        # 忽略 RoPE (旋转位置编码) 的预计算矩阵，避免不同卡之间不必要的同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 确保每个 epoch 数据打乱顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        
        # 计算需要跳过的 batch 数量（用于断点续训）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 自定义 BatchSampler 支持跳过特定的步骤
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        
        # 构建 DataLoader
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        # 调用训练函数执行具体的 epoch 训练流程
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
