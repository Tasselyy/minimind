# =============================================================================
# MiniMind LoRA 微调训练脚本
# 功能：在 Full SFT 或预训练权重基础上，仅训练 LoRA 低秩适配器，实现轻量微调
# =============================================================================

import os
import sys

# 声明当前包为 trainer，便于相对导入
__package__ = "trainer"
# 将项目根目录加入 Python 路径，使可执行文件从任意位置运行时都能正确导入模块
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(_project_root)

# 加载 .env 中的环境变量（如 WANDB_API_KEY、SWANLAB_API_KEY 等）
from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, '.env'))

import argparse   # 解析命令行参数
import time       # 计时、ETA 计算
import warnings   # 控制警告输出
import torch
import torch.distributed as dist  # 分布式训练（多 GPU）
from contextlib import nullcontext  # 用于 CPU 时提供“空”的 autocast 上下文
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # 多卡 DDP 包装
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载与分布式采样

# 项目内模块
from model.model_minimind import MiniMindConfig   # 模型配置（hidden_size、层数、MoE 等）
from dataset.lm_dataset import SFTDataset         # SFT 对话格式数据集
from model.model_lora import save_lora, apply_lora  # LoRA 注入与保存
from trainer.trainer_utils import (
    get_lr,              # 余弦退火学习率
    Logger,               # 仅主进程打印日志
    is_main_process,      # 是否为主进程（rank 0）
    lm_checkpoint,        # 保存/加载 checkpoint 与 resume
    init_distributed_mode,  # 初始化 DDP 环境
    setup_seed,           # 固定随机种子
    init_model,            # 加载模型和 tokenizer
    SkipBatchSampler,      # 支持从指定 step 续训的 batch 采样器
)

warnings.filterwarnings('ignore')  # 忽略第三方库的警告，保持终端清爽


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    单轮训练循环。
    epoch: 当前轮次（0-based）
    loader: DataLoader
    iters: 本轮总迭代数（用于学习率计算和日志）
    lora_params: 仅 LoRA 参数的列表，用于梯度裁剪和优化
    start_step: 续训时本轮要跳过的 step 数
    wandb: wandb 或 swanlab 实例，用于记录指标
    """
    start_time = time.time()
    # enumerate(loader, start=start_step+1) 使 step 从 1 开始计数，与续训逻辑一致
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将 batch 放到指定设备（单卡为 cuda:0，DDP 为 cuda:local_rank）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 余弦退火学习率：随 step 从 lr 逐渐衰减到约 0.1*lr
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度前向：在 autocast 上下文中计算，减少显存、加速
        with autocast_ctx:
            res = model(input_ids, labels=labels)  # 返回 loss、aux_loss 等
            loss = res.loss + res.aux_loss         # 主损失 + 辅助损失（若有）
            loss = loss / args.accumulation_steps  # 梯度累积：每步只算 1/N 的损失

        # 混合精度反向：scaler 对 loss 缩放，避免 fp16 下溢
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步做一次参数更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 将梯度反缩放回原始尺度
            # 只对 LoRA 参数做梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)   # 优化器更新参数
            scaler.update()         # 更新 scaler 的缩放因子
            optimizer.zero_grad(set_to_none=True)  # 清空梯度，set_to_none 更省显存

        # 按 log_interval 打印日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 还原为“等效单步”的 loss（便于与未使用梯度累积时对比）
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 预估本 epoch 剩余时间（分钟）
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 按 save_interval 保存 LoRA 权重和 resume 信息（仅主进程）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            # LoRA 权重单独保存为 .pth，便于后续推理时加载
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # LoRA 只保存 LoRA 参数，不保存基座权重
            save_lora(model, lora_save_path)
            # 同时写一份完整 resume 信息（含 optimizer、scaler、epoch、step、wandb_id）
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        # 及时释放显存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 若通过 torchrun 启动且设置了 RANK，则初始化 NCCL 进程组并返回 local_rank
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 每张卡使用自己的 device
    # 固定随机种子，多卡时按 rank 偏移，保证可复现且各卡数据不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查 ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 构建与基座一致的模型配置（须与 from_weight 对应权重一致）
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 若开启续训，则从 ../checkpoints 下加载 *_{hidden_size}_resume.pth
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不需要 autocast，用 nullcontext 避免报错
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置 wandb / SwanLab ==========
    wandb = None
    if args.use_wandb and is_main_process():
        swanlab_key = os.environ.get("SWANLAB_API_KEY")
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            import wandb
            wandb.login(key=wandb_key)
        elif swanlab_key:
            import swanlab as wandb
            wandb.login(api_key=swanlab_key)
        else:
            import wandb
            wandb.login()
        # 续训时沿用之前的 run id，便于曲线连续
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、应用 LoRA、冻结非 LoRA 参数 ==========
    # 从 from_weight（如 full_sft / pretrain）加载基座模型和 tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 在所有符合条件的 Linear 层上注入 LoRA（默认 rank=8），前向变为 Wx + B(Ax)
    apply_lora(model)

    # 统计参数：总参数量 vs LoRA 参数量
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # 冻结基座，只训练 LoRA：非 lora 的 requires_grad=False，lora 的加入优化列表
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 下使用 DistributedSampler 保证各卡看到不同数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # float16 时需要 GradScaler 防止下溢；bfloat16 通常不需要
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 7. 从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)  # strict=False 兼容只保存 LoRA 的 ckp
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 8. DDP 包装模型 ==========
    if dist.is_initialized():
        # 不将 RoPE 的 freqs 参与 DDP 的 broadcast，避免多余同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每轮打乱采样顺序（DDP 下由 sampler 控制）
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 续训时首轮跳过前 start_step 个 batch
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # ========== 10. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
