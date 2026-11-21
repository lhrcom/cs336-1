import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model.config import Config, parse_args, build_model, get_device
from cs336_basics.model.loss_func import cross_entropy
from cs336_basics.model.optim_adamw import AdamW
from cs336_basics.model.grad_clipping import gradient_clipping
from cs336_basics.model.loaddata import data_loading
from cs336_basics.model.checkpoint import checkpointing
from cs336_basics.model.lr_schedule import learning_rate_schedule
from cs336_basics.model.experiment_log import ExperimentLogger, make_run_name


def compute_grad_norm(model, device):
    total = torch.zeros((), device=device)
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum()
    return total.sqrt().item()


def make_eval_starts(num_tokens, context_length, max_windows=None):
    """验证集切成互不重叠的窗口，返回每个窗口的起点。
    max_windows=None:全部窗口（完整验证集）；否则等间隔取 max_windows 个（固定子集）"""
    num_windows = (num_tokens - 1) // context_length      # 目标要多看 1 个 token
    starts = np.arange(num_windows) * context_length
    if max_windows is not None and max_windows < num_windows:
        idx = np.linspace(0, num_windows - 1, max_windows).astype(np.int64)
        starts = starts[idx]
    return starts


@torch.no_grad()
def evaluate_loss(model, val_data, starts, context_length, batch_size, device):
    """在给定窗口上算逐 token 平均的验证 loss"""
    offsets = np.arange(context_length + 1)
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for i in range(0, len(starts), batch_size):
        s = starts[i : i + batch_size]
        chunk = np.asarray(val_data[s[:, None] + offsets[None, :]], dtype=np.int64)   # (n, T+1)
        chunk = torch.from_numpy(chunk).to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        n_tok = y.numel()
        total_loss += cross_entropy(model(x), y) * n_tok    # 按 token 数加权
        total_tokens += n_tok
    model.train()
    return (total_loss / total_tokens).item()


def train(cfg: Config):
    device = get_device()

    train_data = np.load(cfg.train_path, mmap_mode="r")
    val_data = np.load(cfg.val_path, mmap_mode="r")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = build_model(cfg, device).to(device)   # 注意力层建在 CPU 上，统一搬到 device
    optimizer = AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    run_name = cfg.run_name or make_run_name(cfg)
    logger = ExperimentLogger(cfg, run_name)
    ckpt_path = cfg.ckpt_path or str(logger.run_dir / "model.pt")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = checkpointing()

    # 固定的评估窗口：训练开始前算一次，之后每次评估都用同一批数据
    subset_starts = make_eval_starts(len(val_data), cfg.context_length, cfg.eval_windows)
    full_starts = make_eval_starts(len(val_data), cfg.context_length)

    tokens_per_step = cfg.batch_size * cfg.context_length

    # 计时：wallclock 扣掉评估花的时间
    start_time = time.perf_counter()
    eval_time_total = 0.0

    def train_elapsed():
        return time.perf_counter() - start_time - eval_time_total

    # 吞吐量统计区间
    interval_start = time.perf_counter()
    steps_in_interval = 0

    for step in range(cfg.max_iters):
        x, y = data_loading(train_data, cfg.batch_size, cfg.context_length, device, rng=rng)
        lr = learning_rate_schedule(step, cfg.max_learning_rate, cfg.min_learning_rate, cfg.warmup_iters, cfg.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss = cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()

        is_log_step = step % cfg.log_interval == 0 or step == cfg.max_iters - 1
        if is_log_step:
            grad_norm = compute_grad_norm(model, device)   # 裁剪前的范数，只在记录时算
        gradient_clipping(model.parameters(), cfg.max_l2_norm)
        optimizer.step()
        steps_in_interval += 1

        # 训练指标
        if is_log_step:
            loss_val = loss.item()     # .item() 会等 GPU 算完，之后读时间才准
            interval_time = time.perf_counter() - interval_start
            logger.log({
                "step": step,
                "wallclock": train_elapsed(),
                "tokens_seen": (step + 1) * tokens_per_step,
                "train_loss": loss_val,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_sec": steps_in_interval * tokens_per_step / interval_time,
            })
            interval_start = time.perf_counter()
            steps_in_interval = 0

        # 定期评估：固定子集
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            eval_start = time.perf_counter()
            val_loss = evaluate_loss(model, val_data, subset_starts, cfg.context_length, cfg.eval_batch_size, device)
            eval_time = time.perf_counter() - eval_start
            eval_time_total += eval_time
            logger.log({
                "step": step,
                "wallclock": train_elapsed(),
                "val_loss": val_loss,
                "val_ppl": math.exp(val_loss),
                "eval_time": eval_time,
            })
            # 评估打断了吞吐量统计区间，重新开始
            interval_start = time.perf_counter()
            steps_in_interval = 0

        if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            ckpt.save_checkpoint(model, optimizer, step, ckpt_path)

    # 最终评估：完整验证集（报告里用这个数）
    final_val_loss = evaluate_loss(model, val_data, full_starts, cfg.context_length, cfg.eval_batch_size, device)
    logger.log({
        "step": cfg.max_iters,
        "wallclock": train_elapsed(),
        "final_val_loss": final_val_loss,
        "final_val_ppl": math.exp(final_val_loss),
        "final_val_tokens": len(full_starts) * cfg.context_length,
    })
    logger.finish()


if __name__ == "__main__":
    train(parse_args())
