import argparse
from dataclasses import dataclass, fields
import torch
from torch.profiler import schedule
from cs336_basics.model.model_component import transformer_lm
from cs336_basics.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR, resolve_input, resolve_output


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
 
    # 相对路径会依次在 CWD、DATA_DIR、项目根目录下查找（见 cs336_basics/paths.py）
    train_path: str = str(DATA_DIR / "data_npy/tinystories_train.npy")
    val_path: str = str(DATA_DIR / "data_npy/tinystories_valid.npy")

    seed: int = 337

    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16         
    d_ff: int = 1344
    theta: float = 10000.0
    # max_iters: int = 10000
    max_iters: int = 1000
    # batch_size: int = 128
    batch_size: int = 32
    
    max_l2_norm: float = 1.0
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    ckpt_path: str = ""          # 留空则保存到 runs/<run_name>/model.pt

    # lr schedule
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    # warmup_iters: int = 1000
    warmup_iters: int = 10

    # eval
    log_interval: int = 100
    eval_interval: int = 1000
    eval_windows: int = 4096     # 定期评估用的固定子集窗口数（4096 × 256 ≈ 100 万 token）
    eval_batch_size: int = 256   # 评估不存梯度，可以比训练 batch 大

    # 日志
    run_name: str = ""           # 留空则自动生成，如 lr0.0003_bs64_ctx256_1121-153000
    out_dir: str = str(RUNS_DIR)  # 每个 run 的 config.json / metrics.jsonl 存在 out_dir/run_name/；相对路径基于项目根目录

    # 续训：--resume 1 且 checkpoint 存在时，从中断处接着训练（需配合 --run_name，或 --ckpt_path）
    resume: int = 0

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    for f in fields(Config):
        arg_type = f.type if f.type in (int, float, str) else str
        parser.add_argument(f"--{f.name}", type=arg_type, default=f.default)
    cfg = Config(**vars(parser.parse_args()))
    # 统一解析成绝对路径，之后无论在哪个目录启动都指向同一位置
    cfg.train_path = str(resolve_input(cfg.train_path))
    cfg.val_path = str(resolve_input(cfg.val_path))
    cfg.out_dir = str(resolve_output(cfg.out_dir))
    if cfg.ckpt_path:
        cfg.ckpt_path = str(resolve_input(cfg.ckpt_path, PROJECT_ROOT))  # 已存在则沿用，否则放在项目根目录下
    return cfg


def build_model(cfg: Config, device) -> transformer_lm:
    return transformer_lm(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        theta=cfg.theta,
        device=device,
    )