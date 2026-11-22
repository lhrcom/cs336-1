"""项目里所有路径的唯一来源，保证在任何目录、任何机器上启动都能找到文件。

- PROJECT_ROOT：由本文件位置推出，与当前工作目录(CWD)无关
- DATA_DIR：数据目录，可用环境变量 CS336_DATA_DIR 覆盖（例如服务器数据盘 /root/autodl-tmp/data）
- RUNS_DIR：实验输出目录，可用环境变量 CS336_RUNS_DIR 覆盖
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CS336_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
RUNS_DIR = Path(os.environ.get("CS336_RUNS_DIR", PROJECT_ROOT / "runs")).expanduser().resolve()
FIGURES_DIR = PROJECT_ROOT / "figures"


def resolve_input(p, *bases) -> Path:
    """解析要读取的路径：绝对路径原样返回；相对路径依次尝试 CWD、各个 base，
    返回第一个存在的；都不存在时返回拼到第一个 base 上的路径（报错信息里能看到完整路径）。"""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    bases = bases or (DATA_DIR, PROJECT_ROOT)
    candidates = [Path.cwd() / p] + [Path(b) / p for b in bases]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (Path(bases[0]) / p).resolve()


def resolve_output(p, base=PROJECT_ROOT) -> Path:
    """解析要写入的路径：绝对路径原样返回；相对路径拼到 base(默认项目根目录)上。"""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (Path(base) / p).resolve()


def require(p) -> Path:
    """文件不存在时给出带绝对路径的明确报错。"""
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到文件: {p}\n"
            f"  DATA_DIR = {DATA_DIR}（可用环境变量 CS336_DATA_DIR 修改）\n"
            f"  PROJECT_ROOT = {PROJECT_ROOT}"
        )
    return p
