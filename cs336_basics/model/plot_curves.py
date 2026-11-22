"""读取 runs/<run_name>/metrics.jsonl，把多次 run 的 loss 曲线画在同一张图上，并打印汇总表。

用法示例：
    # 对比所有学习率实验，图例显示学习率
    uv run --with matplotlib python -m cs336_basics.model.plot_curves runs/lr* \
        --label_keys max_learning_rate --out figures/lr_sweep.png

    # 同时画训练 loss(淡色细线)
    uv run --with matplotlib python -m cs336_basics.model.plot_curves runs/bs* \
        --label_keys batch_size --train --out figures/batch_size.png

每张图有两个子图：横轴分别是梯度步数和墙钟时间（可用 --xs 改成 tokens_seen 等）。
"""
import argparse
import json
import math
from pathlib import Path

from cs336_basics.paths import FIGURES_DIR

X_LABELS = {
    "step": "gradient steps",
    "wallclock": "wall-clock time (min)",
    "tokens_seen": "tokens seen",
}


def load_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train = [r for r in records if "train_loss" in r]
    val = [r for r in records if "val_loss" in r]
    final = next((r for r in reversed(records) if "final_val_loss" in r), None)
    return config, train, val, final


def is_bad(x):
    return x is None or math.isnan(x) or math.isinf(x)


def stability(train, val):
    """返回 "diverged" / "unstable" / ""。
    diverged:loss 出现 NaN/inf,或最后的验证 loss 比初始值还高；
    unstable:中途验证 loss 冲到初始值的 1.1 倍以上，但后来又降回来了。"""
    losses = [r["train_loss"] for r in train] + [r["val_loss"] for r in val]
    if any(is_bad(x) for x in losses):
        return "diverged"
    if len(val) >= 2:
        first = val[0]["val_loss"]
        if val[-1]["val_loss"] > first:
            return "diverged"
        if max(r["val_loss"] for r in val) > 1.1 * first:
            return "unstable"
    return ""


def sort_key(item, label_keys):
    """有 label_keys 时按这些配置值排序（数值按大小），否则按目录名。"""
    run_dir, config = item
    if not label_keys:
        return (run_dir.name,)
    key = []
    for k in label_keys:
        v = config.get(k)
        key.append((0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v)))
    return tuple(key)


def make_label(run_dir, config, label_keys):
    if not label_keys:
        return run_dir.name
    return ", ".join(f"{k}={config.get(k)}" for k in label_keys)


def x_values(records, x_key):
    if x_key == "wallclock":
        return [r["wallclock"] / 60 for r in records]
    return [r[x_key] for r in records]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run 目录，可以用通配符，如 runs/lr*")
    parser.add_argument("--xs", nargs="+", default=["step", "wallclock"], choices=list(X_LABELS))
    parser.add_argument("--label_keys", nargs="*", default=[], help="图例里显示的 config 字段，如 max_learning_rate")
    parser.add_argument("--train", action="store_true", help="同时画训练 loss（淡色细线）")
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None, help="默认取各 run 初始验证 loss 的最大值，发散的曲线会冲出图外")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", default=str(FIGURES_DIR / "loss_curves.png"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = [Path(p) for p in args.runs if (Path(p) / "metrics.jsonl").exists()]
    if not run_dirs:
        raise SystemExit("没有找到包含 metrics.jsonl 的 run 目录")
    items = [(d, json.loads((d / "config.json").read_text())) for d in run_dirs]
    run_dirs = [d for d, _ in sorted(items, key=lambda it: sort_key(it, args.label_keys))]

    fig, axes = plt.subplots(1, len(args.xs), figsize=(6.5 * len(args.xs), 4.5), squeeze=False)
    axes = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    summary = []
    first_vals = []
    all_vals = []
    for i, run_dir in enumerate(run_dirs):
        config, train, val, final = load_run(run_dir)
        color = colors[i % len(colors)]
        status = stability(train, val)
        div = status != ""
        label = make_label(run_dir, config, args.label_keys) + (f"  ({status})" if status else "")
        if val and not is_bad(val[0]["val_loss"]):
            first_vals.append(val[0]["val_loss"])
        all_vals += [r["val_loss"] for r in val if not is_bad(r["val_loss"])]

        for ax, x_key in zip(axes, args.xs):
            if args.train and train:
                ax.plot(x_values(train, x_key), [r["train_loss"] for r in train],
                        color=color, alpha=0.25, linewidth=0.8)
            if val:
                ax.plot(x_values(val, x_key), [r["val_loss"] for r in val],
                        color=color, linewidth=1.8, marker="o", markersize=2.5, label=label,
                        linestyle="--" if div else "-")

        good_vals = [r["val_loss"] for r in val if not is_bad(r["val_loss"])]
        summary.append({
            "run": run_dir.name,
            "label": make_label(run_dir, config, args.label_keys),
            "final_val_loss": final["final_val_loss"] if final else None,
            "best_val_loss": min(good_vals) if good_vals else None,
            "steps": train[-1]["step"] + 1 if train else 0,
            "minutes": (train[-1]["wallclock"] / 60) if train else 0.0,
            "status": status or "ok",
        })

    # 纵轴范围：上限默认是初始 loss，下限默认是最低验证 loss 再留一点余量，
    # 发散 run 的巨大 loss 不会把正常曲线压扁
    ymax = args.ymax if args.ymax is not None else (max(first_vals) * 1.02 if first_vals else None)
    ymin = args.ymin
    if ymin is None and all_vals and ymax is not None:
        lo = min(all_vals)
        ymin = lo - 0.05 * (ymax - lo)
    for ax, x_key in zip(axes, args.xs):
        ax.set_xlabel(X_LABELS[x_key])
        ax.set_ylabel("validation loss" + (" (faint: train)" if args.train else ""))
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}\n")

    # 汇总表：Markdown 格式，可以直接粘进实验日志
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    print("| run | label | final val loss (full set) | best val loss (subset) | steps | train min | status |")
    print("|---|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['run']} | {s['label']} | {fmt(s['final_val_loss'])} | {fmt(s['best_val_loss'])} "
              f"| {s['steps']} | {s['minutes']:.1f} | {s['status']} |")


if __name__ == "__main__":
    main()
