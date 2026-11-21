import json
import time
from dataclasses import asdict
from pathlib import Path


def make_run_name(cfg) -> str:
    """能看出关键配置的 run 名字，末尾加时间戳避免重名覆盖。"""
    stamp = time.strftime("%m%d-%H%M%S")
    return f"lr{cfg.max_learning_rate:g}_bs{cfg.batch_size}_ctx{cfg.context_length}_{stamp}"


class ExperimentLogger:
    """每条记录同时：打印到终端、追加到 runs/<run_name>/metrics.jsonl。"""

    def __init__(self, cfg, run_name: str):
        self.run_name = run_name
        self.run_dir = Path(cfg.out_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 记录实验配置
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        self.metrics_file = open(self.run_dir / "metrics.jsonl", "a")

        print(f"run: {run_name}  ->  {self.run_dir}")

    def log(self, record: dict):
        # 终端
        parts = []
        for k, v in record.items():
            if k == "lr":
                parts.append(f"{k} {v:.2e}")
            elif k == "wallclock" or k.endswith("_time"):
                parts.append(f"{k} {v:.1f}s")
            elif "tokens" in k or k.endswith("ppl"):
                parts.append(f"{k} {v:,.0f}")
            elif isinstance(v, float):
                parts.append(f"{k} {v:.4f}")
            else:
                parts.append(f"{k} {v}")
        print(" | ".join(parts))

        # 本地文件：每行一个 JSON，flush 保证中途崩溃也不丢数据
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()

    def finish(self):
        self.metrics_file.close()
