# CS336 Assignment 1 — Experiment Log

所有尝试都记在这里，包括失败的。每次 run 的原始数据在 `runs/<run_name>/`（`config.json` + `metrics.jsonl`），图在 `figures/`。

## 工作流程

1. 训练（run 名字留空会自动生成，如 `lr0.0003_bs128_ctx256_1121-153000`）：
   ```bash
   uv run python -m cs336_basics.model.train --max_learning_rate 1e-3 --batch_size 128 --max_iters 10000 --eval_interval 250 --log_interval 20
   ```
2. 画图 + 汇总表（表格是 Markdown，可直接粘到下面对应小节）：
   ```bash
   uv run --with matplotlib python -m cs336_basics.model.plot_curves runs/<pattern>* --label_keys max_learning_rate --train --out figures/<name>.png
   ```
3. 在本文件对应小节追加一条记录（模板见下）。

### 指标说明

- **val loss（曲线上的点）**：每 `eval_interval` 步，在验证集上等间隔取的固定子集（`eval_windows` 个窗口）评估，各 run 之间可直接比较。
- **final val loss**：训练结束后在**完整验证集**上的逐 token 平均 loss，报告和达标判断（TinyStories ≤ 1.45）用这个数。
- **wallclock**：训练开始至今的秒数，**已扣除验证评估时间**（checkpoint 保存时间包含在内）。

### 单次 run 记录模板

```markdown
#### <run_name>
- **目的**：想验证什么
- **配置**：只写和基线不同的地方（完整配置见 runs/<run_name>/config.json）
- **结果**：final val loss = ?；训练用时 ? min；是否发散（第几步、什么现象）
- **观察**：曲线形状、grad_norm / tokens_per_sec 有无异常
- **结论 & 下一步**：
```

---

## 基线配置（TinyStories）

| 参数 | 值 |
|---|---|
| vocab_size | 10000 |
| context_length | 256 |
| d_model / d_ff | 512 / 1344 |
| num_layers / num_heads | 4 / 16 |
| RoPE theta | 10000 |
| 总 token 数 | 327,680,000（batch_size × max_iters × 256） |
| batch_size × max_iters | 待定 |
| max / min lr，warmup | 待定 |
| AdamW β1, β2, weight_decay | 0.9, 0.95, 0.01 |
| 梯度裁剪 max_l2_norm | 1.0 |
| eval_interval / eval_windows | 待定 / 4096 |

硬件：（GPU 型号）。基线一次 run 的耗时：? min，吞吐量：? tokens/s。

---

## learning_rate (a)：学习率扫描

**交付**：多个学习率的 learning curve；说明搜索策略；一个 final val loss ≤ 1.45 的模型。

**搜索策略**：

**汇总表**（粘 `plot_curves` 的输出）：

**图**：`figures/lr_sweep.png`

**记录**：

**结论**：

## learning_rate (b)：发散边界与最佳学习率

**交付**：逐步增大学习率的曲线（至少包含一次发散的 run）；分析发散点和最佳学习率的关系。

**记录**：

**结论**：

## batch_size_experiment：batch size 变化

**交付**：从 1 到显存上限多个 batch size 的曲线（包含 64、128），必要时重新调学习率；几句话讨论。

> 提示：不同 batch size 时保持总 token 数不变，比较时可加 `--xs tokens_seen wallclock`。

**记录**：

**结论**：

## generate：生成文本

**交付**：至少 256 个 token 的生成文本（或到第一个 `<|endoftext|>` 为止）；评价流畅度，以及影响输出质量的至少两个因素。

**使用的 checkpoint / temperature / top-p**：

**生成文本**：

**评价**：

## layer_norm_ablation：去掉 RMSNorm

**交付**：去掉 RMSNorm 后在原最佳学习率下的曲线，以及在更低学习率下的曲线；几句话评论 RMSNorm 的作用。

**记录**：

**结论**：

## pre_norm_ablation：post-norm vs pre-norm

**交付**：post-norm Transformer 与 pre-norm 的曲线对比。

**记录**：

**结论**：

## no_pos_emb：NoPE vs RoPE

**交付**：RoPE 与 NoPE 的曲线对比。

**记录**：

**结论**：

## swiglu_ablation：SwiGLU vs SiLU

**交付**：参数量大致相同的 SwiGLU 与 SiLU（d_ff = 4 × d_model）前馈层的曲线对比；几句话讨论。

**记录**：

**结论**：

## main_experiment：OpenWebText

**交付**：OWT 上的 learning curve，并解释和 TinyStories loss 的差别；OWT 模型的生成文本，并解释为什么质量更差。

> 运行时换成 `--train_path data/data_npy/owt_train.npy --val_path data/data_npy/owt_valid.npy --vocab_size 32000`（vocab_size 以你的 OWT tokenizer 为准）。

**记录**：

**结论**：

## leaderboard

**交付**：最终 val loss；横轴为墙钟时间且 < 45 分钟的 learning curve；改动说明。

**改动列表**（每项单独记录效果）：

**最终结果**：
