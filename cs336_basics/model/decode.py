import json
from dataclasses import fields
from pathlib import Path

import torch
from cs336_basics.model.config import Config, build_model, get_device
from cs336_basics.tokenization import Tokenizer
from cs336_basics.model.model_component import softmax

def load_config(ckpt, ckpt_path) -> Config:
    """优先用 checkpoint 里存的 config；旧 checkpoint 没存时，退回到同目录下 ExperimentLogger 写的 config.json。"""
    cfg_dict = ckpt.get("extra", {}).get("config")
    if cfg_dict is None:
        cfg_file = Path(ckpt_path).parent / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"checkpoint 里没有 config，且找不到 {cfg_file}")
        cfg_dict = json.loads(cfg_file.read_text())
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in cfg_dict.items() if k in known})   # 忽略已删除的旧字段

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = load_config(ckpt, ckpt_path)
    model = build_model(cfg, device).to(device)   # 和 train.py 一样：部分层建在 CPU 上，统一搬到 device
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

def top_p_sampling(probs,top_p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumsum - sorted_probs) < top_p
    sorted_probs = sorted_probs * keep
    filtered = torch.zeros_like(probs).scatter(-1,sorted_idx,sorted_probs)
    return filtered / filtered.sum()

@torch.no_grad
def decoding(model, top_p, temperature, prompt_ids, max_new_tokens,context_length, special_tokens_id):
    device = next(model.parameters()).device
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):

        x = torch.tensor(ids[-context_length:], device=device).unsqueeze(0)
        logits = model(x)[0,-1,:]
        if temperature == 0:
            next_id = int(torch.argmax(logits))
        else:
            probs = softmax(logits / temperature)
            if top_p < 1.0:
                probs = top_p_sampling(probs,top_p)

            next_id = int(torch.multinomial(probs, 1))

        ids.append(next_id)
        if next_id == special_tokens_id:
            break

    return ids
    
def generate(model, tokenizer, prompt, max_new_tokens, context_length, temperature=1.0, top_p=1.0):
    
    prompt_ids = tokenizer.encode(prompt)
    special_tokens_id = tokenizer.token_to_id[b"<|endoftext|>"]

    ids = decoding(model, top_p, temperature, prompt_ids,max_new_tokens, context_length, special_tokens_id)

    if ids and ids[-1] == special_tokens_id:   
        ids = ids[:-1]
    return tokenizer.decode(ids)

if __name__ == "__main__":
    import argparse
    from cs336_basics.paths import DATA_DIR, RUNS_DIR, require, resolve_input

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径，如 runs/<run_name>/model.pt 或 <run_name>/model.pt")
    parser.add_argument("--bpe_dir", default=str(DATA_DIR / "tinystories_bpe"), help="包含 vocab.json / merges.txt 的目录")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    ckpt_path = require(resolve_input(args.ckpt, RUNS_DIR))
    bpe_dir = resolve_input(args.bpe_dir)
    device = get_device()
    model, cfg = load_model(ckpt_path, device)
    tokenizer = Tokenizer.from_files(
        require(bpe_dir / "vocab.json"),
        require(bpe_dir / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )
    print(generate(model, tokenizer, args.prompt,
                   max_new_tokens=args.max_new_tokens, context_length=cfg.context_length,
                   temperature=args.temperature, top_p=args.top_p))