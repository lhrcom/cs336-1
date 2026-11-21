import torch
from cs336_basics.model.config import Config, build_model, get_device
from cs336_basics.tokenization import Tokenizer
from cs336_basics.model.model_component import softmax

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["config"])     
    model = build_model(cfg, device)   
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
    device = get_device()
    model, cfg = load_model("checkpoint/model.pt", device)
    tokenizer = Tokenizer.from_files(
        "data/tinystories_bpe/vocab.json",
        "data/tinystories_bpe/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    print(generate(model, tokenizer, "Once upon a time",
                   max_new_tokens=256, context_length=cfg.context_length,
                   temperature=0.8, top_p=0.9))