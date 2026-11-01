"""Inference-cost benchmark for the deployed path of each configuration.

Deployed path = backbone -> [CLS] -> head (decoder dropped). Measures, on the stated hardware:
  params (backbone / head / total), representation size per example (bytes, fp32),
  GPU throughput (batch 64, bf16), CPU latency (batch 1, 8 threads, fp32).
Random weights are fine here: cost does not depend on the trained values.
"""
import time, json, sys, platform, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

texts = ["this product broke after two days and support never answered my emails, very disappointed"] * 64
cfgs = [("distilbert-base-uncased", None), ("distilbert-base-uncased", 16), ("distilbert-base-uncased", 32),
        ("distilbert-base-uncased", 64), ("distilbert-base-uncased", 128), ("distilbert-base-uncased", 256),
        ("roberta-base", None), ("microsoft/deberta-v3-base", None), ("answerdotai/ModernBERT-base", None)]
out = dict(cpu=platform.processor() or platform.machine(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rows=[])
torch.set_num_threads(8)
for name, d in cfgs:
    tok = AutoTokenizer.from_pretrained(name); bb = AutoModel.from_pretrained(name, dtype=torch.float32).eval()
    H = bb.config.hidden_size
    head = nn.Linear(H, 5) if d is None else nn.Sequential(nn.Linear(H, d), nn.GELU(), nn.Linear(d, 5))
    nb, nh = sum(p.numel() for p in bb.parameters()), sum(p.numel() for p in head.parameters())
    x = tok(texts, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    x = {k: v for k, v in x.items() if k in ("input_ids", "attention_mask")}
    row = dict(model=name, d=d, params_backbone=nb, params_head=nh, rep_bytes=4 * (d or H))
    # GPU throughput
    bb.cuda(); head.cuda(); xg = {k: v.cuda() for k, v in x.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled="deberta" not in name):
        for _ in range(5): head(bb(**xg).last_hidden_state[:, 0].float())
        torch.cuda.synchronize(); t = time.time(); n = 20
        for _ in range(n): head(bb(**xg).last_hidden_state[:, 0].float())
        torch.cuda.synchronize(); row["gpu_ex_per_s"] = 64 * n / (time.time() - t)
    # head-only cost (what the bottleneck changes)
    z = torch.randn(4096, H, device="cuda")
    with torch.no_grad():
        for _ in range(5): head(z)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(200): head(z)
        torch.cuda.synchronize(); row["head_us_per_ex"] = (time.time() - t) / (200 * 4096) * 1e6
    # CPU latency, batch 1
    bb.cpu(); head.cpu(); x1 = {k: v[:1] for k, v in x.items()}
    with torch.no_grad():
        for _ in range(3): head(bb(**x1).last_hidden_state[:, 0])
        t = time.time(); n = 30
        for _ in range(n): head(bb(**x1).last_hidden_state[:, 0])
        row["cpu_ms_batch1"] = (time.time() - t) / n * 1000
    print(row, flush=True); out["rows"].append(row)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/bench.json", "w"), indent=1)
