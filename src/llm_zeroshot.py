"""Zero-shot instruction-tuned LLM baseline (accuracy-cost reference point).
Scores label options by next-token logits of the digit after the prompt; no generation parsing.
usage: python llm_zeroshot.py MODEL DATASET[,DATASET...] OUT.jsonl
"""
import sys, json, time, numpy as np, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef

model_name, dsets, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
AMZ = ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"]
tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda().eval()

SPEC = {
    "sst5": ("movie review sentence", ["very negative", "negative", "neutral", "positive", "very positive"]),
    "tweet": ("tweet", ["negative", "neutral", "positive"]),
    "amz": ("product review", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
}


def load(name):
    from datasets import load_dataset
    if name == "sst5":
        d = load_dataset("SetFit/sst5")["test"]; return list(d["text"]), np.array(d["label"]), "sst5"
    if name == "tweet":
        d = load_dataset("cardiffnlp/tweet_eval", "sentiment")["test"]; return list(d["text"]), np.array(d["label"]), "tweet"
    doms = AMZ if name == "amz_all" else [name[4:]]
    df = pd.concat([pd.read_json(f"data/processed/amz_{x}_test.jsonl", lines=True) for x in doms])
    return df.text.tolist(), df.label.values, "amz"


for ds in dsets:
    X, y, kind = load(ds); what, labels = SPEC[kind]
    opts = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(labels))
    digit_ids = [tok.encode(str(i + 1), add_special_tokens=False)[0] for i in range(len(labels))]
    prompts = [tok.apply_chat_template([{"role": "user", "content":
               f"Classify the sentiment of this {what}.\n\nText: {t[:1500]}\n\nOptions:\n{opts}\n\nAnswer with the option number only."}],
               tokenize=False, add_generation_prompt=True) for t in X]
    preds, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            b = tok(prompts[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=640).to("cuda")
            # logits_to_keep=1: only the last position's vocab logits (full-sequence logits OOM'd on long reviews)
            logits = m(**b, logits_to_keep=1).logits[:, -1, :]
            preds.append(logits[:, digit_ids].float().argmax(-1).cpu().numpy())
    pr = np.concatenate(preds); dt = time.time() - t0
    rec = dict(model=model_name, dataset=ds, n=len(y), acc=accuracy_score(y, pr), macro_f1=f1_score(y, pr, average="macro"),
               mcc=matthews_corrcoef(y, pr), qwk=cohen_kappa_score(y, pr, weights="quadratic"), mae=float(np.abs(pr - y).mean()),
               ex_per_s=len(y) / dt, params=sum(p.numel() for p in m.parameters()))
    print(rec, flush=True); open(out, "a").write(json.dumps(rec) + "\n")
