"""Single experiment run: one (dataset, backbone, method, d, lambda, seed).

Methods
  tfidf         TF-IDF (1-2 grams) + logistic regression
  frozen_linear frozen backbone, linear head on [CLS]
  frozen_ae     frozen backbone, autoencoder bottleneck head (recon target = [CLS] itself)
  ft_linear     fine-tuned backbone, dropout + linear head           (standard baseline)
  bn            fine-tuned backbone, bottleneck Linear(H,d)+GELU -> head, lambda forced to 0
  ae            proposed: bn + decoder Linear(d,H) reconstructing the frozen pretrained [CLS]
                embedding; L = CE + lambda * MSE. Decoder is dropped at inference.
  lora_linear   LoRA (r=8, q/v) adapted backbone + linear head
Results (one JSON line per run) are appended to --out; test predictions saved for paired tests.
"""
import argparse, json, os, time, random, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)               # e.g. sst5 | tweet | amz_Books | amz_all
p.add_argument("--tests", default="")                  # comma list; default = same as train
p.add_argument("--model", default="distilbert-base-uncased")
p.add_argument("--method", default="ae")
p.add_argument("--d", type=int, default=128)
p.add_argument("--lam", type=float, default=0.5)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--head_lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--maxlen", type=int, default=128)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--tag", default="")
p.add_argument("--out", default="results/runs.jsonl")
p.add_argument("--data", default="data/processed")
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
AMZ = ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"]


def load(name, split):
    import pandas as pd
    from datasets import load_dataset
    if name == "sst5":
        d = load_dataset("SetFit/sst5")["validation" if split == "val" else split]
        return list(d["text"]), np.array(d["label"])
    if name == "tweet":
        d = load_dataset("cardiffnlp/tweet_eval", "sentiment")["validation" if split == "val" else split]
        return list(d["text"]), np.array(d["label"])
    doms = AMZ if name == "amz_all" else [name[4:]]
    df = pd.concat([pd.read_json(f"{args.data}/amz_{x}_{split}.jsonl", lines=True) for x in doms])
    return df.text.tolist(), df.label.values


tests = args.tests.split(",") if args.tests else [args.train]
Xtr, ytr = load(args.train, "train"); Xva, yva = load(args.train, "val")
TEST = {t: load(t, "test") for t in tests}
C = int(max(ytr.max(), max(v[1].max() for v in TEST.values())) + 1)


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(prob, y):
    pr = prob.argmax(1); cm = confusion_matrix(y, pr, labels=list(range(C)))
    err = pr != y
    return dict(acc=accuracy_score(y, pr), macro_f1=f1_score(y, pr, average="macro"),
                weighted_f1=f1_score(y, pr, average="weighted"), mcc=matthews_corrcoef(y, pr),
                qwk=cohen_kappa_score(y, pr, weights="quadratic"), ece=ece(prob, y),
                mae=float(np.abs(pr - y).mean()),
                adjacent_err_share=float((np.abs(pr - y)[err] == 1).mean()) if err.any() else 0.0,
                per_class_f1=f1_score(y, pr, average=None, labels=list(range(C))).tolist(), cm=cm.tolist())


rec = dict(vars(args)); rec["C"] = C; rec["n_train"] = len(ytr)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
pred_dir = os.path.join(os.path.dirname(args.out), "preds"); os.makedirs(pred_dir, exist_ok=True)
run_id = f"{args.tag}_{args.train}_{args.model.split('/')[-1]}_{args.method}_d{args.d}_l{args.lam}_s{args.seed}"
rec["run_id"] = run_id
t0 = time.time()

# ---------------------------------------------------------------- TF-IDF baseline
if args.method == "tfidf":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    A = vec.fit_transform(Xtr)
    best = None
    for Cr in [0.5, 1, 2, 4, 8]:
        clf = LogisticRegression(C=Cr, max_iter=3000).fit(A, ytr)
        f = f1_score(yva, clf.predict(vec.transform(Xva)), average="macro")
        if best is None or f > best[0]: best = (f, clf, Cr)
    rec["val_macro_f1"], clf, rec["tfidf_C"] = best
    rec["test"] = {}
    for t, (X, y) in TEST.items():
        prob = clf.predict_proba(vec.transform(X)); rec["test"][t] = metrics(prob, y)
        np.save(f"{pred_dir}/{run_id}__{t}.npy", prob.argmax(1))
    rec["params_trainable"] = int(clf.coef_.size + clf.intercept_.size); rec["time_s"] = time.time() - t0
    for t in rec["test"]: rec["test"][t] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in rec["test"][t].items()}
    open(args.out, "a").write(json.dumps(rec) + "\n"); print(json.dumps({k: rec[k] for k in ["run_id", "val_macro_f1"]}), {t: round(m["macro_f1"], 4) for t, m in rec["test"].items()}); raise SystemExit

# ---------------------------------------------------------------- neural methods
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(args.model)
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled="deberta" not in args.model) if dev == "cuda" else dict(device_type="cpu", enabled=False)


class Head(nn.Module):
    def __init__(self, H, method, d, C, p):
        super().__init__(); self.method = method; self.drop = nn.Dropout(p)
        if method in ("ft_linear", "frozen_linear", "lora_linear"):
            self.cls = nn.Linear(H, C)
        else:
            self.enc = nn.Sequential(nn.Linear(H, d), nn.GELU())         # h = f(Wx+b)
            self.cls = nn.Linear(d, C)
            if method in ("ae", "frozen_ae"):
                self.dec = nn.Linear(d, H)                                 # x_hat = W'h + b'

    def forward(self, x, decode=False):
        if not hasattr(self, "enc"): return self.cls(self.drop(x)), None
        h = self.enc(x); logits = self.cls(self.drop(h))
        return logits, (self.dec(h) if decode and hasattr(self, "dec") else None)


def encode(texts):
    return tok(texts, truncation=True, max_length=args.maxlen, padding=True, return_tensors="pt")


backbone = AutoModel.from_pretrained(args.model, dtype=torch.float32).to(dev)
H = backbone.config.hidden_size
frozen = args.method.startswith("frozen")
lam = 0.0 if args.method in ("bn", "ft_linear", "frozen_linear", "lora_linear") else args.lam
rec["lam_effective"] = lam


def cls_emb(model, batch):
    out = model(**{k: v.to(dev) for k, v in batch.items() if k in ("input_ids", "attention_mask")})
    return out.last_hidden_state[:, 0].float().clone()  # clone: a view would pin the whole hidden-state tensor


if args.method == "lora_linear":
    from peft import LoraConfig, get_peft_model
    tm = {"distilbert": ["q_lin", "v_lin"], "roberta": ["query", "value"], "bert": ["query", "value"]}.get(backbone.config.model_type, "all-linear")
    backbone = get_peft_model(backbone, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=tm))

head = Head(H, args.method, args.d, C, args.dropout).to(dev)

# teacher = frozen copy of the pretrained backbone, gives reconstruction targets for "ae"
teacher = None
if args.method == "ae" and lam > 0:
    teacher = AutoModel.from_pretrained(args.model, dtype=torch.float32).to(dev).eval()
    for q in teacher.parameters(): q.requires_grad_(False)
tnorm = lambda z: F.layer_norm(z, z.shape[-1:])  # scale-free reconstruction target

if frozen:  # precompute embeddings once
    backbone.eval()
    for q in backbone.parameters(): q.requires_grad_(False)

    @torch.no_grad()
    def embed_all(X):
        out = []
        for i in range(0, len(X), 256):
            with torch.autocast(**amp): out.append(cls_emb(backbone, encode(X[i:i + 256])))
        return torch.cat(out)
    Etr, Eva = embed_all(Xtr), embed_all(Xva); ETE = {t: embed_all(v[0]) for t, v in TEST.items()}
    args.epochs = max(args.epochs, 30)

params = [{"params": head.parameters(), "lr": args.head_lr}]
if not frozen: params.append({"params": [q for q in backbone.parameters() if q.requires_grad], "lr": args.lr if args.method != "lora_linear" else 5e-4})
opt = torch.optim.AdamW(params, weight_decay=0.01)
steps_per_epoch = math.ceil(len(ytr) / args.bs); total = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.06 * total))) * max(0.0, (total - s) / max(1, total - int(0.06 * total))))
ytr_t = torch.tensor(ytr)


@torch.no_grad()
def predict(X, E=None):
    backbone.eval(); head.eval(); P = []
    for i in range(0, len(X), 256):
        with torch.autocast(**amp):
            z = E[i:i + 256] if E is not None else cls_emb(backbone, encode(X[i:i + 256]))
            logits, _ = head(z.float())
        P.append(F.softmax(logits.float(), -1).cpu())
    return torch.cat(P).numpy()


if dev == "cuda": torch.cuda.reset_peak_memory_stats()
hist, best = [], (-1, None, -1)
for ep in range(args.epochs):
    if not frozen: backbone.train()
    head.train(); perm = torch.randperm(len(ytr)); s_cls = s_rec = 0.0
    for i in range(0, len(ytr), args.bs):
        idx = perm[i:i + args.bs]; yb = ytr_t[idx].to(dev)
        with torch.autocast(**amp):
            if frozen:
                z = Etr[idx]
            else:
                batch = encode([Xtr[j] for j in idx.tolist()]); z = cls_emb(backbone, batch)
            logits, xh = head(z.float(), decode=lam > 0)
            l_cls = F.cross_entropy(logits.float(), yb)
            l_rec = torch.zeros((), device=dev)
            if lam > 0:
                if teacher is not None:
                    with torch.no_grad(): tgt = cls_emb(teacher, batch)
                else:
                    tgt = z.detach()
                l_rec = F.mse_loss(xh.float(), tnorm(tgt.float()))
            loss = l_cls + lam * l_rec
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in params for q in g["params"]], 1.0); opt.step(); sched.step()
        n = len(idx); s_cls += l_cls.item() * n; s_rec += l_rec.item() * n
    pv = predict(Xva, Eva if frozen else None); mv = f1_score(yva, pv.argmax(1), average="macro")
    hist.append(dict(ep=ep + 1, train_cls=s_cls / len(ytr), train_rec=s_rec / len(ytr), val_macro_f1=mv, val_acc=accuracy_score(yva, pv.argmax(1))))
    if mv > best[0]:
        best = (mv, {k: v.detach().clone() for k, v in head.state_dict().items()},
                ep, None if frozen else {k: v.detach().clone() for k, v in backbone.state_dict().items()})
    print(run_id, hist[-1], flush=True)

head.load_state_dict(best[1])
if best[3] is not None: backbone.load_state_dict(best[3])
rec.update(val_macro_f1=best[0], best_epoch=best[2] + 1, history=hist, time_s=time.time() - t0,
           peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None))
deploy_head = sum(q.numel() for n_, q in head.named_parameters() if not n_.startswith("dec"))
rec["params_head_deployed"] = deploy_head
rec["params_trainable"] = sum(q.numel() for q in head.parameters()) + (0 if frozen else sum(q.numel() for q in backbone.parameters() if q.requires_grad))
rec["params_backbone"] = sum(q.numel() for q in backbone.parameters())
rec["rep_dim"] = H if not hasattr(head, "enc") else args.d
rec["test"] = {}
for t, (X, y) in TEST.items():
    prob = predict(X, ETE[t] if frozen else None); rec["test"][t] = metrics(prob, y)
    np.save(f"{pred_dir}/{run_id}__{t}.npy", prob.argmax(1))
for t in rec["test"]:
    rec["test"][t] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in rec["test"][t].items()}
open(args.out, "a").write(json.dumps(rec) + "\n")
print("DONE", run_id, "val", round(best[0], 4), {t: round(m["macro_f1"], 4) for t, m in rec["test"].items()}, f"{rec['time_s']:.0f}s", flush=True)
