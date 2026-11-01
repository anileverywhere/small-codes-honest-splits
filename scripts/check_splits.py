"""Verify locally built Amazon-4D / leakage splits against the published checksums.
usage: python scripts/check_splits.py [data/processed]"""
import glob, hashlib, json, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
ref = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "summary", "split_checksums.json")))
bad = 0
for name, r in sorted(ref.items()):
    p = os.path.join(D, name)
    if not os.path.exists(p): print("MISSING", name); bad += 1; continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    lines = sorted(hashlib.md5(x["text"].encode()).hexdigest() + "\t" + str(x["label"]) for x in rows)
    ok = len(rows) == r["n"] and hashlib.sha256("\n".join(lines).encode()).hexdigest() == r["sha256"]
    bad += not ok; print("OK  " if ok else "DIFF", name)
print("all splits match" if not bad else f"{bad} split(s) differ"); sys.exit(bool(bad))
