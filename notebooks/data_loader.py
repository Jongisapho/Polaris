from datasets import load_dataset
from collections import Counter 
import numpy as np

# Data Exploration Script 

ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
names = ["negative", "neutral", "positive"]

for split in ["train", "validation", "test"]:
    counts = Counter(ds[split]["label"])
    total = len(ds[split])
    print(split, {names[k]: f"{counts[k] / total:.1%}" for k in sorted(counts)})

lengths = [len(t.split()) for t in ds["train"]["text"]]
print(np.percentile(lengths, [50, 90, 95, 99]), max(lengths))

tokens = [w for t in ds["train"]["text"] for w in t.lower().split()]
freq = Counter(tokens)

print("distinct tokens:", len(freq))
for m in (1, 2, 3, 5):
    print(f"min_freq={m}:", sum(1 for c in freq.values() if c >= m))
print(freq.most_common(25))