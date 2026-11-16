# Embedding-distance compatibility (added 7 October 2026)

This is the optional “embedding distance if easy” term from the start guide.
It uses frozen R4/R5/R6 checkpoints. It does not retrain, does not change
Tables 1–4, and does not use target test labels to fit a threshold.

For a target test item x with saved probabilities p and saved prediction ŷ:
uncertainty = 1 − H(p)/ln(2)
cls(x) = L2-normalized DistilBERT [CLS] (max_len 128)
centroid[y] = mean cls of source TRAIN items with label y, then L2-normalized
label_compat = (1 + cosine(cls(x), centroid[ŷ])) / 2
nn_compat = (1 + max cosine(cls(x), source TRAIN cls)) / 2
label_rank = share of source VAL items whose label_compat is ≤ this item
combined = min(uncertainty, label_compat)
combined_rank = min(uncertainty, label_rank)

Answer if the chosen score ≥ t, t ∈ {0.3, 0.4, 0.5}. Report every rule and
threshold. Raw cosine values on this DistilBERT sit near 0.8–0.95, so the
guide’s 0.3/0.4/0.5 cuts barely move. label_rank rescales distance using
source validation only, which is allowed; target test labels are not used.

Limitation: cosine-to-centroid is still a proxy, not a proof of label meaning.
