"""
Generate a *synthetic* EAV-style feature set (same .pkl layout that datasets.py expects) so the
pipeline can be smoke-tested without the real data.  Accuracy on this data is NOT a research result.

Structure planted in the data (so the soft-computing modules have something to find):
  * only `n_informative` features per modality carry class information, the rest are pure noise
  * a random 25% of samples have one randomly chosen modality corrupted (-> fuzzy gate should down-weight it)
  * 8% of training labels are flipped (-> rough boundary region / sample weighting)
"""
import os, pickle, argparse
import numpy as np

DIMS = {"aud": 1582, "vis": 1024, "eeg": 1024}
FOLDERS = {"aud": "Audio_openSMILE", "vis": "Vision_manet", "eeg": "EEG_NESTA", "lab": "lab"}


def make(root, subject=1, n_train=280, n_test=120, n_classes=5, n_informative=24, seed=0):
    rng = np.random.RandomState(seed + subject)
    n = n_train + n_test
    y = np.tile(np.arange(n_classes), n // n_classes + 1)[:n]
    rng.shuffle(y)
    info_idx, feats = {}, {}
    strength = {"eeg": 0.9, "aud": 1.2, "vis": 1.0}
    for m, d in DIMS.items():
        idx = rng.choice(d, n_informative, replace=False)
        proto = rng.randn(n_classes, n_informative)
        X = rng.randn(n, d)
        X[:, idx] += strength[m] * proto[y]
        info_idx[m], feats[m] = idx, X
    corrupt = rng.rand(n) < 0.25
    which = rng.randint(0, 3, n)
    for k, m in enumerate(DIMS):
        bad = corrupt & (which == k)
        feats[m][bad] = rng.randn(bad.sum(), DIMS[m]) * 2.0
    y_noisy = y.copy()
    flip = rng.rand(n_train) < 0.08
    y_noisy[:n_train][flip] = rng.randint(0, n_classes, flip.sum())
    for m in DIMS:
        d = os.path.join(root, "features", FOLDERS[m]); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"subject_{subject:02d}_{m}.pkl"), "wb") as f:
            pickle.dump((list(feats[m][:n_train].astype(np.float32)), list(feats[m][n_train:].astype(np.float32))), f)
    d = os.path.join(root, "features", FOLDERS["lab"]); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"subject_{subject:02d}_lab.pkl"), "wb") as f:
        pickle.dump((list(y_noisy[:n_train]), list(y[n_train:])), f)
    return info_idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subject", type=int, default=1)
    a = ap.parse_args()
    make(a.root, a.subject)
    print("wrote synthetic EAV features to", a.root)
