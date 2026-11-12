"""
Build Hyper-MML-style feature pickles from the open AFFEC dataset (Zenodo 14794876, CC-BY 4.0).

IMPORTANT: these are classical hand-crafted features, NOT the paper's pretrained ABEMA / MA-Net embeddings,
so results are not comparable with the published numbers.  They are meant to let the soft-computing
pipeline run on real physiological data.

  EEG  : log band power (delta, theta, alpha, beta, gamma) per channel          -> 63 x 5 = 315 dims
  GSR  : statistics of skin conductance / resistance / temperature + SCR peaks  -> 31 dims
  EYE  : gaze position + fixation statistics + pupil-diameter statistics        -> 56 dims
  label: felt valence (1-9) -> 3 classes: low (<=3), medium (4-6), high (>=7)

Each trial (stimulus viewing phase: scenario + video) is cut into N equal windows. The train/test split is
done per TRIAL (stratified) before windowing, so windows of one trial never appear in both splits.

Usage:  python -I build_affec_features.py RAW_ROOT OUT_ROOT [subject ...]
  RAW_ROOT contains sub-XXX/ folders (BIDS);  output -> OUT_ROOT/AFFEC/feature/sub-XXX/sub-XXX_{eeg,gsr,eye,lab}.pkl
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd
from scipy import signal, stats

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_WINDOWS = 5
TEST_FRAC = 0.2
SEED = 0


def read_physio(root, sub, kind, run):
    base = f"{root}/{sub}/beh/{sub}_task-fer_run-{run}_recording-{kind}_physio"
    cols = json.load(open(base + ".json"))["Columns"]
    return pd.read_csv(base + ".tsv.gz", sep="\t", header=None, names=cols, na_values="n/a")


def stats13(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return [np.nan] * 13
    t = np.arange(len(x))
    slope = np.polyfit(t, x, 1)[0] * len(x)
    d = np.diff(x)
    return [x.mean(), x.std(), x.min(), x.max(), np.median(x), *np.percentile(x, [10, 25, 75, 90]),
            slope, np.abs(d).mean(), stats.skew(x), stats.kurtosis(x)]


def gsr_features(df):  # 13 + 13 + 2 + 3 = 31
    cond = df["GSR_Conductance_cal"].to_numpy(float)
    res = df["GSR_cal"].to_numpy(float)
    temp = df["Temperature_cal"].to_numpy(float)
    peaks, props = signal.find_peaks(cond, prominence=0.02) if len(cond) > 5 else ([], {"prominences": []})
    prom = props["prominences"]
    return (stats13(cond) + stats13(res) + [np.nanmean(temp), np.nanstd(temp)] +
            [len(peaks), prom.mean() if len(prom) else 0.0, np.nanmax(cond) - np.nanmin(cond)])


def eye_features(gaze, pupil):  # 13 + 13 + 13 + 13 + 4 = 56
    ok = gaze["BPOGV"].to_numpy() == 1
    gx = np.clip(gaze["BPOGX"].to_numpy(float), 0, 1)[ok]
    gy = np.clip(gaze["BPOGY"].to_numpy(float), 0, 1)[ok]
    lp = pupil["LPD"].to_numpy(float)
    rp = pupil["RPD"].to_numpy(float)
    lp = lp[(lp > 1) & (lp < 40)]
    rp = rp[(rp > 1) & (rp < 40)]
    fix = gaze.drop_duplicates("FPOGID")
    fix = fix[fix["FPOGV"] == 1]
    step = np.hypot(np.diff(gx), np.diff(gy)) if len(gx) > 1 else np.array([0.0])
    extra = [ok.mean(), len(fix) / max(gaze["onset"].max() - gaze["onset"].min(), 1e-3),
             np.clip(fix["FPOGD"], 0, 5).mean() if len(fix) else 0.0, step.mean()]
    return stats13(gx) + stats13(gy) + stats13(lp) + stats13(rp) + extra


def eeg_features(data, sfreq):  # data [ch, t] -> ch*5
    f, pxx = signal.welch(data, fs=sfreq, nperseg=min(int(sfreq), data.shape[1]), axis=-1)
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out.append(np.log10(pxx[:, m].mean(axis=1) + 1e-20))
    return np.stack(out, axis=1).ravel()


def felt_valence_class(v):
    return 0 if v <= 3 else (1 if v <= 6 else 2)


def build_subject(raw_root, out_root, sub):
    import mne
    sub = sub if sub.startswith("sub-") else "sub-" + sub
    rng = np.random.RandomState(SEED)
    trials = []  # (run, trial, label, eeg[W,315], gsr[W,31], eye[W,56])
    for run in range(4):
        ev_path = f"{raw_root}/{sub}/{sub}_task-fer_run-{run}_events.tsv"
        if not os.path.exists(ev_path):
            continue
        ev = pd.read_csv(ev_path, sep="\t")
        beh = pd.read_csv(f"{raw_root}/{sub}/beh/{sub}_task-fer_run-{run}_beh.tsv", sep="\t")
        raw = mne.io.read_raw_edf(f"{raw_root}/{sub}/eeg/{sub}_task-fer_run-{run}_eeg.edf", preload=True, verbose="ERROR")
        eeg, sf = raw.get_data(), raw.info["sfreq"]
        gsr, gaze, pupil = (read_physio(raw_root, sub, k, run) for k in ("gsr", "gaze", "pupil"))
        for _, b in beh.iterrows():
            e = ev[(ev.trial == b.trial) & ev.flag.isin(["scenario", "video"])]
            if e.empty:
                continue
            t0, t1 = e.onset.min(), (e.onset + e.duration).max()
            if t1 > raw.times[-1] or t1 - t0 < N_WINDOWS * 2.0:
                continue
            edges = np.linspace(t0, t1, N_WINDOWS + 1)
            F = {"eeg": [], "gsr": [], "eye": []}
            for a, z in zip(edges[:-1], edges[1:]):
                F["eeg"].append(eeg_features(eeg[:, int(a * sf):int(z * sf)], sf))
                F["gsr"].append(gsr_features(gsr[(gsr.onset >= a) & (gsr.onset < z)]))
                F["eye"].append(eye_features(gaze[(gaze.onset >= a) & (gaze.onset < z)], pupil[(pupil.onset >= a) & (pupil.onset < z)]))
            trials.append((run, int(b.trial), felt_valence_class(int(b.f_emotion_v)), *(np.array(F[k]) for k in ("eeg", "gsr", "eye"))))
        print(f"  {sub} run{run}: {len(trials)} trials so far", flush=True)

    # stratified split at trial level
    labels = np.array([t[2] for t in trials])
    test_idx = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test_idx.extend(idx[:max(1, int(round(TEST_FRAC * len(idx))))])
    is_test = np.zeros(len(trials), dtype=bool)
    is_test[test_idx] = True

    def collect(mask):
        out = {k: [] for k in ("eeg", "gsr", "eye", "lab", "grp")}
        for keep, t in zip(mask, trials):
            if not keep:
                continue
            for w in range(N_WINDOWS):
                out["eeg"].append(t[3][w]); out["gsr"].append(t[4][w]); out["eye"].append(t[5][w]); out["lab"].append(t[2]); out["grp"].append(t[0] * 1000 + t[1])
        return out

    tr, te = collect(~is_test), collect(is_test)
    for k in ("gsr", "eye"):  # impute NaN with the TRAIN median
        med = np.nanmedian(np.array(tr[k]), axis=0)
        for d in (tr, te):
            arr = np.array(d[k]); r, c = np.where(~np.isfinite(arr)); arr[r, c] = med[c]; d[k] = arr
    d_out = f"{out_root}/AFFEC/feature/{sub}"
    os.makedirs(d_out, exist_ok=True)
    for k in ("eeg", "gsr", "eye"):
        pickle.dump((list(np.asarray(tr[k], dtype=np.float32)), list(np.asarray(te[k], dtype=np.float32))),
                    open(f"{d_out}/{sub}_{k}.pkl", "wb"))
    # slots 3/7 of the lab tuple hold trial ids (window groups) so splits can keep trials together
    pickle.dump((None, None, tr["lab"], tr["grp"], None, None, te["lab"], te["grp"]),
                open(f"{d_out}/{sub}_lab.pkl", "wb"))
    print(f"{sub}: {len(trials)} trials ({(~is_test).sum()} train / {is_test.sum()} test) -> "
          f"{len(tr['lab'])} train / {len(te['lab'])} test windows | label dist train {np.bincount(tr['lab'])} test {np.bincount(te['lab'])}")
    print("feature dims: eeg", np.array(tr['eeg']).shape[1], "gsr", np.array(tr['gsr']).shape[1], "eye", np.array(tr['eye']).shape[1])


if __name__ == "__main__":
    raw_root, out_root, subs = sys.argv[1], sys.argv[2], sys.argv[3:]
    for s in subs:
        build_subject(raw_root, out_root, s)
