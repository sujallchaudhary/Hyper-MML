"""
Run Hyper-MML + soft-computing experiments on Modal GPUs.

One-time setup for REAL data (the preprocessed .pkl files from the README):
    modal volume create hypermml-data
    modal volume put hypermml-data /path/to/EAV /EAV          # must contain EAV/features/{Audio_openSMILE,Vision_manet,EEG_NESTA,lab}

Commands
    modal run modal_app.py::smoke                       # ablation table on SYNTHETIC data (pipeline check only)
    modal run modal_app.py::smoke --data real --subject 1
    modal run modal_app.py::pso   --data real --subject 1 --particles 6 --iters 4
"""
import json
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("hyper-mml-softcomputing")
vol = modal.Volume.from_name("hypermml-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torch_geometric==2.8.1", "numpy", "pandas", "scikit-learn", "scipy")
    .add_local_dir(HERE, remote_path="/app", ignore=["**/.git", "**/__pycache__", "*.pdf", "*.png", "preprocess", "*.pk", "*.json"])
)

# name -> extra train.py flags.  All use a 20% validation split so selection never touches the test set.
CONFIGS = {
    "1_baseline (Hyper-MML)": [],
    "2_+fuzzy-rough QuickReduct": ["--rs_select", "quickreduct"],
    "3_+fuzzy-rough GA": ["--rs_select", "ga"],
    "4_+fuzzy TSK gate": ["--fuzzy_gate"],
    "5_+QuickReduct +boundary weights": ["--rs_select", "quickreduct", "--rs_weight"],
    "6_ALL (GA + weights + gate)": ["--rs_select", "ga", "--rs_weight", "--fuzzy_gate"],
}


@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=3600)
def train_remote(args, data="synthetic", subject=1):
    import subprocess, sys, tempfile
    sys.path.insert(0, "/app")
    env = dict(os.environ, PYTHONPATH="/app")
    if data == "synthetic":
        import make_synthetic
        make_synthetic.make("/tmp/synth/EAV", subject)
        env["HYPERMML_ROOT"] = "/tmp/synth"
    else:
        env["HYPERMML_ROOT"] = "/data"
    cmd = [sys.executable, "/app/train.py", "--subject", str(subject), "--valid_frac", "0.2"] + list(args)
    p = subprocess.run(cmd, cwd=tempfile.mkdtemp(), capture_output=True, text=True, env=env)
    result = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
    gate = [l for l in p.stdout.splitlines() if l.startswith(("Fuzzy gate", "  overall", "  class", "  IF ", "  certainty", "  agreement"))]
    return {"result": result, "gate_report": gate, "gpu": "cuda" if "Running on GPU" in p.stdout else "cpu",
            "err": "" if result else (p.stderr[-2000:] + p.stdout[-1000:])}


@app.local_entrypoint()
def smoke(epochs: int = 30, data: str = "synthetic", subject: int = 1):
    names = list(CONFIGS)
    jobs = [(["--epochs", str(epochs)] + CONFIGS[n], data, subject) for n in names]
    outs = list(train_remote.starmap(jobs))
    print(f"\nHyper-MML soft-computing ablation | data={data} subject={subject} epochs={epochs} (device: {outs[0]['gpu']})")
    print(f"{'configuration':<36}{'valid F1':>9}{'test F1@bestVal':>17}{'test Acc@bestVal':>18}{'#feat (eeg,aud,vis)':>22}")
    for n, o in zip(names, outs):
        r = o["result"]
        if r is None:
            print(f"{n:<36} FAILED\n{o['err']}")
            continue
        print(f"{n:<36}{r['best_valid_f1']:>9.2f}{r['test_f1_at_best_valid']:>17.2f}{r['test_acc_at_best_valid']:>18.2f}{str(r['n_features']):>22}")
    for n, o in zip(names, outs):
        if o["gate_report"]:
            print(f"\n[{n}] fuzzy gate report")
            print("\n".join(o["gate_report"]))
    with open(os.path.join(HERE, f"results_smoke_{data}.json"), "w") as f:
        json.dump({n: o["result"] for n, o in zip(names, outs)}, f, indent=2)


@app.local_entrypoint()
def affec(subjects: str = "1,2", seeds: int = 5, epochs: int = 30):
    """Real AFFEC data (feature pickles in volume path /AFFEC/feature/sub-*/). Every config x subject x seed is one GPU job."""
    import statistics as st
    subs = [int(s) for s in subjects.split(",")]
    names = list(CONFIGS)
    jobs, keys = [], []
    for sub in subs:
        for n in names:
            for seed in range(1, seeds + 1):
                jobs.append((["--Dataset", "AFFEC", "--epochs", str(epochs), "--seed", str(seed)] + CONFIGS[n], "real", sub))
                keys.append((sub, n, seed))
    outs = list(train_remote.starmap(jobs))
    res = {}
    for (sub, n, seed), o in zip(keys, outs):
        if o["result"] is None:
            print(f"FAILED subject={sub} {n} seed={seed}\n{o['err']}")
            continue
        res.setdefault(n, {}).setdefault(sub, []).append(o["result"])
    ms = lambda v: f"{st.mean(v):5.1f}±{(st.pstdev(v) if len(v) > 1 else 0):.1f}"
    print(f"\nAFFEC felt-valence, real data | subjects={subs} seeds={seeds} epochs={epochs} (device: {outs[0]['gpu']})")
    print(f"{'configuration':<36}" + "".join(f"{'sub'+str(s)+' testF1':>16}" for s in subs) + f"{'all runs testF1':>18}{'#feat (eeg,gsr,eye)':>24}")
    for n in names:
        per = [[r["test_f1_at_best_valid"] for r in res.get(n, {}).get(s, [])] for s in subs]
        allv = [x for p in per for x in p]
        nf = next((r["n_features"] for s in subs for r in res.get(n, {}).get(s, [])), "")
        print(f"{n:<36}" + "".join(f"{ms(p) if p else 'n/a':>16}" for p in per) + f"{ms(allv) if allv else 'n/a':>18}{str(nf):>24}")
    with open(os.path.join(HERE, "results_affec_modal.json"), "w") as f:
        json.dump({n: {str(s): v for s, v in d.items()} for n, d in res.items()}, f, indent=2)


@app.function(image=image, timeout=6 * 3600)
def pso_remote(particles, iters, epochs, data, subject, rs_method):
    import sys
    sys.path.insert(0, "/app")
    from softcomputing import PSO
    from tune_pso import SPACE, decode, fitness_of
    base = ["--epochs", str(epochs), "--rs_select", rs_method, "--rs_weight", "--fuzzy_gate"]
    pso = PSO([b for _, b in SPACE], n_particles=particles, iters=iters, seed=0)

    def evaluate_batch(xs):  # whole swarm is evaluated in parallel on separate GPU containers
        cfgs = [decode(x) for x in xs]
        jobs = [(base + sum([["--" + k, str(v)] for k, v in c.items()], []), data, subject) for c in cfgs]
        outs = list(train_remote.starmap(jobs))
        for c, o in zip(cfgs, outs):
            print("   ", c, "-> valid F1 =", fitness_of(o["result"]))
        return [fitness_of(o["result"]) for o in outs]

    best_x, best_f, hist = pso.run(evaluate_batch)
    best_cfg = decode(best_x)
    final = train_remote.remote(base + sum([["--" + k, str(v)] for k, v in best_cfg.items()], []), data, subject)
    return {"best_config": best_cfg, "best_valid_f1": best_f, "history": hist, "final_run": final["result"]}


@app.local_entrypoint()
def pso(particles: int = 6, iters: int = 4, epochs: int = 20, data: str = "synthetic", subject: int = 1, rs_method: str = "quickreduct"):
    out = pso_remote.remote(particles, iters, epochs, data, subject, rs_method)
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, f"results_pso_{data}.json"), "w") as f:
        json.dump(out, f, indent=2)
