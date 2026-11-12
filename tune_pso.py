"""
Particle-Swarm-Optimisation of Hyper-MML hyper-parameters and soft-computing settings.

Each particle is a configuration; its fitness is the *validation* weighted-F1 reported by train.py
(--valid_frac > 0), so the test set is never used for tuning.  After the swarm converges the best
configuration is re-run once and its test score at the best-validation epoch is reported.

Search space (continuous box, rounded where needed):
    lr, l2 (log-scale), dropout, num_L (hypergraph layers), rs_sigma, rs_max_features, aux_weight

Usage (local):  python tune_pso.py --subject 1 --particles 6 --iters 4 --epochs 20 --no-cuda
"""
import argparse, json, os, subprocess, sys, tempfile, math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from softcomputing import PSO

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (low, high)
SPACE = [("log10_lr", (-5.0, -3.0)), ("log10_l2", (-6.0, -3.0)), ("dropout", (0.2, 0.7)),
         ("num_L", (1.0, 4.0)), ("rs_sigma", (1.5, 5.0)), ("rs_max_features", (16.0, 96.0)),
         ("aux_weight", (0.0, 1.0))]


def decode(x):
    d = dict(zip([n for n, _ in SPACE], x))
    return {"lr": float(10 ** d["log10_lr"]), "l2": float(10 ** d["log10_l2"]), "dropout": float(round(d["dropout"], 3)),
            "num_L": int(round(d["num_L"])), "rs_sigma": float(round(d["rs_sigma"], 3)),
            "rs_max_features": int(round(d["rs_max_features"])), "aux_weight": float(round(d["aux_weight"], 3))}


def run_config(cfg, base_args, python=sys.executable):
    """Run train.py once with `cfg`; returns the parsed RESULT dict (or None on failure)."""
    cmd = [python, os.path.join(HERE, "train.py")] + list(base_args)
    for k, v in cfg.items():
        cmd += ["--" + k, str(v)]
    with tempfile.TemporaryDirectory() as wd:  # train.py drops record_*.pk in cwd
        p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=HERE))
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    sys.stderr.write(p.stderr[-1500:] + "\n")
    return None


def fitness_of(res):
    if res is None or res.get("best_valid_f1") is None or (isinstance(res["best_valid_f1"], float) and math.isnan(res["best_valid_f1"])):
        return 0.0
    return float(res["best_valid_f1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--particles", type=int, default=6)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1, help="parallel evaluations (local)")
    ap.add_argument("--rs_method", default="quickreduct", choices=["quickreduct", "ga"])
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--out", default="pso_result.json")
    a = ap.parse_args()

    base = ["--subject", str(a.subject), "--epochs", str(a.epochs), "--valid_frac", "0.2",
            "--rs_select", a.rs_method, "--rs_weight", "--fuzzy_gate"] + (["--no-cuda"] if a.no_cuda else [])
    pso = PSO([b for _, b in SPACE], n_particles=a.particles, iters=a.iters, seed=0)

    def evaluate_batch(xs):
        cfgs = [decode(x) for x in xs]
        with ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(lambda c: run_config(c, base), cfgs))
        for c, r in zip(cfgs, results):
            print("   ", c, "-> valid F1 =", fitness_of(r))
        return [fitness_of(r) for r in results]

    best_x, best_f, hist = pso.run(evaluate_batch)
    best_cfg = decode(best_x)
    final = run_config(best_cfg, base)
    out = {"best_config": best_cfg, "best_valid_f1": best_f, "history": hist, "final_run": final}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
