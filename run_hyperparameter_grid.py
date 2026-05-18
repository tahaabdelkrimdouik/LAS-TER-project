"""Grid search sur les hyperparametres du Moteur de Confiance.

Teste un sous-ensemble de (drop, regain, window) sur scenario adversarial.
Plus rapide que le test 30 jours (~10-15 minutes).

Usage:
    python scripts/run_hyperparameter_grid.py > results_hyperparam.txt
"""
import copy
import itertools
import json
import statistics
from fractions import Fraction
from operator import add
from random import randint, seed

from scheduling_algorithms import LAS, Optimal_Alg
from scheduling_functions import compute_energy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def job_instance_creation(ws, T):
    return {i + 1: (w, i, i + T) for i, w in enumerate(ws)}


def random_walk_creation(num_jobs, step_size, random_seed, m, M):
    seed(random_seed)
    ws = [randint(m, M)]
    for _ in range(num_jobs - 1):
        nxt = ws[-1] + randint(-step_size, step_size)
        ws.append(max(m, min(M, nxt)))
    return ws


def LAS_energy_ratio(_J_true, _J_pred, epsilon, alpha, dt):
    J_true = copy.deepcopy(_J_true)
    J_pred = copy.deepcopy(_J_pred)
    speed_sol = LAS(J_pred, J_true, epsilon, dt, alpha)
    e_las = sum(s ** alpha for s in speed_sol) * dt
    J_true = copy.deepcopy(_J_true)
    opt_speed, _ = Optimal_Alg(J_true)
    e_opt = compute_energy(opt_speed, alpha)
    return float(e_las) / e_opt


def apply_dynamic_confidence(_J_pred, _J_true, tolerance_ratio, window, drop, regain, min_trust=0.1):
    trust_level = 1.0
    reality_window = []
    J_filtered = {}
    for i in sorted(_J_pred.keys()):
        w_pred, r, d = _J_pred[i]
        w_true, _, _ = _J_true[i]
        safe_fallback = sum(reality_window) / len(reality_window) if reality_window else w_pred
        w_filt = trust_level * w_pred + (1 - trust_level) * safe_fallback
        J_filtered[i] = (max(0, int(w_filt)), r, d)
        ref_level = safe_fallback if reality_window else max(w_pred, 1)
        err = abs(w_true - w_pred) / max(1, ref_level)
        if err > tolerance_ratio:
            trust_level = max(min_trust, trust_level - drop)
        else:
            trust_level = min(1.0, trust_level + regain)
        reality_window.append(w_true)
        if len(reality_window) > window:
            reality_window.pop(0)
    return J_filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Setup instance adversariale (bruit massif)
    num_jobs, D, alpha, dt = 80, 10, 3, 0.01
    epsilon = Fraction(1, 20)

    # Plusieurs seeds pour moyenner
    seeds = [10, 42, 123, 7, 99]
    noise_level = 500

    # Grille
    drops = [0.3, 0.5, 0.7]
    regains = [0.1, 0.2, 0.3]
    windows = [3, 5, 10]
    tols = [0.4]  # on fixe tol pour reduire la combinatoire

    print(f"=== Grid search hyperparametres (bruit massif +/-{noise_level}) ===")
    print(f"Drops={drops} | Regains={regains} | Windows={windows} | Tols={tols}")
    print(f"Seeds={seeds} (moyenne sur {len(seeds)} repetitions)\n")

    results = []
    grid = list(itertools.product(drops, regains, windows, tols))
    print(f"Total : {len(grid)} configurations x {len(seeds)} seeds = {len(grid)*len(seeds)} runs\n")

    for idx, (drop, regain, window, tol) in enumerate(grid):
        ratios = []
        for s in seeds:
            seed(s)
            w_true = random_walk_creation(num_jobs, 10, s, 10, 100)
            w_pred = [max(0, w + randint(-noise_level, noise_level)) for w in w_true]
            J_true = job_instance_creation(w_true, D)
            J_pred = job_instance_creation(w_pred, D)
            J_filt = apply_dynamic_confidence(J_pred, J_true, tol, window, drop, regain)
            r = LAS_energy_ratio(J_true, J_filt, epsilon, alpha, dt)
            ratios.append(r)
        m = statistics.mean(ratios)
        s_dev = statistics.stdev(ratios)
        results.append({
            "drop": drop, "regain": regain, "window": window, "tol": tol,
            "mean_ratio": m, "std_ratio": s_dev, "ratios": ratios,
        })
        print(f"[{idx+1:2d}/{len(grid)}] drop={drop} regain={regain} k={window} tol={tol} -> "
              f"mean={m:.4f} std={s_dev:.4f}")

    # Trouver le meilleur
    best = min(results, key=lambda r: r["mean_ratio"])
    print("\n" + "=" * 70)
    print("MEILLEURE CONFIGURATION (ratio moyen minimal sur bruit massif)")
    print("=" * 70)
    print(f"drop={best['drop']}  regain={best['regain']}  k={best['window']}  tol={best['tol']}")
    print(f"-> mean ratio = {best['mean_ratio']:.4f} (std {best['std_ratio']:.4f})")

    # Reference : valeurs actuelles du rapport
    ref = next((r for r in results if r["drop"] == 0.5 and r["regain"] == 0.2 and r["window"] == 5), None)
    if ref:
        print(f"\nConfiguration ACTUELLE du rapport (drop=0.5, regain=0.2, k=5) :")
        print(f"-> mean ratio = {ref['mean_ratio']:.4f} (std {ref['std_ratio']:.4f})")
        if ref["mean_ratio"] - best["mean_ratio"] < 0.01:
            print("[OK] Configuration actuelle quasi-optimale (ecart < 0.01)")
        else:
            print(f"[!] Configuration actuelle sub-optimale (ecart {ref['mean_ratio']-best['mean_ratio']:.4f})")

    # Sauvegarde
    with open("results_hyperparam.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultats sauvegardes dans results_hyperparam.json")


if __name__ == "__main__":
    main()
