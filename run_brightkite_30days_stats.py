"""Test 30 jours Brightkite avec statistiques completes.

Sortie : moyennes, ecarts-types, test de Wilcoxon paire LAS vs Moteur.
A executer depuis le dossier racine du projet (LAS-TER-project/).

Usage:
    python scripts/run_brightkite_30days_stats.py > results_30days.txt
"""
import copy
import json
import statistics
from fractions import Fraction
from operator import add
from random import randint, seed

from scheduling_algorithms import (
    Avg_rate,
    BKP_alg,
    LAS,
    OptimalOnline,
    Optimal_Alg,
)
from scheduling_functions import compute_energy


# ---------------------------------------------------------------------------
# Helpers (identiques aux notebooks)
# ---------------------------------------------------------------------------
def AVR_energy_ratio(_J, alpha):
    J = copy.deepcopy(_J)
    avr_speed = Avg_rate(J)
    e_avr = compute_energy(avr_speed, alpha)
    J = copy.deepcopy(_J)
    opt_speed, _ = Optimal_Alg(J)
    e_opt = compute_energy(opt_speed, alpha)
    return float(e_avr) / e_opt


def OA_energy_ratio(_J, alpha):
    J = copy.deepcopy(_J)
    oa_speed = OptimalOnline(J)
    e_oa = sum(s ** alpha for s in oa_speed)
    J = copy.deepcopy(_J)
    opt_speed, _ = Optimal_Alg(J)
    e_opt = compute_energy(opt_speed, alpha)
    return float(e_oa) / e_opt


def LAS_energy_ratio(_J_true, _J_pred, epsilon, alpha, dt):
    J_true = copy.deepcopy(_J_true)
    J_pred = copy.deepcopy(_J_pred)
    speed_sol = LAS(J_pred, J_true, epsilon, dt, alpha)
    e_las = sum(s ** alpha for s in speed_sol) * dt
    J_true = copy.deepcopy(_J_true)
    opt_speed, _ = Optimal_Alg(J_true)
    e_opt = compute_energy(opt_speed, alpha)
    return float(e_las) / e_opt


def apply_dynamic_confidence(
    _J_pred,
    _J_true,
    tolerance_ratio=0.4,
    window=5,
    drop=0.5,
    regain=0.2,
    min_trust=0.1,
):
    """Filtre dynamique du Moteur de Confiance. Conforme au notebook."""
    trust_level = 1.0
    reality_window = []
    J_filtered = {}
    for i in sorted(_J_pred.keys()):
        w_pred, r, d = _J_pred[i]
        w_true, _, _ = _J_true[i]
        safe_fallback = (
            sum(reality_window) / len(reality_window) if reality_window else w_pred
        )
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
# Preparation des donnees Brightkite (extrait minimal du notebook)
# ---------------------------------------------------------------------------
def time_to_int(t):
    t = t[:-1].split(":")[0:2]
    t[1] = t[1][0]
    return int("".join(t))


def date_to_int(d):
    return int("".join(d.split("-")))


def create_input(lst, D):
    time_weight = sorted([(t, lst.count(t)) for t in set(lst)], key=lambda x: x[0])
    res = {}
    for i, (_, w) in enumerate(time_weight):
        res[i + 1] = (w, i, i + D)
    return res


def load_brightkite(path="loc-brightkite_totalCheckins.txt"):
    dates_dict = {}
    with open(path) as fin:
        for line in fin:
            try:
                date_s, time_s = line.split()[1].split("T")
            except (IndexError, ValueError):
                continue
            if len(date_s) != 10:
                continue
            d = date_to_int(date_s)
            t = time_to_int(time_s)
            dates_dict.setdefault(d, []).append(t)
    # filter days with enough data
    filtered = {}
    for d in sorted(dates_dict):
        if len(set(dates_dict[d])) >= 144:
            filtered[d] = sorted(dates_dict[d])
    return filtered


# ---------------------------------------------------------------------------
# Wilcoxon paired test (implementation minimale, evite scipy)
# ---------------------------------------------------------------------------
def wilcoxon_paired(x, y):
    """Wilcoxon signed-rank test (deux echantillons paires).

    Retourne (W, n_used). Approximation normale pour n >= 10 :
    z = (W - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24)
    """
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n < 5:
        return None, n, None
    abs_diffs = sorted(((abs(d), 1 if d > 0 else -1) for d in diffs), key=lambda t: t[0])
    ranks = list(range(1, n + 1))
    W_plus = sum(r for r, (_, s) in zip(ranks, abs_diffs) if s > 0)
    W_minus = sum(r for r, (_, s) in zip(ranks, abs_diffs) if s < 0)
    W = min(W_plus, W_minus)
    mean = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    z = (W - mean) / sd if sd else None
    # p-value bilateral approximee (table normale)
    import math
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2)) if z is not None else None
    return W, n, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(num_days=30, D=20, alpha=3, dt=0.01, epsilon=Fraction(1, 100)):
    print(f"=== Brightkite, {num_days} jours, alpha={alpha}, eps={epsilon} ===\n")
    dates_dict = load_brightkite()
    dates_lst = sorted(dates_dict.keys())
    if len(dates_lst) < num_days + 1:
        print(f"Pas assez de jours valides ({len(dates_lst)}). Reduit num_days.")
        return

    y_AVR, y_OA, y_LAS, y_DYN = [], [], [], []

    for i in range(num_days):
        prev_day = dates_lst[i]
        today = dates_lst[i + 1]
        J_pred_raw = create_input(dates_dict[prev_day], D)
        J_true = create_input(dates_dict[today], D)

        # Aligner les cles (meme nombre de taches)
        n = min(len(J_pred_raw), len(J_true))
        J_pred = {k: J_pred_raw[k] for k in list(J_pred_raw)[:n]}
        J_true = {k: J_true[k] for k in list(J_true)[:n]}

        r_avr = AVR_energy_ratio(J_true, alpha)
        r_oa = OA_energy_ratio(J_true, alpha)
        r_las = LAS_energy_ratio(J_true, J_pred, epsilon, alpha, dt)

        J_filt = apply_dynamic_confidence(J_pred, J_true)
        r_dyn = LAS_energy_ratio(J_true, J_filt, epsilon, alpha, dt)

        print(
            f"Jour {i+1:2d} | AVR={r_avr:.4f} | OA={r_oa:.4f} | "
            f"LAS={r_las:.4f} | DYN={r_dyn:.4f}"
        )
        y_AVR.append(r_avr)
        y_OA.append(r_oa)
        y_LAS.append(r_las)
        y_DYN.append(r_dyn)

    print("\n" + "=" * 70)
    print("STATISTIQUES")
    print("=" * 70)
    for name, vals in [("AVR", y_AVR), ("OA", y_OA), ("LAS", y_LAS), ("DYN", y_DYN)]:
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"{name:>4} : mean={m:.4f} | std={s:.4f} | max={max(vals):.4f}")

    print("\nGAIN MOYEN (LAS - DYN) :", round(statistics.mean(y_LAS) - statistics.mean(y_DYN), 4))
    print(f"GAIN RELATIF : {(1 - statistics.mean(y_DYN)/statistics.mean(y_LAS))*100:.2f}%")

    W, n, p = wilcoxon_paired(y_LAS, y_DYN)
    print(f"\nWilcoxon (LAS vs DYN) : W={W}, n={n}, p~={p:.6f}" if p else f"\nWilcoxon : n trop petit")

    # Sauvegarde JSON
    out = {
        "num_days": num_days, "alpha": alpha, "epsilon": str(epsilon),
        "AVR": y_AVR, "OA": y_OA, "LAS": y_LAS, "DYN": y_DYN,
        "wilcoxon_W": W, "wilcoxon_p": p,
    }
    with open("results_30days.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResultats sauvegardes dans results_30days.json")


if __name__ == "__main__":
    main(num_days=30)
