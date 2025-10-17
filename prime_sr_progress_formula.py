#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prime Symbolic Regression – Validierung, Huber-Loss, π-Seeding (/7), Glättung, Mini-Batches,
Konstanten-Finetuning, reproduzierbare Checkpoints, schnelleres Sieb (odd-bytearray).

Abhängigkeiten: deap, tqdm
"""

import os
import re
import csv
import math
import time
import argparse
import random
import statistics
import copy
import pickle
from typing import List, Sequence, Tuple, Optional

# tqdm (Progressbar) – Fallback ohne Balken
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

# DEAP
from deap import base, gp, tools, algorithms, creator


# ----------------- Utils -----------------
def set_seeds(seed: int = 1337):
    random.seed(seed)

def pretty_sec(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, r = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{r:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"

def safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else 0.0


# ----------------- Primes / Sieve (verbessert) -----------------
def upper_bound_for_nth_prime(n: int) -> int:
    # Für sehr kleine n genügt eine kleine Schranke
    if n < 6:
        return 15
    ln = math.log(n)
    lnln = math.log(ln)
    return int(n * (ln + lnln) * 1.2) + 64

def sieve_primes_upto(limit: int, show_progress: bool = True) -> List[int]:
    """
    Schnelles odd-only Sieve mit bytearray.
    Speichert nur ungerade Zahlen: Index i repräsentiert Zahl (2*i+1).
    """
    if limit < 2:
        return []
    size = (limit // 2) + 1  # für odd: 1,3,5,...,limit (2*i+1)
    sieve = bytearray(b"\x01") * size
    sieve[0] = 0  # 1 ist nicht prim
    r = int(limit ** 0.5)
    it = range(3, r + 1, 2)
    if show_progress:
        it = tqdm(it, desc=f"Sieb (odd) bis {limit:,}", dynamic_ncols=True, leave=False)
    for p in it:
        if sieve[p // 2]:
            start = p * p
            step = 2 * p
            # Nur ungerade Indizes markieren
            for m in range(start, limit + 1, step):
                sieve[m // 2] = 0
    primes = [2] + [2 * i + 1 for i in range(1, size) if sieve[i]]
    return primes

def first_n_primes(N: int, show_progress: bool = True) -> List[int]:
    tries = 0
    lim = upper_bound_for_nth_prime(N)
    while True:
        tries += 1
        print(f"[PRIMES] Versuch {tries}: limit={lim:,}")
        t0 = time.time()
        primes = sieve_primes_upto(lim, show_progress=show_progress)
        print(f"[PRIMES] Gefunden: {len(primes):,} ≤ {lim:,} in {pretty_sec(time.time()-t0)}")
        if len(primes) >= N:
            print(f"[PRIMES] Nehme erste {N:,} Primzahlen.")
            return primes[:N]
        lim = int(lim * 1.3) + 1000
        print("[PRIMES] Erhöhe Grenze und wiederhole…")


# ----------------- Baseline & Dataset -----------------
def p_baseline_list(n_list: Sequence[int]) -> List[float]:
    out: List[float] = []
    small = [0.0, 2.0, 3.0, 5.0, 7.0, 11.0]
    for n in n_list:
        if n < 6:
            out.append(small[n])
            continue
        ln = math.log(n)
        lnln = math.log(ln)
        out.append(n * (ln + lnln - 1.0 + (lnln - 2.0) / ln))
    return out

def make_dataset(N: int, save_prefix: str, progress: bool = True):
    print(f"[DATA] Erzeuge Datensatz N={N:,} …")
    t0 = time.time()
    primes = first_n_primes(N, show_progress=progress)
    idx = list(range(1, N + 1))
    baseline = p_baseline_list(idx)
    residual = [float(primes[i]) - baseline[i] for i in range(N)]

    rows = []
    it = range(N)
    if progress:
        it = tqdm(it, desc="Feature-Engineering", dynamic_ncols=True)
    for i in it:
        n = idx[i]
        ln = safe_log(float(n))
        lnln = safe_log(ln)
        invln = (1.0 / ln) if ln > 0.0 else 0.0
        sqrt_n = math.sqrt(float(n))
        rows.append({
            "n": n,
            "p": primes[i],
            "baseline": baseline[i],
            "residual": residual[i],
            "ln": ln,
            "lnln": lnln,
            "invln": invln,
            "sqrt_n": sqrt_n,
            "ones": 1.0,
        })

    csv_path = f"{save_prefix}_dataset.csv"
    print(f"[DATA] Schreibe CSV → {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        it = rows if not progress else tqdm(rows, desc="CSV write", dynamic_ncols=True, leave=False)
        for r in it:
            writer.writerow(r)

    print(f"[DATA] Fertig in {pretty_sec(time.time()-t0)}")
    return idx, primes, baseline, rows


# ----------------- GP Primitives -----------------
def p_add(a, b): return a + b
def p_sub(a, b): return a - b
def p_mul(a, b): return a * b
def p_div(a, b): return a / b if abs(b) > 1e-12 else a
def p_min(a, b): return a if a < b else b
def p_max(a, b): return a if a > b else b
def p_log(a):   return math.log(abs(a) + 1e-12)
def p_sqrt(a):  return math.sqrt(abs(a))
def p_abs(a):   return abs(a)
def p_inv(a):   return 1.0 / a if abs(a) > 1e-12 else 0.0
def p_sq(a):    return a * a
def p_cu(a):    return a * a * a
def p_log1p(a): return math.log1p(abs(a))

def _rand_const() -> float:
    # Breiter Bereich, aber nicht zu brutal (vermeidet NaNs)
    return random.uniform(-10.0, 10.0)


# ----------------- Formel-Pretty-Printer -----------------
def _fmt_num(x) -> str:
    try:
        xf = float(x)
        if abs(xf - int(xf)) < 1e-12:
            return str(int(xf))
        return f"{xf:.6g}"
    except Exception:
        return str(x)

def individual_to_infix(ind: gp.PrimitiveTree, var_names: List[str]) -> str:
    def binop(sym, a, b): return f"({a} {sym} {b})"
    fmap_bin = {
        "p_add": lambda a, b: binop("+", a, b),
        "p_sub": lambda a, b: binop("-", a, b),
        "p_mul": lambda a, b: binop("*", a, b),
        "p_div": lambda a, b: binop("/", a, b),
        "p_min": lambda a, b: f"min({a}, {b})",
        "p_max": lambda a, b: f"max({a}, {b})",
    }
    fmap_un = {
        "p_log":   lambda a: f"log(|{a}| + 1e-12)",
        "p_sqrt":  lambda a: f"sqrt(|{a}|)",
        "p_abs":   lambda a: f"abs({a})",
        "p_inv":   lambda a: f"(1/({a}))",
        "p_sq":    lambda a: f"({a})^2",
        "p_cu":    lambda a: f"({a})^3",
        "p_log1p": lambda a: f"log1p(|{a}|)",
    }
    def render(i: int):
        node = ind[i]
        if isinstance(node, gp.Terminal):
            if node.value is None:
                name = node.name
                m = re.search(r'(\d+)$', name)
                if m:
                    idx = int(m.group(1))
                    return var_names[idx], i + 1
                return name, i + 1
            name = getattr(node, "name", "")
            if name == "PI":  # Benannte Konstante hübsch drucken
                return "pi", i + 1
            return _fmt_num(node.value), i + 1
        if isinstance(node, gp.Primitive):
            next_i = i + 1
            args = []
            for _ in range(node.arity):
                s, next_i = render(next_i)
                args.append(s)
            if node.arity == 2 and node.name in fmap_bin:
                return fmap_bin[node.name](args[0], args[1]), next_i
            if node.arity == 1 and node.name in fmap_un:
                return fmap_un[node.name](args[0]), next_i
            return f"{node.name}(" + ", ".join(args) + ")", next_i
        return str(node), i + 1
    s, _ = render(0)
    return s

VAR_NAMES = ["ln", "lnln", "invln", "sqrt_n", "one"]


# ----------------- Toolbox, π-Seeds & Eval -----------------
def build_primitive_set(feature_names: List[str]) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("MAIN", [float for _ in feature_names], float, prefix="X")
    # Binär
    pset.addPrimitive(p_add, [float, float], float)
    pset.addPrimitive(p_sub, [float, float], float)
    pset.addPrimitive(p_mul, [float, float], float)
    pset.addPrimitive(p_div, [float, float], float)
    pset.addPrimitive(p_min, [float, float], float)
    pset.addPrimitive(p_max, [float, float], float)
    # Unär
    pset.addPrimitive(p_log,   [float], float)
    pset.addPrimitive(p_sqrt,  [float], float)
    pset.addPrimitive(p_abs,   [float], float)
    pset.addPrimitive(p_inv,   [float], float)
    pset.addPrimitive(p_sq,    [float], float)
    pset.addPrimitive(p_cu,    [float], float)
    pset.addPrimitive(p_log1p, [float], float)
    # Konstanten (fix) + PI
    for c in [-5.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0]:
        pset.addTerminal(float(c), float)
    pset.addTerminal(math.pi, float, name="PI")  # π
    # Ephemere Konstanten
    pset.addEphemeralConstant("rand", _rand_const, float)
    return pset

def build_pi_seeds(pset, count: int) -> List[gp.PrimitiveTree]:
    """
    π-Startformeln (mit explizitem '/ 7.0'), um deinen Wunsch zu erfüllen.
    Variablen: X0=ln, X1=lnln, X2=invln, X3=sqrt_n, X4=one.
    """
    # PI/7 * irgendwas Sinnvolles rund um ln, lnln
    exprs = [
        # f1 = (PI/7) * inv( ln + lnln )
        "p_mul(p_div(PI, 7.0), p_inv(p_add(X0, X1)))",
        # f2 = (PI/7) / ( 8 + ln^2 )
        "p_div(p_div(PI, 7.0), p_add(8.0, p_sq(X0)))",
        # f3 = (PI/7) / ( 8 + ln * |invln| )
        "p_div(p_div(PI, 7.0), p_add(8.0, p_mul(X0, p_abs(X2))))",
        # f4 = (PI/7) * inv( 0.5 + sqrt_n )
        "p_mul(p_div(PI, 7.0), p_inv(p_add(0.5, X3)))",
        # f5 = (PI/7) / ( 8 + (ln + lnln)^2 )
        "p_div(p_div(PI, 7.0), p_add(8.0, p_sq(p_add(X0, X1))))",
    ]
    exprs = exprs[:max(0, count)]
    seeds = []
    for s in exprs:
        try:
            t = gp.PrimitiveTree.from_string(s, pset)
            seeds.append(t)
        except Exception:
            pass
    return seeds


# ----------------- Metriken -----------------
def _round_nearest(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))

def metrics_report(y_true: List[int], y_pred: List[float], prefix: str):
    abs_err = [abs(y_pred[i] - y_true[i]) for i in range(len(y_true))]
    rel_err = [abs_err[i] / max(1.0, abs(y_true[i])) for i in range(len(y_true))]
    pred_int = [_round_nearest(v) for v in y_pred]
    d = [abs(pred_int[i] - y_true[i]) for i in range(len(y_true))]
    hits1 = sum(1 for v in d if v <= 1) / len(d)
    hits2 = sum(1 for v in d if v <= 2) / len(d)
    hits5 = sum(1 for v in d if v <= 5) / len(d)
    print(f"[METRICS] {prefix}:")
    print(f"  MAE: {statistics.mean(abs_err):.3f} | Median AE: {statistics.median(abs_err):.3f}")
    print(f"  Rel. Fehler Ø: {statistics.mean(rel_err):.5f}")
    print(f"  Treffer ±1: {hits1*100:.2f}% | ±2: {hits2*100:.2f}% | ±5: {hits5*100:.2f}%")


# ----------------- Evaluation/Toolbox mit Val-Split, Huber, Smoothness, Mini-Batches -----------------
def make_toolbox_with_eval(
    *,
    X_train: List[Tuple[float, ...]],
    y_train: List[float],
    B_train: List[float],
    X_val: List[Tuple[float, ...]],
    y_val: List[float],
    B_val: List[float],
    scale_train: Optional[List[float]],
    scale_val: Optional[List[float]],
    pred_mode: str,
    residual_scale: str,
    tournsize: int,
    huber_delta: float,
    fitness_mix: float,
    smoothness_coef: float,
    parsimony: float,
    init_depth_min: int,
    init_depth_max: int,
    mut_depth_min: int,
    mut_depth_max: int,
    max_size: int
):
    """
    Erstellt Toolbox & PSet sowie eval_ind, das:
      - auf Val (und optional Train) bewertet,
      - Huber-Loss nutzt,
      - glättende Krümmungsstrafe ansetzt,
      - Parsimony bremst,
      - optional skaliertes Residuum verwendet,
      - im Modus 'residual' oder 'multiplicative' vorhersagt.
    """
    assert pred_mode in ("residual", "multiplicative")
    pset = build_primitive_set(VAR_NAMES)

    # DEAP Creator
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=init_depth_min, max_=init_depth_max)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", copy.deepcopy)

    # Referenz-MAE (Baseline) auf Val zur Normierung
    baseline_abs_err_val = [abs(B_val[i] - y_val[i]) for i in range(len(y_val))]
    BASE_MAE_VAL = max(1e-9, statistics.mean(baseline_abs_err_val))

    def huber(e: float, delta: float) -> float:
        ae = abs(e)
        return 0.5 * e * e if ae <= delta else delta * (ae - 0.5 * delta)

    # Helfer: Vorhersage auf Listen
    def predict_list(func, X, B, scale, mode: str) -> List[float]:
        out: List[float] = []
        if mode == "residual":
            if scale is None:
                for i in range(len(X)):
                    out.append(B[i] + float(func(*X[i])))
            else:  # skaliertes Residuum
                for i in range(len(X)):
                    out.append(B[i] + float(func(*X[i])) * scale[i])
        else:  # multiplicative
            if scale is None:
                for i in range(len(X)):
                    out.append(B[i] * (1.0 + float(func(*X[i]))))
            else:
                # Multiplikativ + Skala ist ungewöhnlich; interpretieren als: 1 + f(X)*scale
                for i in range(len(X)):
                    out.append(B[i] * (1.0 + float(func(*X[i])) * scale[i]))
        return out

    # Optionale Mini-Batches – Werte später über closure-Variablen
    batch_val_count = 0
    batch_repeats = 1

    def set_batch_params(n_val: int, cnt: int, reps: int):
        nonlocal batch_val_count, batch_repeats
        batch_val_count = max(0, min(cnt, n_val))
        batch_repeats = max(1, reps)

    def eval_split_loss(func, use_batches: bool = False) -> float:
        """
        Gibt aggregierten Score (je höher desto besser) zurück.
        Bewertet primär Val (optional gemischt mit Train).
        """
        # Hilfsfunktion für einen Datensatz:
        def dataset_loss(XL, YL, BL, SL) -> Tuple[float, float, float, float]:
            # y_pred
            y_pred = predict_list(func, XL, BL, SL, pred_mode)
            # Fehler
            errs = [y_pred[i] - YL[i] for i in range(len(YL))]
            mae = statistics.mean(abs(e) for e in errs)
            hub = statistics.mean(huber(e, huber_delta) for e in errs)
            # Glättung (2. Differenz) für sanfte Kurven
            smooth = 0.0
            if len(y_pred) >= 3:
                last2 = y_pred[0]
                last1 = y_pred[1]
                for k in range(2, len(y_pred)):
                    cur = y_pred[k]
                    smooth += abs(cur - 2 * last1 + last2)
                    last2, last1 = last1, cur
                smooth /= (len(y_pred) - 2)
            return mae, hub, smooth, float(len(y_pred))

        # Val – ggf. mini-batch
        if use_batches and batch_val_count > 0 and batch_val_count < len(X_val):
            # wiederhole mehrere Batches und mitteln
            mae_v_sum = hub_v_sum = smooth_v_sum = 0.0
            for _ in range(batch_repeats):
                idxs = random.sample(range(len(X_val)), batch_val_count)
                XL = [X_val[i] for i in idxs]
                YL = [y_val[i] for i in idxs]
                BL = [B_val[i] for i in idxs]
                SL = None if scale_val is None else [scale_val[i] for i in idxs]
                mae_v, hub_v, sm_v, _ = dataset_loss(XL, YL, BL, SL)
                mae_v_sum += mae_v; hub_v_sum += hub_v; smooth_v_sum += sm_v
            mae_val = mae_v_sum / batch_repeats
            hub_val = hub_v_sum / batch_repeats
            smooth_val = smooth_v_sum / batch_repeats
        else:
            mae_val, hub_val, smooth_val, _ = dataset_loss(X_val, y_val, B_val, scale_val)

        # Train – nur wenn Mischung < 1.0
        if fitness_mix < 1.0:
            mae_tr, hub_tr, smooth_tr, _ = dataset_loss(X_train, y_train, B_train, scale_train)
            mae = fitness_mix * mae_val + (1.0 - fitness_mix) * mae_tr
            hub = fitness_mix * hub_val + (1.0 - fitness_mix) * hub_tr
            smooth = fitness_mix * smooth_val + (1.0 - fitness_mix) * smooth_tr
        else:
            mae, hub, smooth = mae_val, hub_val, smooth_val

        # relativer Gewinn ggü. Baseline
        gain = (BASE_MAE_VAL - mae_val) / BASE_MAE_VAL

        # Score (Maximierung)
        score = (
            1.0 * gain
            - 0.2 * (hub / (1.0 + BASE_MAE_VAL))
            - smoothness_coef * smooth
        )
        return score

    def eval_ind(individual):
        func = toolbox.compile(expr=individual)
        try:
            score = eval_split_loss(func, use_batches=True)
            # Parsimony-Strafe separat
            score -= parsimony * len(individual)
        except Exception:
            score = -1e9
        return (float(score),)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=mut_depth_min, max_=mut_depth_max)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_size))

    # Batch-Parameter setter zurückgeben
    return toolbox, pset, set_batch_params, BASE_MAE_VAL


# ----------------- π-Seeds injizieren -----------------
def inject_seeds(pop, seed_inds: List[gp.PrimitiveTree], max_inject: int):
    if not seed_inds or max_inject <= 0:
        return
    k = min(max_inject, len(pop), len(seed_inds))
    for i in range(k):
        pop[i] = creator.Individual(copy.deepcopy(seed_inds[i]))


# ----------------- Konstanten-Finetuning (ohne SciPy) -----------------
def tune_constants_local_hillclimb(
    ind: gp.PrimitiveTree,
    toolbox: base.Toolbox,
    X_train: List[Tuple[float, ...]],
    y_train: List[float],
    B_train: List[float],
    scale_train: Optional[List[float]],
    pred_mode: str,
    iters: int = 150,
    step0: float = 1.0,
    step_decay: float = 0.9
):
    """
    Einfache, robuste Hillclimb-Suche über ephemere Konstanten.
    Ändert die Werte direkt im Baum, wenn es Train-MAE verbessert.
    """
    # Sammle Indizes ephemerer/konstanter Terminals (keine Variablen, kein 'PI')
    const_idx: List[int] = []
    for i, node in enumerate(ind):
        if isinstance(node, gp.Terminal):
            if node.value is None:
                continue
            if getattr(node, "name", "") == "PI":
                continue
            # float-Konstanten
            if isinstance(node.value, float):
                const_idx.append(i)

    if not const_idx:
        return ind  # Nichts zu tun

    # Helpers
    def predict_list(func, X, B, S) -> List[float]:
        out = []
        if pred_mode == "residual":
            if S is None:
                for i in range(len(X)): out.append(B[i] + float(func(*X[i])))
            else:
                for i in range(len(X)): out.append(B[i] + float(func(*X[i])) * S[i])
        else:
            if S is None:
                for i in range(len(X)): out.append(B[i] * (1.0 + float(func(*X[i]))))
            else:
                for i in range(len(X)): out.append(B[i] * (1.0 + float(func(*X[i])) * S[i]))
        return out

    def train_mae(cur_ind) -> float:
        f = toolbox.compile(expr=cur_ind)
        pred = predict_list(f, X_train, B_train, scale_train)
        return statistics.mean(abs(pred[i] - y_train[i]) for i in range(len(y_train)))

    # Start
    best = copy.deepcopy(ind)
    best_mae = train_mae(best)
    step = float(step0)

    for _ in range(max(1, iters)):
        improved = False
        for j in const_idx:
            for sign in (+1.0, -1.0):
                trial = copy.deepcopy(best)
                trial[j].value = float(trial[j].value) + sign * step
                mae = train_mae(trial)
                if mae + 1e-12 < best_mae:
                    best, best_mae, improved = trial, mae, True
        step *= step_decay
        if not improved:
            # kleine Zufallsstörung, um aus Plateaus zu kommen
            for j in const_idx:
                best[j].value = float(best[j].value) + random.uniform(-0.1 * step0, 0.1 * step0)
            # keine MAE-Prüfung hier; nächste Runde entscheidet
    # Ergebnis zurücktragen
    for i, node in enumerate(best):
        ind[i] = node
    return ind


# ----------------- Evolution (mit Elitismus, Checkpoints, Mini-Batches) -----------------
def _safe_formula_text(ind) -> str:
    try:
        return individual_to_infix(ind, VAR_NAMES)
    except Exception as e:
        raw = str(ind)
        for i, nm in enumerate(VAR_NAMES):
            raw = raw.replace(f"X{i}", nm).replace(f"ARG{i}", nm)
        raw = raw.replace("PI", "pi")
        return raw + f"   # Fallback ({type(e).__name__})"

def _save_formula(ind, path: str):
    txt = _safe_formula_text(ind)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Formel f(ln, lnln, invln, sqrt_n, one) – π erlaubt als 'pi'\n")
        f.write(txt + "\n")
    print(f"[SAVE] Formel gespeichert → {path}")

def gp_evolve_with_progress(
    *,
    toolbox: base.Toolbox,
    pset,
    pop_size: int,
    generations: int,
    cxpb: float,
    mutpb: float,
    progress: bool,
    ckpt_every: int,
    elite: int,
    print_topk: int,
    early_stop: int,
    max_formula_chars: int,
    seed_inds: List[gp.PrimitiveTree],
    set_batch_params,  # Funktion: (n_val, batch_count, repeats)
    batch_val_count: int,
    batch_repeats: int,
    save_prefix: str
):
    pop = toolbox.population(n=pop_size)
    inject_seeds(pop, seed_inds, max_inject=len(seed_inds))

    hof = tools.HallOfFame(max(5, print_topk))

    # Batch-Params aktivieren
    # Wir kennen die Val-Länge erst im Main, deshalb wird set_batch_params dort aufgerufen.

    # Initiale Fitness
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit
    hof.update(pop)

    def _avg(xs): return statistics.mean(xs) if xs else 0.0
    def _std(xs): return statistics.pstdev(xs) if len(xs) > 1 else 0.0
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", _avg); stats.register("std", _std)
    stats.register("min", min);  stats.register("max", max)

    t0 = time.time()
    iterator = range(1, generations + 1)
    if progress:
        iterator = tqdm(iterator, desc="GP Evolution", dynamic_ncols=True)

    best_so_far = hof[0].fitness.values[0]
    stall = 0

    for gen in iterator:
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))

        # Elitismus: top 'elite' Individuen sichern (ersetze schlechteste)
        if elite > 0 and len(hof) > 0:
            pop_sorted = sorted(pop, key=lambda ind: ind.fitness.values[0])
            elites = [copy.deepcopy(hof[i % len(hof)]) for i in range(elite)]
            pop_sorted[:elite] = elites
            pop = pop_sorted

        hof.update(pop)
        rec = stats.compile(pop)

        if progress:
            iterator.set_postfix(best=f"{rec['max']:.3f}", avg=f"{rec['avg']:.3f}",
                                 std=f"{rec['std']:.3f}", size=len(hof[0]))
        else:
            print(f"[GP] Gen {gen:04d} | best={rec['max']:.4f} avg={rec['avg']:.4f} std={rec['std']:.4f} size={len(hof[0])}")

        # Checkpoints
        if ckpt_every and (gen % ckpt_every == 0):
            try:
                tqdm.write(f"[FORMEL] Gen {gen:04d} | size={len(hof[0])} | best={rec['max']:.4f}")
            except Exception:
                print(f"[FORMEL] Gen {gen:04d} | size={len(hof[0])} | best={rec['max']:.4f}")
            k = min(print_topk, len(hof))
            for i in range(k):
                txt = _safe_formula_text(hof[i])
                if len(txt) > max_formula_chars:
                    txt = txt[:max_formula_chars] + " …"
                line = f"  TOP{i+1}: {txt}"
                try: tqdm.write(line)
                except Exception: print(line)

            # Speichere Text + Pickle (Repro)
            _save_formula(hof[0], f"{save_prefix}_ckpt_gen{gen}_best_formula.txt")
            try:
                with open(f"{save_prefix}_ckpt_gen{gen}_best.pkl", "wb") as fh:
                    pickle.dump({"ind": hof[0], "pset_name": pset.name}, fh)
                print(f"[SAVE] Pickle gespeichert → {save_prefix}_ckpt_gen{gen}_best.pkl")
            except Exception:
                pass

        # Early-Stop?
        if rec['max'] > best_so_far + 1e-12:
            best_so_far = rec['max']
            stall = 0
        else:
            stall += 1
            if early_stop and stall >= early_stop:
                print(f"[EARLY-STOP] Keine Verbesserung für {stall} Generationen → Abbruch.")
                break

    print(f"[GP] Fertig in {pretty_sec(time.time()-t0)} – Beste Fitness: {hof[0].fitness.values[0]:.6f}")
    return pop, hof, stats


# ----------------- Save CSV -----------------
def save_predictions(n, y_true, y_pred, baseline_vals, save_prefix: str, k: int = 5000, progress: bool = True):
    k = min(k, len(n))
    path = f"{save_prefix}_predictions_head{k}.csv"
    print(f"[SAVE] Schreibe {k:,} Beispiel-Vorhersagen → {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "prime_true", "baseline", "pred", "pred_int", "abs_err", "rel_err"])
        rng = range(k)
        if progress:
            rng = tqdm(rng, desc="Save head CSV", dynamic_ncols=True, leave=False)
        for i in rng:
            pred_int = _round_nearest(y_pred[i])
            abs_err = abs(y_pred[i] - y_true[i])
            rel_err = abs_err / max(1.0, abs(y_true[i]))
            w.writerow([n[i], y_true[i], baseline_vals[i], y_pred[i], pred_int, abs_err, rel_err])
    print(f"[SAVE] Fertig: {path}")

def save_predictions_full(n, y_true, y_pred, baseline_vals, save_prefix: str, progress: bool = True):
    path = f"{save_prefix}_predictions_full.csv"
    print(f"[SAVE] Schreibe volle Vorhersagen → {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "prime_true", "baseline", "pred", "pred_int", "abs_err", "rel_err"])
        rng = range(len(n))
        if progress:
            rng = tqdm(rng, desc="Save full CSV", dynamic_ncols=True, leave=False)
        for i in rng:
            pred_int = _round_nearest(y_pred[i])
            abs_err = abs(y_pred[i] - y_true[i])
            rel_err = abs_err / max(1.0, abs(y_true[i]))
            w.writerow([n[i], y_true[i], baseline_vals[i], y_pred[i], pred_int, abs_err, rel_err])
    print(f"[SAVE] Fertig: {path}")


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser(description="Prime SR – Val/HUBER/π-Seeds(/7)/Smooth/Minibatch/ConstTune/SieveFast")
    # „Stark“-Defaults
    parser.add_argument("--N", type=int, default=60_000)
    parser.add_argument("--pop", type=int, default=900)
    parser.add_argument("--generations", type=int, default=90)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--save-prefix", type=str, default="prime_sr_strong")
    parser.add_argument("--save-full", action="store_true")
    parser.add_argument("--progress", type=int, default=1)
    parser.add_argument("--ckpt", type=int, default=10)
    parser.add_argument("--mode", type=str, default="residual", choices=["residual", "multiplicative"])

    # Feintuning & Anti-Bloat
    parser.add_argument("--init-depth-min", type=int, default=2)
    parser.add_argument("--init-depth-max", type=int, default=7)
    parser.add_argument("--mut-depth-min", type=int, default=1)
    parser.add_argument("--mut-depth-max", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=600)
    parser.add_argument("--tournsize", type=int, default=9)
    parser.add_argument("--parsimony", type=float, default=0.0002)

    # (1) Validation-Split
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Anteil Validation (0..0.5 sinnvoll)")

    # (2) Huber-Delta & (5) Glättung (Krümmungsstrafe)
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--smoothness", type=float, default=0.001)

    # (6) Mini-Batch Val
    parser.add_argument("--batch-val-count", type=int, default=0, help="0=volle Val; sonst zufällige Val-Teilmengen")
    parser.add_argument("--batch-repeats", type=int, default=1, help="Anzahl wiederholter Val-Minibatches")

    # (1) Mischung Train/Val für Fitness
    parser.add_argument("--fitness-mix", type=float, default=1.0, help="Gewicht Val in [0..1]; 1.0 = nur Val")

    # (3) Residual-Skalierung
    parser.add_argument("--residual-scale", type=str, default="none", choices=["none", "ln", "sqrt_n"])

    # (4) Konstanten-Finetuning
    parser.add_argument("--const-tune-iters", type=int, default=150, help="0=aus; sonst Hillclimb-Iterationen")

    # (7) Checkpoint-Top-K + Early-Stop + Formelzeilenlimit
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--print-topk", type=int, default=3)
    parser.add_argument("--early-stop", type=int, default=0)
    parser.add_argument("--max-formula-chars", type=int, default=1000)

    # (π-Seeding)
    parser.add_argument("--seed-pi-count", type=int, default=3, help="Wieviele π/7-Startformeln injizieren (0=aus)")

    args = parser.parse_args()
    set_seeds(args.seed)
    progress = bool(args.progress)

    # Datensatz
    n_idx, p_true, baseline_vals, rows = make_dataset(args.N, args.save_prefix, progress=progress)

    # Features
    X = [(r["ln"], r["lnln"], r["invln"], r["sqrt_n"], r["ones"]) for r in rows]
    y_true_primes = [float(v) for v in p_true]

    # Residual-Skalierung vorbereiten (3)
    scale_all: Optional[List[float]]
    if args.residual-scale if False else None:  # placeholder to appease syntax highlighters
        pass  # (will never execute)
    if args.residual_scale == "ln":
        scale_all = [max(1.0, rows[i]["ln"]) for i in range(args.N)]
    elif args.residual_scale == "sqrt_n":
        scale_all = [max(1.0, rows[i]["sqrt_n"]) for i in range(args.N)]
    else:
        scale_all = None

    # (1) Validation-Split (Block am Ende)
    split = int((1.0 - max(0.0, min(0.5, args.val_ratio))) * args.N)
    split = max(2, min(args.N - 2, split))  # Sicherheitskappen
    idx_train = list(range(split))
    idx_val   = list(range(split, args.N))

    X_train = [X[i] for i in idx_train];  y_train = [y_true_primes[i] for i in idx_train]
    B_train = [baseline_vals[i] for i in idx_train]
    S_train = None if scale_all is None else [scale_all[i] for i in idx_train]

    X_val = [X[i] for i in idx_val];      y_val = [y_true_primes[i] for i in idx_val]
    B_val = [baseline_vals[i] for i in idx_val]
    S_val = None if scale_all is None else [scale_all[i] for i in idx_val]

    # Baseline-Reports
    metrics_report(p_true, baseline_vals, prefix="Baseline (gesamter Bereich)")
    # Val-spezifisch
    # einfache Baseline-Pred = baseline_vals
    print(f"[INFO] Val-Bereich: n in [{idx_val[0]+1} .. {idx_val[-1]+1}] (Anteil {len(idx_val)/args.N:.2%})")

    # Toolbox mit Evaluator (2,5,6,1,3 zusammengeführt)
    toolbox, pset, set_batch_params, BASE_MAE_VAL = make_toolbox_with_eval(
        X_train=X_train, y_train=y_train, B_train=B_train,
        X_val=X_val, y_val=y_val, B_val=B_val,
        scale_train=S_train, scale_val=S_val,
        pred_mode=args.mode,
        residual_scale=args.residual_scale,
        tournsize=args.tournsize,
        huber_delta=args.huber_delta,
        fitness_mix=max(0.0, min(1.0, args.fitness_mix)),
        smoothness_coef=max(0.0, args.smoothness),
        parsimony=args.parsimony,
        init_depth_min=args.init_depth_min,
        init_depth_max=args.init_depth_max,
        mut_depth_min=args.mut_depth_min,
        mut_depth_max=args.mut_depth_max,
        max_size=args.max_size
    )

    # Mini-Batch-Parameter setzen (6)
    set_batch_params(len(X_val), args.batch_val_count, args.batch_repeats)

    # π-Seeds vorbereiten (inkl. /7.0) (π)
    seed_inds = build_pi_seeds(pset, args.seed_pi_count) if args.seed_pi_count > 0 else []

    # Multiprocessing (Windows/Android: lieber aus)
    if args.jobs and args.jobs > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=args.jobs)
        toolbox.register("map", pool.map)
    else:
        pool = None

    # Evolution (mit Checkpoints & Pickle) (7)
    pop, hof, stats = gp_evolve_with_progress(
        toolbox=toolbox, pset=pset,
        pop_size=args.pop, generations=args.generations,
        cxpb=0.7, mutpb=0.25, progress=progress, ckpt_every=args.ckpt,
        elite=args.elite, print_topk=args.print_topk,
        early_stop=args.early_stop, max_formula_chars=args.max_formula_chars,
        seed_inds=seed_inds,
        set_batch_params=set_batch_params,
        batch_val_count=args.batch_val_count,
        batch_repeats=args.batch_repeats,
        save_prefix=args.save_prefix
    )

    best = hof[0]

    # (4) Konstanten-Finetuning (Train) – ohne SciPy
    if args.const_tune_iters > 0:
        print(f"[TUNE] Konstanten-Finetuning (Hillclimb, iters={args.const_tune_iters}) …")
        best = tune_constants_local_hillclimb(
            copy.deepcopy(best),
            toolbox,
            X_train, y_train, B_train, S_train,
            pred_mode=args.mode,
            iters=args.const_tune_iters
        )
        # Nach Tuning Textformel/Checkpoint speichern
        _save_formula(best, f"{args.save_prefix}_post_tune_formula.txt")
        try:
            with open(f"{args.save_prefix}_post_tune.pkl", "wb") as fh:
                pickle.dump({"ind": best, "pset_name": pset.name}, fh)
            print(f"[SAVE] Pickle gespeichert → {args.save_prefix}_post_tune.pkl")
        except Exception:
            pass

    # Finale Formel
    best_infix = _safe_formula_text(best)
    print("\n===== BESTE GEFUNDENE FORMEL (VALIDIERT) =====")
    if args.mode == "residual":
        print("Residual f(ln, lnln, invln, sqrt_n, one) =")
        print(best_infix)
        print("\nBaseline(n) = n*(ln n + ln ln n - 1 + (ln ln n - 2)/ln n)")
        if args.residual_scale == "none":
            print("=> p_hat(n) = Baseline(n) + f( ln n, ln ln n, 1/ln n, sqrt(n), 1 )")
        else:
            print(f"=> p_hat(n) = Baseline(n) + f( ln n, ln ln n, 1/ln n, sqrt(n), 1 ) * scale(n)  # scale = {args.residual_scale}")
    else:
        print("Multiplikativ f(ln, lnln, invln, sqrt_n, one) =")
        print(best_infix)
        if args.residual_scale == "none":
            print("\n=> p_hat(n) = Baseline(n) * ( 1 + f( ln n, ln ln n, 1/ln n, sqrt(n), 1 ) )")
        else:
            print(f"\n=> p_hat(n) = Baseline(n) * ( 1 + f( ln n, ln ln n, 1/ln n, sqrt(n), 1 ) * scale(n) )  # scale = {args.residual_scale}")
    print("==============================================\n")
    _save_formula(best, f"{args.save_prefix}_best_formula.txt")
    try:
        with open(f"{args.save_prefix}_best.pkl", "wb") as fh:
            pickle.dump({"ind": best, "pset_name": pset.name}, fh)
        print(f"[SAVE] Pickle gespeichert → {args.save_prefix}_best.pkl")
    except Exception:
        pass

    # Vorhersage (gesamter Bereich) & Metriken
    print("[PRED] Wende bestes Modell auf alle Daten an …")
    func = toolbox.compile(best)

    def predict_all():
        out = []
        if args.mode == "residual":
            if scale_all is None:
                for i in range(len(X)): out.append(baseline_vals[i] + float(func(*X[i])))
            else:
                for i in range(len(X)): out.append(baseline_vals[i] + float(func(*X[i])) * scale_all[i])
        else:
            if scale_all is None:
                for i in range(len(X)): out.append(baseline_vals[i] * (1.0 + float(func(*X[i]))))
            else:
                for i in range(len(X)): out.append(baseline_vals[i] * (1.0 + float(func(*X[i])) * scale_all[i]))
        return out

    it = range(len(X))
    if progress:
        it = tqdm(it, desc="Predict", dynamic_ncols=True)
    y_pred_all: List[float] = []
    for i in it:
        pass  # (nur für animierte Progressbar)
    # Ohne Overhead – einmalig rufen:
    y_pred_all = predict_all()

    # Metriken: Gesamt / Train / Val
    metrics_report(p_true, y_pred_all, prefix="GESAMT")
    y_pred_train = y_pred_all[:len(X_train)]
    y_pred_val   = y_pred_all[len(X_train):]
    metrics_report([int(v) for v in y_train], y_pred_train, prefix="TRAIN (fit)")
    metrics_report([int(v) for v in y_val],   y_pred_val,   prefix="VAL (fit/selection)")

    # Saves
    save_predictions(n_idx, p_true, y_pred_all, baseline_vals, args.save_prefix, k=5000, progress=progress)
    if args.save_full:
        save_predictions_full(n_idx, p_true, y_pred_all, baseline_vals, args.save_prefix, progress=progress)

    if 'pool' in locals() and pool:
        pool.close(); pool.join()

    print("\n==== Zusammenfassung ====")
    print(f"Dataset: N={args.N:,}  | Val-Ratio: {args.val_ratio:.2f}")
    print(f"Pipeline: Baseline → GP ({args.mode}, Residual-Scale={args.residual_scale})")
    print(f"Fitness: Val-gewichtet={args.fitness_mix:.2f}, Huber-Δ={args.huber_delta}, Smooth={args.smoothness}")
    print(f"Minibatch Val: count={args.batch_val_count}, repeats={args.batch_repeats}")
    print(f"GP: pop={args.pop}, generations={args.generations}, best size={len(best)}")
    print(f"π-Seeds injiziert: {args.seed_pi_count} (mit '/7.0')")
    print("=========================\n")


if __name__ == "__main__":
    main()
