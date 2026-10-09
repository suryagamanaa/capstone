"""
Evaluasi Kinerja Metode MC, QMC, dan RQMC pada Penentuan Harga Opsi Lookback
=============================================================================
Repositori ini mereproduksi seluruh tabel dan gambar pada paper:

  Tabel 3 : Nilai Analitik Opsi Lookback (continuous monitoring)
  Tabel 4 : Harga Opsi Lookback Floating Strike vs N
  Tabel 5 : Harga Opsi Lookback Fixed Strike vs N
  Tabel 6 : Galat Relatif Lookback Floating Strike vs N
  Tabel 7 : Galat Relatif Lookback Fixed Strike vs N
  Tabel 8 : Standard Error Lookback Floating Strike vs N
  Tabel 9 : Standard Error Lookback Fixed Strike vs N
  Gambar 1-4  : Konvergensi Harga (Floating Call/Put, Fixed Call/Put)
  Gambar 9-12 : Standard Error  (Floating Call/Put, Fixed Call/Put)

Metode yang dibandingkan:
  1. Monte Carlo (MC)              - pseudorandom
  2. Quasi-Monte Carlo Sobol       - scramble=False
  3. Quasi-Monte Carlo Halton      - scramble=False
  4. Randomized QMC Sobol          - scramble=True
  5. Randomized QMC Halton         - scramble=True

Model: Geometric Brownian Motion (GBM), discrete monitoring.
Referensi formula analitik:
  - Goldman, Sosin, Gatto (1979)  -> floating strike
  - Conze & Viswanathan (1991)    -> fixed strike
"""

import os
import warnings
from datetime import datetime

import numpy as np
from scipy.stats import qmc, norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. KONFIGURASI GLOBAL
# ============================================================
MASTER_SEED = 42
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter model (lihat Tabel 1 paper)
S0      = 100.0
K_fixed = 100.0
r       = 0.05
q       = 0.0
sigma   = 0.20
T       = 1.0
n_steps = 252

# Parameter simulasi
N_list = [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 640_000]
M      = 30   # jumlah replikasi (sesuai paper: R = 30)

# Warna konsisten untuk setiap metode (dipakai di semua grafik)
COLORS = {
    "MC":          "#d63031",
    "QMC_Sobol":   "#fdcb6e",
    "QMC_Halton":  "#e17055",
    "RQMC_Sobol":  "#0984e3",
    "RQMC_Halton": "#00b894",
}
METHODS_ORDER = ["MC", "QMC_Sobol", "QMC_Halton", "RQMC_Sobol", "RQMC_Halton"]
MARKERS       = {"MC": "o", "QMC_Sobol": "s", "QMC_Halton": "^",
                 "RQMC_Sobol": "D", "RQMC_Halton": "v"}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "savefig.bbox":    "tight",
})


# ============================================================
# 2. FORMULA ANALITIK (TRUE VALUE) - Tabel 3
# ============================================================
def floating_lookback_call_analytic(S0, r, q, sigma, T):
    """Floating Strike Lookback CALL - Goldman-Sosin-Gatto (1979)."""
    b  = r - q
    a1 = (b + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    a2 = a1 - sigma * np.sqrt(T)

    if abs(b) > 1e-15:
        coef = sigma**2 / (2 * b)
        price = (
            S0 * np.exp((b - r) * T) * norm.cdf(a1)
            - S0 * np.exp(-r * T) * norm.cdf(a2)
            + S0 * np.exp(-r * T) * coef
              * (norm.cdf(-a1 + (2 * b / sigma) * np.sqrt(T))
                 - np.exp(b * T) * norm.cdf(-a1))
        )
    else:
        price = (
            S0 * np.exp(-r * T) * norm.cdf(a1)
            - S0 * np.exp(-r * T) * norm.cdf(a2)
            + S0 * np.exp(-r * T) * sigma * np.sqrt(T)
              * (norm.pdf(a1) + a1 * (norm.cdf(a1) - 1))
        )
    return max(price, 0.0)


def floating_lookback_put_analytic(S0, r, q, sigma, T):
    """Floating Strike Lookback PUT - Goldman-Sosin-Gatto (1979)."""
    b  = r - q
    b1 = (b + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    b2 = b1 - sigma * np.sqrt(T)

    if abs(b) > 1e-15:
        coef = sigma**2 / (2 * b)
        price = (
            S0 * np.exp(-r * T) * norm.cdf(-b2)
            - S0 * np.exp((b - r) * T) * norm.cdf(-b1)
            + S0 * np.exp(-r * T) * coef
              * (np.exp(b * T) * norm.cdf(b1)
                 - norm.cdf(b1 - (2 * b / sigma) * np.sqrt(T)))
        )
    else:
        price = (
            S0 * np.exp(-r * T) * norm.cdf(-b2)
            - S0 * np.exp(-r * T) * norm.cdf(-b1)
            + S0 * np.exp(-r * T) * sigma * np.sqrt(T)
              * (norm.pdf(b1) + b1 * norm.cdf(b1))
        )
    return max(price, 0.0)


def fixed_lookback_call_analytic(S0, r, q, sigma, T, K):
    """Fixed Strike Lookback CALL - Conze & Viswanathan (1991)."""
    S_max = S0
    b     = r - q

    if sigma <= 0 or T <= 0:
        return max(0.0, S_max - K) if T <= 0 else 0.0

    coef = sigma**2 / (2 * b) if abs(b) > 1e-15 else 0.0

    if K >= S_max:                       # S_L = K
        d1 = (np.log(S0 / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = (
            S0 * np.exp((b - r) * T) * norm.cdf(d1)
            - K  * np.exp(-r * T) * norm.cdf(d2)
            - coef * S0 * np.exp(-r * T) * (S0 / K)**(2 * b / sigma**2)
              * norm.cdf(d1 - (2 * b / sigma) * np.sqrt(T))
            + coef * S0 * np.exp(-r * T) * np.exp(b * T) * norm.cdf(d1)
        )
    else:                                # S_L = S_max
        e1 = (b + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
        e2 = e1 - sigma * np.sqrt(T)

        price = (
            np.exp(-r * T) * (S_max - K)
            + S0    * np.exp((b - r) * T) * norm.cdf(e1)
            - S_max * np.exp(-r * T) * norm.cdf(e2)
            - coef * S0 * np.exp(-r * T) * norm.cdf(e1 - (2 * b / sigma) * np.sqrt(T))
            + coef * S0 * np.exp(-r * T) * np.exp(b * T) * norm.cdf(e1)
        )
    return max(price, 0.0)


def fixed_lookback_put_analytic(S0, r, q, sigma, T, K):
    """Fixed Strike Lookback PUT - Conze & Viswanathan (1991)."""
    S_min = S0
    b     = r - q

    if sigma <= 0 or T <= 0:
        return max(0.0, K - S_min) if T <= 0 else 0.0

    coef = sigma**2 / (2 * b) if abs(b) > 1e-15 else 0.0

    if K <= S_min:                       # S_L = K
        d1 = (np.log(S0 / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = (
            K  * np.exp(-r * T) * norm.cdf(-d2)
            - S0 * np.exp((b - r) * T) * norm.cdf(-d1)
            + coef * S0 * np.exp(-r * T) * (S0 / K)**(2 * b / sigma**2)
              * norm.cdf(-d1 + (2 * b / sigma) * np.sqrt(T))
            - coef * S0 * np.exp(-r * T) * np.exp(b * T) * norm.cdf(-d1)
        )
    else:                                # S_L = S_min
        f1 = (b + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
        f2 = f1 - sigma * np.sqrt(T)

        price = (
            np.exp(-r * T) * (K - S_min)
            - S0    * np.exp((b - r) * T) * norm.cdf(-f1)
            - S_min * np.exp(-r * T) * norm.cdf(-f2)
            + coef * S0 * np.exp(-r * T) * norm.cdf(-f1 + (2 * b / sigma) * np.sqrt(T))
            - coef * S0 * np.exp(-r * T) * np.exp(b * T) * norm.cdf(-f1)
        )
    return max(price, 0.0)


# ============================================================
# 3. SIMULASI GBM & PAYOFF
# ============================================================
def simulate_gbm_paths(S0, r, sigma, T, n_steps, N, random_draws):
    """Simulasi lintasan GBM diskrit (Persamaan 5 di paper)."""
    dt        = T / n_steps
    drift     = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    log_returns = drift + diffusion * random_draws
    log_prices  = np.log(S0) + np.cumsum(log_returns, axis=1)

    S = np.empty((N, n_steps + 1))
    S[:, 0]  = S0
    S[:, 1:] = np.exp(log_prices)
    return S


def payoff_floating_call(S_paths, r, T):
    """Payoff (6): e^{-rT} (S_T - m_{0,T})."""
    return np.exp(-r * T) * (S_paths[:, -1] - S_paths[:, 1:].min(axis=1))


def payoff_floating_put(S_paths, r, T):
    """Payoff (7): e^{-rT} (M_{0,T} - S_T)."""
    return np.exp(-r * T) * (S_paths[:, 1:].max(axis=1) - S_paths[:, -1])


def payoff_fixed_call(S_paths, r, T, K):
    """Payoff (8): e^{-rT} max(M_{0,T} - K, 0)."""
    return np.exp(-r * T) * np.maximum(S_paths[:, 1:].max(axis=1) - K, 0.0)


def payoff_fixed_put(S_paths, r, T, K):
    """Payoff (9): e^{-rT} max(K - m_{0,T}, 0)."""
    return np.exp(-r * T) * np.maximum(K - S_paths[:, 1:].min(axis=1), 0.0)


# ============================================================
# 4. GENERATOR BILANGAN ACAK (5 METODE)
# ============================================================
def _uniform_to_normal(u):
    u = np.clip(u, 1e-15, 1 - 1e-15)
    return norm.ppf(u)


def generate_mc(N, n_steps, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, n_steps))


def generate_qmc_sobol(N, n_steps, seed):
    sampler = qmc.Sobol(d=n_steps, scramble=False, seed=seed)
    return _uniform_to_normal(sampler.random(N))


def generate_qmc_halton(N, n_steps, seed):
    sampler = qmc.Halton(d=n_steps, scramble=False, seed=seed)
    return _uniform_to_normal(sampler.random(N))


def generate_rqmc_sobol(N, n_steps, seed):
    sampler = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
    return _uniform_to_normal(sampler.random(N))


def generate_rqmc_halton(N, n_steps, seed):
    sampler = qmc.Halton(d=n_steps, scramble=True, seed=seed)
    return _uniform_to_normal(sampler.random(N))


GENERATORS = {
    "MC":          generate_mc,
    "QMC_Sobol":   generate_qmc_sobol,
    "QMC_Halton":  generate_qmc_halton,
    "RQMC_Sobol":  generate_rqmc_sobol,
    "RQMC_Halton": generate_rqmc_halton,
}

# Offset seed per metode (deterministik & reproducible)
SEED_OFFSETS = {
    "MC": 0, "QMC_Sobol": 100, "QMC_Halton": 200,
    "RQMC_Sobol": 1000, "RQMC_Halton": 1100,
}


# ============================================================
# 5. RUNNER SIMULASI PER TIPE OPSI
# ============================================================
def run_simulation(payoff_fn, generator_func, N, n_steps, S0, r, sigma, T,
                   M, base_seed, extra=None):
    """
    Jalankan M replikasi, kembalikan (mean_price, std_error, all_prices).
    extra = argumen tambahan untuk payoff_fn (mis. K untuk fixed strike).
    """
    extra = extra or {}
    prices = np.empty(M)

    for rep in range(M):
        seed         = base_seed + rep * 10_000
        random_draws = generator_func(N, n_steps, seed)
        S_paths      = simulate_gbm_paths(S0, r, sigma, T, n_steps, N, random_draws)
        prices[rep]  = payoff_fn(S_paths, r, T, **extra).mean()

    return prices.mean(), prices.std(ddof=1) / np.sqrt(M), prices


# ============================================================
# 6. LOOP UTAMA: 5 METODE x len(N_list)
# ============================================================
def run_all_simulations(payoff_fn, label, extra=None):
    """Jalankan 5 metode untuk semua N_list. Return dict hasil."""
    results = {m: {"prices": [], "se": []} for m in METHODS_ORDER}

    print(f"\n{'='*80}\n  SIMULASI: {label}\n{'='*80}")
    for i, N in enumerate(N_list):
        print(f"  N = {N:>8,}", end="  ")
        for m in METHODS_ORDER:
            seed = MASTER_SEED + i * 1_000 + SEED_OFFSETS[m]
            mean_p, se, _ = run_simulation(
                payoff_fn, GENERATORS[m], N, n_steps, S0, r, sigma, T,
                M, seed, extra=extra
            )
            results[m]["prices"].append(mean_p)
            results[m]["se"].append(se)
            print(".", end="", flush=True)
        print(" done")

    return results


# ============================================================
# 7. CETAK TABEL (Tabel 4-9)
# ============================================================
def _fmt_row(values, widths, formats):
    return " | ".join(f"{f.format(v):>{w}}"
                      for v, w, f in zip(values, widths, formats))


def _print_header(headers, widths):
    line = " | ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))


def print_tables(results, analytic_price, label):
    """
    Cetak 4 tabel untuk satu tipe opsi:
      - Harga vs N                (Tabel 4 / 5)
      - Galat Relatif vs N        (Tabel 6 / 7)
      - Standard Error vs N       (Tabel 8 / 9)
    """
    # ---------- Tabel Harga ----------
    print(f"\n{'='*105}")
    print(f"TABEL HARGA - {label}")
    print(f"  Reference (Analytic Continuous): {analytic_price:.4f}")
    print(f"{'='*105}")
    widths  = [10] + [14] * 5
    headers = ["N"] + METHODS_ORDER
    _print_header(headers, widths)
    for i, N in enumerate(N_list):
        vals = [N] + [results[m]["prices"][i] for m in METHODS_ORDER]
        fmts = ["{:,}"] + ["{:.4f}"] * 5
        print(_fmt_row(vals, widths, fmts))

    # ---------- Tabel Galat Relatif ----------
    print(f"\n{'='*105}")
    print(f"TABEL GALAT RELATIF (%) - {label}")
    print(f"{'='*105}")
    headers = ["N"] + [f"{m} RE%" for m in METHODS_ORDER]
    _print_header(headers, widths)
    for i, N in enumerate(N_list):
        vals = [N] + [
            abs(results[m]["prices"][i] - analytic_price) / analytic_price * 100
            for m in METHODS_ORDER
        ]
        fmts = ["{:,}"] + ["{:.2f}"] * 5
        print(_fmt_row(vals, widths, fmts))

    # ---------- Tabel Standard Error ----------
    print(f"\n{'='*105}")
    print(f"TABEL STANDARD ERROR - {label}")
    print(f"{'='*105}")
    headers = ["N"] + [f"{m} SE" for m in METHODS_ORDER]
    _print_header(headers, widths)
    for i, N in enumerate(N_list):
        vals = [N] + [results[m]["se"][i] for m in METHODS_ORDER]
        fmts = ["{:,}"] + ["{:.2e}"] * 5
        print(_fmt_row(vals, widths, fmts))
    print(f"{'='*105}")


# ============================================================
# 8. GRAFIK
# ============================================================
def plot_convergence(results, analytic_price, title, save_name):
    """Gambar 1-4: konvergensi harga vs N."""
    fig, ax = plt.subplots(figsize=(12, 7))
    N_arr = np.array(N_list)

    ax.axhline(analytic_price, color="black", linewidth=2.2, alpha=0.85,
               label=f"Analytic (Continuous): {analytic_price:.4f}")

    for m in METHODS_ORDER:
        prices = np.array(results[m]["prices"])
        ses    = np.array(results[m]["se"])
        ax.plot(N_arr, prices, f"-{MARKERS[m]}", color=COLORS[m],
                linewidth=2, markersize=7,
                label=m.replace("_", " "), alpha=0.9)
        if m in ("MC", "RQMC_Sobol", "RQMC_Halton"):
            ax.fill_between(N_arr, prices - 1.96 * ses, prices + 1.96 * ses,
                            color=COLORS[m], alpha=0.10)

    ax.set_xscale("log")
    ax.set_xlabel("Banyak Iterasi (N) [Log Scale]")
    ax.set_ylabel("Harga Opsi")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name))
    plt.close()
    print(f"  [saved] {save_name}")


def plot_standard_error(results, title, save_name):
    """Gambar 9-12: standard error vs N."""
    fig, ax = plt.subplots(figsize=(12, 7))
    N_arr = np.array(N_list)

    for m in METHODS_ORDER:
        ses   = np.array(results[m]["se"])
        valid = ses > 1e-15
        if valid.any():
            ax.plot(N_arr[valid], ses[valid], f"-{MARKERS[m]}", color=COLORS[m],
                    linewidth=2, markersize=7,
                    label=m.replace("_", " "), alpha=0.9)

    # garis referensi O(1/sqrt(N))
    ref = results["MC"]["se"][0] * np.sqrt(N_list[0]) / np.sqrt(N_arr)
    ax.plot(N_arr, ref, "--", color="gray", linewidth=1.5, alpha=0.7,
            label=r"$O(1/\sqrt{N})$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Banyak Iterasi (N) [Log Scale]")
    ax.set_ylabel("Standard Error [Log Scale]")
    ax.set_title(f"Standard Error - {title}", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name))
    plt.close()
    print(f"  [saved] {save_name}")


# ============================================================
# 9. MAIN
# ============================================================
def main():
    print("=" * 80)
    print(" EVALUASI MC, QMC, RQMC PADA HARGA OPSI LOOKBACK")
    print(f" Tanggal: {datetime.now():%d %B %Y}")
    print("=" * 80)
    print(f" Parameter: S0={S0}, K={K_fixed}, r={r}, q={q}, "
          f"sigma={sigma}, T={T}, n_steps={n_steps}")
    print(f" N_list   : {N_list}")
    print(f" M        : {M}")
    print("=" * 80)

    # ---------- Nilai Analitik (Tabel 3) ----------
    ana_float_call = floating_lookback_call_analytic(S0, r, q, sigma, T)
    ana_float_put  = floating_lookback_put_analytic (S0, r, q, sigma, T)
    ana_fixed_call = fixed_lookback_call_analytic   (S0, r, q, sigma, T, K_fixed)
    ana_fixed_put  = fixed_lookback_put_analytic    (S0, r, q, sigma, T, K_fixed)

    print("\nNILAI ANALITIK (Continuous Monitoring) - Tabel 3")
    print(f"  Floating Strike CALL : {ana_float_call:.4f}")
    print(f"  Floating Strike PUT  : {ana_float_put:.4f}")
    print(f"  Fixed Strike CALL    : {ana_fixed_call:.4f}")
    print(f"  Fixed Strike PUT     : {ana_fixed_put:.4f}")

    # ---------- BAGIAN 1: Floating CALL ----------
    res_fc = run_all_simulations(payoff_floating_call, "Floating CALL")
    print_tables(res_fc, ana_float_call, "Floating CALL")
    plot_convergence(res_fc, ana_float_call,
                     "Konvergensi Harga - Floating Strike Lookback CALL",
                     "gambar_1_konvergensi_floating_call.png")
    plot_standard_error(res_fc, "Floating CALL",
                        "gambar_9_se_floating_call.png")

    # ---------- BAGIAN 2: Floating PUT ----------
    res_fp = run_all_simulations(payoff_floating_put, "Floating PUT")
    print_tables(res_fp, ana_float_put, "Floating PUT")
    plot_convergence(res_fp, ana_float_put,
                     "Konvergensi Harga - Floating Strike Lookback PUT",
                     "gambar_2_konvergensi_floating_put.png")
    plot_standard_error(res_fp, "Floating PUT",
                        "gambar_10_se_floating_put.png")

    # ---------- BAGIAN 3: Fixed CALL ----------
    res_xc = run_all_simulations(payoff_fixed_call, "Fixed CALL", extra={"K": K_fixed})
    print_tables(res_xc, ana_fixed_call, "Fixed CALL")
    plot_convergence(res_xc, ana_fixed_call,
                     "Konvergensi Harga - Fixed Strike Lookback CALL",
                     "gambar_3_konvergensi_fixed_call.png")
    plot_standard_error(res_xc, "Fixed CALL",
                        "gambar_11_se_fixed_call.png")

    # ---------- BAGIAN 4: Fixed PUT ----------
    res_xp = run_all_simulations(payoff_fixed_put, "Fixed PUT", extra={"K": K_fixed})
    print_tables(res_xp, ana_fixed_put, "Fixed PUT")
    plot_convergence(res_xp, ana_fixed_put,
                     "Konvergensi Harga - Fixed Strike Lookback PUT",
                     "gambar_4_konvergensi_fixed_put.png")
    plot_standard_error(res_xp, "Fixed PUT",
                        "gambar_12_se_fixed_put.png")

    print(f"\nSemua output disimpan di folder: '{OUTPUT_DIR}/'")
    print("SELESAI.")


if __name__ == "__main__":
    main()