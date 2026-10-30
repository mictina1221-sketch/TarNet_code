import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
TARGET_WORD = "公民"
LR, K_FOLDS = 1e-5, 2
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT_DIR = os.path.join(DESKTOP, "TarNet_Research_Results", f"Topic_{TARGET_WORD}_v2")
DATA_FILE = os.path.join(DESKTOP, "6th_to_10th.csv")
HIDDEN_DIR = os.path.join(DESKTOP, "全文本", "hidden_states_全文本_1")
HIDDEN_PREFIX = "hidden_last_"
LOG_FILE = os.path.join(DESKTOP, "Centralized_Causal_Results_v2.csv")
LIB_PATH = r"C:\Users\gg\Desktop\gemini_cli\gpi_pack-main\src"

# CSV column names (must match the data file)
TERM_COL, SESSION_COL, TEXT_COL, OUTCOME_COL, TIME_COL, CAUCUS_COL = (
    "屆期", "會期", "關係文書", "三讀", "提案時間", "黨團提案")

os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, LIB_PATH)
from gpi_pack.TarNet import estimate_k_ate, load_hiddens

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def count_nonnull(df, prefix):
    cols = [c for c in df.columns if re.fullmatch(rf"{prefix}\d+", c)]
    return df[cols].notna().sum(axis=1).values


def run_ate(R, Y, T, C, **kw):
    # Shared TarNet settings
    return estimate_k_ate(
        R=R, Y=Y, T=T, C=C, K=K_FOLDS, valid_perc=0.2,
        architecture_z=[128, 64], architecture_y=[256, 128, 1],
        dropout=0.3, nepoch=150, lr=LR, patience=5, verbose=False, **kw)


def main():
    # Load data and build treatment / outcome
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", low_memory=False)
    df["T"] = df[TEXT_COL].fillna("").apply(lambda x: int(TARGET_WORD in x))
    df["Y"] = df[OUTCOME_COL].fillna(0).astype(float)
    T, Y = df["T"].values.astype(int), df["Y"].values
    n_a, n_b = int(T.sum()), int((T == 0).sum())
    print(f"N={len(df)} (T=1: {n_a}, T=0: {n_b}), topic: {TARGET_WORD}")

    # Descriptive stats
    desc = (df[df["T"] == 1].groupby(TERM_COL)
            .agg(n=("Y", "count"), pass_rate=("Y", "mean"), pass_se=("Y", "sem"))
            .reset_index())
    print(desc.to_string(index=False))
    desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"), index=False, encoding="utf-8-sig")

    # Covariates C
    df["year"] = pd.to_datetime(df[TIME_COL], errors="coerce").dt.year.fillna(2010)
    df["n_prop"] = count_nonnull(df, "提案人")
    df["n_sec"] = count_nonnull(df, "連署人")
    extra = ["year", "n_prop", "n_sec"]
    if CAUCUS_COL in df.columns:
        df["is_caucus_proposal"] = df[CAUCUS_COL].fillna(0).astype(float)
        extra.append("is_caucus_proposal")
    C_df = pd.concat([
        pd.get_dummies(df[TERM_COL], prefix="term").astype(float),
        pd.get_dummies(df[SESSION_COL], prefix="sess").astype(float),
        df[extra]], axis=1)
    C_np = C_df.values.astype(np.float32)

    # Hidden states
    R_np = load_hiddens(directory=HIDDEN_DIR, hidden_list=df["Global_UID"].astype(int).tolist(),
                        prefix=HIDDEN_PREFIX, device="cpu")

    # Global ATE
    base_rate = T.mean()
    ate, se = run_ate(R_np, Y, T, C_np, bn=True, cluster=df[TERM_COL].tolist(), plot_propensity=True)
    lo, hi = ate - 1.96 * se, ate + 1.96 * se
    print(f"Global ATE = {ate:.4f} +/- {se:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

    # Term-specific ATE
    key = lambda x: int(re.search(r"\d+", str(x)).group()) if re.search(r"\d+", str(x)) else 999
    rows = []
    for term in sorted(df[TERM_COL].dropna().unique(), key=key):
        idx = np.where((df[TERM_COL] == term).values)[0]
        T_t, Y_t = T[idx], Y[idx]
        n_t1, n_t0 = int(T_t.sum()), int((T_t == 0).sum())
        if n_t1 == 0 or n_t0 == 0:
            continue
        # Drop term dummies and constant columns within a term
        C_t = C_df.iloc[idx].drop(columns=[c for c in C_df.columns if str(c).startswith("term_")])
        C_t = C_t.loc[:, C_t.nunique() > 1]
        try:
            a, s = run_ate(R_np[idx], Y_t, T_t, C_t.values.astype(np.float32),
                           bn=False, cluster=None, plot_propensity=False)
        except Exception as e:
            print(f"Term {term} failed: {e}")
            continue
        print(f"Term {term}: N={len(idx)}, T1={n_t1}, ATE={a:.4f} +/- {s:.4f}")
        rows.append({"Topic": TARGET_WORD, "Term": term, "N": len(idx), "T1": n_t1, "T0": n_t0,
                     "T_rate": T_t.mean(), "Y_mean_T0": Y_t[T_t == 0].mean(),
                     "Y_mean_T1": Y_t[T_t == 1].mean(), "ATE": a, "SE": s,
                     "CI_lower": a - 1.96 * s, "CI_upper": a + 1.96 * s})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{TARGET_WORD}_ATE_by_term.csv"),
                                  index=False, encoding="utf-8-sig")

    # Append global result to log
    row = pd.DataFrame([{
        "Topic": TARGET_WORD, "n_samples": len(df), "n_a": n_a, "n_b": n_b,
        "base_rate": base_rate, "ate": ate, "se": se, "ci_lower": lo, "ci_upper": hi,
        "K_folds": K_FOLDS, "cluster_by": TERM_COL,
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}])
    row.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE),
               index=False, encoding="utf-8-sig")
    print(f"Saved to {OUT_DIR} and {LOG_FILE}")


if __name__ == "__main__":
    main()
