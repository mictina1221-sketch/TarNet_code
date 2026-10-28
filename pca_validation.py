# Embedding validation: PCA of text embeddings by treatment and legislative term

import os
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

# ---------------- Configuration ----------------
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
CSV_FILE = os.path.join(DESKTOP, "6th_to_10th.csv")
HIDDEN_DIR = os.path.join(DESKTOP, "全文本", "hidden_states_全文本_1")
HIDDEN_PREFIX = "hidden_last_"
OUTPUT_DIR = os.path.join(DESKTOP, "TarNet_Research_Results", "Embedding_Validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_WORD = "勞工"  # treatment keyword
TEXT_COL = "關係文書"
TERM_COL = "屆期"
SEED = 42

LABELS = {
    "婦女": "Women",
    "教師": "Teachers",
    "國軍": "Military personnel",
    "原住民": "Indigenous peoples",
    "勞工": "Labor",
}
LABEL = LABELS.get(TARGET_WORD, TARGET_WORD)

np.random.seed(SEED)
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


# ---------------- Helpers ----------------
def term_key(x):
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else 999


def load_hiddens(uids):
    tensors = []
    for uid in tqdm(uids, desc="Loading embeddings"):
        path = os.path.join(HIDDEN_DIR, f"{HIDDEN_PREFIX}{uid}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Hidden state not found: {path}")
        tensors.append(torch.load(path, map_location="cpu", weights_only=True).float())
    out = torch.stack(tensors, dim=0)
    if out.ndim == 3:  # [N, 1, D] -> [N, D]
        out = out.squeeze(1)
    if out.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {out.shape}")
    return out.numpy()


# ---------------- Load data ----------------
df = pd.read_csv(CSV_FILE, encoding="utf-8-sig", low_memory=False)

missing = [c for c in ["Global_UID", TERM_COL, TEXT_COL] if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

df["T"] = df[TEXT_COL].fillna("").astype(str).str.contains(TARGET_WORD, regex=False).astype(int)
df["Global_UID"] = pd.to_numeric(df["Global_UID"], errors="coerce")
df = df[df["Global_UID"].notna()].copy()
df["Global_UID"] = df["Global_UID"].astype(int)

print(f"N = {len(df):,} | T=1: {(df['T'] == 1).sum():,} | "
      f"T=0: {(df['T'] == 0).sum():,} | T rate: {df['T'].mean():.4f}")

R = load_hiddens(df["Global_UID"].tolist())

valid = np.isfinite(R).all(axis=1)
if not valid.all():
    print(f"Removed {(~valid).sum()} entries with NaN/Inf.")
    R, df = R[valid], df[valid].copy()

print(f"R shape = {R.shape}")


# ---------------- PCA ----------------
pca = PCA(n_components=2, random_state=SEED)
R_pca = pca.fit_transform(R)
pc1, pc2 = pca.explained_variance_ratio_
print(f"PC1 = {pc1:.4%} | PC2 = {pc2:.4%} | Total = {pc1 + pc2:.4%}")

pca_df = pd.DataFrame({
    "Global_UID": df["Global_UID"].values,
    "Term": df[TERM_COL].values,
    "T": df["T"].values,
    "PC1": R_pca[:, 0],
    "PC2": R_pca[:, 1],
})

pca_df.to_csv(os.path.join(OUTPUT_DIR, f"PCA_coordinates_{TARGET_WORD}.csv"),
              index=False, encoding="utf-8-sig")

pd.DataFrame({
    "Treatment": [TARGET_WORD],
    "N": [len(df)],
    "T1": [(df["T"] == 1).sum()],
    "T0": [(df["T"] == 0).sum()],
    "Embedding_Dimension": [R.shape[1]],
    "PC1_Explained_Variance": [pc1],
    "PC2_Explained_Variance": [pc2],
    "PC1_PC2_Total": [pc1 + pc2],
}).to_csv(os.path.join(OUTPUT_DIR, f"PCA_summary_{TARGET_WORD}.csv"),
          index=False, encoding="utf-8-sig")


# ---------------- Figure 1: by treatment ----------------
fig, ax = plt.subplots(figsize=(10, 8))
for t, s, a, lab in [(0, 10, 0.25, "T = 0"), (1, 22, 0.65, f"T = 1 ({LABEL})")]:
    m = pca_df["T"] == t
    ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"],
               s=s, alpha=a, label=lab, rasterized=True)
ax.set_xlabel(f"PC1 ({pc1:.2%} variance)")
ax.set_ylabel(f"PC2 ({pc2:.2%} variance)")
ax.set_title(f"PCA of Text Embeddings by Treatment Status\n{LABEL}")
ax.legend(frameon=True)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"Figure1_PCA_Treatment_{TARGET_WORD}.png"),
            bbox_inches="tight")
plt.show()
plt.close()


# ---------------- Figure 2: by legislative term ----------------
fig, ax = plt.subplots(figsize=(11, 9))
for term in sorted(pca_df["Term"].dropna().unique(), key=term_key):
    m = pca_df["Term"] == term
    ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"],
               s=10, alpha=0.35, label=str(term), rasterized=True)
ax.set_xlabel(f"PC1 ({pc1:.2%} variance)")
ax.set_ylabel(f"PC2 ({pc2:.2%} variance)")
ax.set_title(f"PCA of Text Embeddings by Legislative Term\n{LABEL}")
ax.legend(title="Term", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"Figure2_PCA_Term_{TARGET_WORD}.png"),
            bbox_inches="tight")
plt.show()
plt.close()


# ---------------- Term centroids (drift check) ----------------
centroids = (
    pca_df.groupby("Term")
    .agg(N=("Global_UID", "count"), T1=("T", "sum"),
         PC1_mean=("PC1", "mean"), PC2_mean=("PC2", "mean"))
    .reset_index()
)
centroids["T_rate"] = centroids["T1"] / centroids["N"]
centroids = centroids.sort_values("Term", key=lambda x: x.map(term_key))
centroids.to_csv(os.path.join(OUTPUT_DIR, f"PCA_term_centroids_{TARGET_WORD}.csv"),
                 index=False, encoding="utf-8-sig")

print(f"Done. Outputs saved to: {OUTPUT_DIR}")
