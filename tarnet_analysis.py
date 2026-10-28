import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Use a CJK-capable font so Chinese labels in plots do not render as empty boxes.
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Keep minus signs rendering correctly
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
TARGET_WORD = "人民"
RANDOM_SEED = 42
EPOCHS = 150
LEARNING_RATE = 2e-4
N_FOLDS_CONFIG = 5   # cross-fitting folds; every bill is used for training and gets an out-of-fold prediction

# Output paths
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
OUTPUT_ROOT = os.path.join(DESKTOP_PATH, "TarNet_Research_Results")
THEME_DIR = os.path.join(OUTPUT_ROOT, f"Topic_{TARGET_WORD}")

# Plots and reports are saved here
os.makedirs(THEME_DIR, exist_ok=True)

# Path to gpi_pack
LIB_PATH = r'C:\Users\gg\Desktop\gemini_cli\gpi_pack-main\src'
if LIB_PATH not in sys.path: sys.path.insert(0, LIB_PATH)
from gpi_pack.TarNet import TarNet, load_hiddens
from gpi_pack.TNutil import dml_score, SpectralNormClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def find_safe_batch_size(train_size: int, target: int, search_radius: int = 12) -> int:
    best_b, best_score = target, -1
    lo = max(8, target - search_radius)
    hi = min(train_size, target + search_radius)
    for b in range(lo, hi + 1):
        remainder = train_size % b
        score = b if remainder == 0 else remainder
        if score > best_score:
            best_score = score
            best_b = b
    return best_b


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data and build the covariate matrix C
    print(f"\n>>> [Step 1 啟動] 主題: {TARGET_WORD}")
    df = pd.read_csv('C:/Users/gg/Desktop/6th_to_10th.csv', encoding='utf-8-sig', low_memory=False)
    df['T'] = df['關係文書'].fillna('').apply(lambda x: 1 if TARGET_WORD in x else 0)
    df['Y'] = df['三讀'].fillna(0).astype(float)

    # C = term/session dummies, year, sponsor/cosponsor counts, caucus-proposal flag
    term_dummies = pd.get_dummies(df['屆期'], prefix='term').astype(float)
    sess_dummies = pd.get_dummies(df['會期'], prefix='sess').astype(float)
    df['year'] = pd.to_datetime(df['提案時間'], errors='coerce').dt.year.fillna(2010)
    prop_cols = [c for c in df.columns if '提案人' in c]
    sec_cols = [c for c in df.columns if '連署人' in c]
    df['n_prop'] = df[prop_cols].notna().sum(axis=1)
    df['n_sec'] = df[sec_cols].notna().sum(axis=1)

    c_extra_cols = ['year', 'n_prop', 'n_sec']
    if '黨團提案' in df.columns:
        df['is_caucus_proposal'] = df['黨團提案'].fillna(0).astype(float)
        c_extra_cols.append('is_caucus_proposal')
        print("✅ 已將「黨團提案」欄位加入 C。")
    else:
        print("⚠️ 找不到「黨團提案」欄位，這次沒有加入 C，請確認欄名是否正確。")

    C_df = pd.concat([term_dummies, sess_dummies, df[c_extra_cols]], axis=1)

    # Standardize C (year, counts and 0/1 dummies are on very different scales)
    scaler = StandardScaler()
    C_np = scaler.fit_transform(C_df)

    # 2. Load hidden states (R)
    uids = df['Global_UID'].astype(int).tolist()
    print("正在載入 Hidden States...")
    R_np = load_hiddens(directory='C:/Users/gg/Desktop/全文本/hidden_states_全文本_1', hidden_list=uids, prefix="hidden_last_", device="cpu")
    Y, T = df['Y'].values, df['T'].values

    n_a = int((T == 1).sum()); n_b = int((T == 0).sum())
    print(f"樣本數: {len(df)}（T=1「{TARGET_WORD}」: {n_a}, T=0 其他: {n_b}）")

    # ------------------------------------------------------------------
    # Overlap diagnostic on R (not C): R is what the model actually uses,
    # and it may encode T even when C cannot predict T.
    # ------------------------------------------------------------------
    print(">>> 正在檢查 R 能不能預測出 T（overlap 診斷）...")
    r_diag_train_idx, r_diag_test_idx = train_test_split(
        np.arange(len(T)), test_size=0.3, random_state=RANDOM_SEED, stratify=T
    )
    diag_model = SpectralNormClassifier(input_dim=R_np.shape[1], hidden_sizes=[64, 32], lr=1e-4, nepoch=60)
    diag_model.fit(R_np[r_diag_train_idx], T[r_diag_train_idx].astype(int))
    diag_proba = diag_model.predict_proba(R_np[r_diag_test_idx])[:, 1]
    diag_auc = roc_auc_score(T[r_diag_test_idx], diag_proba)
    diag_acc = accuracy_score(T[r_diag_test_idx], diag_proba.round())
    print(f"[R 預測 T 診斷] Accuracy: {diag_acc:.3f}, AUC: {diag_auc:.3f}")
    if diag_auc > 0.85:
        print("⚠️ 警告：AUC 過高，R 幾乎能完美反推 T，overlap 假設可能不成立，"
              "下面算出來的 ATT 請謹慎解讀，建議改用配對反事實方法交叉驗證。")
    elif diag_auc > 0.7:
        print("提醒：AUC 偏高，overlap 有中度風險，下面的結果請謹慎解讀。")
    else:
        print("Overlap 診斷通過（T=1/T=0 兩組傾向分數分布重疊，沒有被推向兩個極端）。")

    # ------------------------------------------------------------------
    # Separate from overlap: how many propensity scores fall outside the AIPW
    # trim range [0.05, 0.95]. A high share inflates the AIPW variance even
    # when overlap holds (typical for a rare T=1 group).
    # ------------------------------------------------------------------
    base_rate = T[r_diag_test_idx].mean()
    trim_lower, trim_upper = 0.05, 0.95
    pct_below_trim = (diag_proba < trim_lower).mean() * 100
    pct_above_trim = (diag_proba > trim_upper).mean() * 100
    print(f"\n[傾向分數位置 vs Trim 影響量化]")
    print(f"  T=1 在診斷樣本中的實際佔比: {base_rate:.3f}"
          f"（如果傾向分數集中的範圍跟這個數字接近，代表分布集中在低值"
          f"主要是反映 T=1 本身稀有，不是模型異常）")
    print(f"  原始傾向分數 < {trim_lower}（會被 trim 夾到下界）: {pct_below_trim:.1f}%")
    print(f"  原始傾向分數 > {trim_upper}（會被 trim 夾到上界）: {pct_above_trim:.1f}%")
    if pct_below_trim > 30 or pct_above_trim > 30:
        print(f"  ⚠️ 有超過 30% 的樣本會被 trim 夾到邊界，代表 AIPW 修正項有相當大一部分"
              f"樣本的傾向分數其實貼著邊界，估計出來的標準誤會因此變寬，"
              f"這是稀有處理組（T=1 佔比低）的固有代價，不是 overlap 被違反，"
              f"但解讀信賴區間時要意識到這一點。")
    else:
        print(f"  Trim 影響的樣本比例不算高，AIPW 修正項的變異數放大問題相對有限。")

    # Propensity score distribution estimated from R
    plt.figure(figsize=(10, 5))
    sns.kdeplot(diag_proba[T[r_diag_test_idx] == 1], label=f"T=1 ({TARGET_WORD})", color="red", fill=True)
    sns.kdeplot(diag_proba[T[r_diag_test_idx] == 0], label="T=0 (其他)", color="blue", fill=True)
    plt.title(f"Propensity Score Distribution — 用 R 估計（{TARGET_WORD}）")
    plt.legend()
    plt.savefig(os.path.join(THEME_DIR, "Propensity_Score_from_R.png"))
    plt.close()
    print(f"✅ 已儲存 Propensity_Score_from_R.png（用 R 估計，取代原本用 C 估計的版本）")

    # ------------------------------------------------------------------
    # K-fold cross-fitting: every bill is used for training in the other folds
    # and receives one out-of-fold prediction in its own fold. The fold models
    # are temporary and not saved; all results use out-of-fold predictions only.
    # ------------------------------------------------------------------
    N_FOLDS = N_FOLDS_CONFIG
    print(f">>> 使用 {N_FOLDS}-fold cross-fitting，全資料都會被訓練、也都會得到樣本外預測...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    n_total = len(T)
    y1_oof = np.full(n_total, np.nan)   # out-of-fold predictions
    y0_oof = np.full(n_total, np.nan)
    fr_oof = None  # out-of-fold shared representation, used for the AIPW propensity model

    for fold_i, (fold_train_idx, fold_test_idx) in enumerate(skf.split(np.arange(n_total), T)):
        min_class_train = min((T[fold_train_idx] == 1).sum(), (T[fold_train_idx] == 0).sum())
        minority_frac = min_class_train / len(fold_train_idx)
        approx_fit_train_size = int(len(fold_train_idx) * 0.9)  # TarNet.fit holds out valid_perc=0.1 internally
        target_batch = max(16, min(int(3 / max(minority_frac, 0.01)), 128))
        dynamic_batch_size = find_safe_batch_size(approx_fit_train_size, target_batch)

        print(f"[Fold {fold_i+1}/{N_FOLDS}] 訓練 {len(fold_train_idx)} 筆, "
              f"驗證 {len(fold_test_idx)} 筆, batch_size={dynamic_batch_size}")

        fold_model = TarNet(
            epochs=150,
            batch_size=dynamic_batch_size,
            architecture_z=[128, 64],
            architecture_y=[256, 128, 1],
            bn=False,
            verbose=False,
            model_dir=None,  # temporary model, not saved
        )
        fold_model.optim = torch.optim.Adam(fold_model.model.parameters(), lr=LEARNING_RATE)
        fold_model.fit(
            R_np[fold_train_idx], Y[fold_train_idx], T[fold_train_idx],
            C=C_np[fold_train_idx], valid_perc=0.1, plot_loss=False
        )

        y1_out, fr_test = fold_model.predict(
            R_np[fold_test_idx], np.ones(len(fold_test_idx)), c=C_np[fold_test_idx]
        )
        y0_out, _ = fold_model.predict(
            R_np[fold_test_idx], np.zeros(len(fold_test_idx)), c=C_np[fold_test_idx]
        )
        # No sigmoid: the built-in loss is MSE, so the output is already a probability estimate
        y1_oof[fold_test_idx] = np.clip(y1_out.detach().cpu().numpy().flatten(), 0.0, 1.0)
        y0_oof[fold_test_idx] = np.clip(y0_out.detach().cpu().numpy().flatten(), 0.0, 1.0)

        if fr_oof is None:
            fr_oof = np.zeros((n_total, fr_test.shape[1]))
        fr_oof[fold_test_idx] = fr_test.detach().cpu().numpy()

    assert not np.isnan(y1_oof).any(), "有資料沒被任何一折評估到，請檢查 StratifiedKFold 設定"
    print("✅ K-fold cross-fitting 完成，全部樣本都已取得樣本外預測。")

    # ------------------------------------------------------------------
    # Check that the model actually uses the keyword signal: if the out-of-fold
    # shared representation fr cannot predict T at all (AUC ~ 0.5), the y1/y0
    # gap is likely driven by C or noise rather than the keyword.
    # ------------------------------------------------------------------
    print("\n>>> 正在檢查 fr（模型內部表徵）能不能預測出 T ——驗證模型有沒有真的學到關鍵字效應...")
    fr_diag_train_idx, fr_diag_test_idx = train_test_split(
        np.arange(n_total), test_size=0.3, random_state=RANDOM_SEED, stratify=T
    )
    fr_diag_model = SpectralNormClassifier(input_dim=fr_oof.shape[1], hidden_sizes=[32, 16], lr=1e-4, nepoch=60)
    fr_diag_model.fit(fr_oof[fr_diag_train_idx], T[fr_diag_train_idx].astype(int))
    fr_diag_proba = fr_diag_model.predict_proba(fr_oof[fr_diag_test_idx])[:, 1]
    fr_diag_auc = roc_auc_score(T[fr_diag_test_idx], fr_diag_proba)
    fr_diag_acc = accuracy_score(T[fr_diag_test_idx], fr_diag_proba.round())
    print(f"[fr 預測 T 診斷] Accuracy: {fr_diag_acc:.3f}, AUC: {fr_diag_auc:.3f}")
    if fr_diag_auc < 0.55:
        print("⚠️ 警告：AUC 接近 0.5，fr 幾乎預測不出 T，代表模型很可能沒有真的學到"
              "「有沒有出現關鍵字」這件事帶來的影響，下面算出來的 y1/y0 差異"
              "（ATT）可信度存疑，很可能主要是 C 或雜訊在驅動，不是關鍵字效應。")
    elif fr_diag_auc < 0.65:
        print("提醒：AUC 偏低，模型對關鍵字資訊的利用程度有限，ATT 的效應量請保守解讀。")
    else:
        print("fr 能有效預測出 T，代表模型確實把關鍵字資訊編碼進內部表徵、並用它做判斷，"
              "下面的 ATT 相對可信是關鍵字帶來的效應，不是純粹在吃 C 或雜訊。")

    # ------------------------------------------------------------------
    # Naive ATT from out-of-fold predictions, covering all T=1 bills
    # ------------------------------------------------------------------
    print(">>> 執行亞群組反事實分析報告（cross-fitted 樣本外版本）...")

    idx1_all = np.where(T == 1)[0]
    y1_p = y1_oof[idx1_all]
    y0_p = y0_oof[idx1_all]
    att_p = y1_p - y0_p
    Y_t1 = Y[idx1_all]
    result_holdout = {
        "n": len(idx1_all),
        "actual_mean": Y_t1.mean(), "actual_se": pd.Series(Y_t1).sem(),
        "y1_mean": y1_p.mean(), "y1_se": pd.Series(y1_p).sem(),
        "y0_mean": y0_p.mean(), "y0_se": pd.Series(y0_p).sem(),
        "att_mean": att_p.mean(), "att_se": pd.Series(att_p).sem(),
        "idx1_local": idx1_all, "y1_p": y1_p, "y0_p": y0_p,
    }
    print(f"[Cross-fitted 樣本外] n={result_holdout['n']}, "
          f"實際通過率={result_holdout['actual_mean']:.4f}±{result_holdout['actual_se']:.4f}, "
          f"y1={result_holdout['y1_mean']:.4f}±{result_holdout['y1_se']:.4f}, "
          f"y0={result_holdout['y0_mean']:.4f}±{result_holdout['y0_se']:.4f}, "
          f"ATT(naive)={result_holdout['att_mean']:.4f}±{result_holdout['att_se']:.4f}")

    # ------------------------------------------------------------------
    # AIPW-corrected ATE: out-of-fold outcome predictions plus a 2-fold
    # cross-fitted propensity model on fr_oof
    # ------------------------------------------------------------------
    aipw_att = None
    aipw_se = None
    if result_holdout["n"] >= 20:
        print(">>> 正在計算 AIPW 校正後的 ATE（跟 naive 版本對照）...")

        idx_a, idx_b = train_test_split(
            np.arange(n_total), test_size=0.5, random_state=RANDOM_SEED, stratify=T
        )
        ps_model_a = SpectralNormClassifier(input_dim=fr_oof.shape[1], hidden_sizes=[32, 16], lr=1e-4, nepoch=60)
        ps_model_b = SpectralNormClassifier(input_dim=fr_oof.shape[1], hidden_sizes=[32, 16], lr=1e-4, nepoch=60)
        ps_model_a.fit(fr_oof[idx_a], T[idx_a].astype(int))
        tpred_b = ps_model_a.predict_proba(fr_oof[idx_b])[:, 1]
        ps_model_b.fit(fr_oof[idx_b], T[idx_b].astype(int))
        tpred_a = ps_model_b.predict_proba(fr_oof[idx_a])[:, 1]

        tpreds = np.zeros(n_total)
        tpreds[idx_a] = tpred_a; tpreds[idx_b] = tpred_b
        tpreds = np.clip(tpreds, 0.05, 0.95)

        psi = dml_score(T, Y, tpreds, y1_oof, y0_oof)
        aipw_att = np.mean(psi)
        aipw_se = np.std(psi) / np.sqrt(len(psi))
        print(f"[AIPW 校正後 ATE（全體樣本，cross-fitted，注意是 ATE 不是 ATT）] {aipw_att:.4f} ± {aipw_se:.4f}")
        print(f"[對照] naive plug-in ATT（僅 T=1 子群）= {result_holdout['att_mean']:.4f} ± {result_holdout['att_se']:.4f}")
        print("兩者概念上不完全一樣（一個是 ATE、一個是 ATT），數字接近才代表校正沒有帶來太大差異。")

    # 6. Final report and per-term analysis (out-of-fold, all T=1 bills)
    if result_holdout is not None:
        df_t1 = df.iloc[result_holdout["idx1_local"]].copy()
        df_t1['y1_p'] = result_holdout["y1_p"]
        df_t1['y0_p'] = result_holdout["y0_p"]
        df_t1['att_p'] = result_holdout["y1_p"] - result_holdout["y0_p"]

        term_summary = df_t1.groupby('屆期').agg({
            'Y': ['mean', 'sem', 'count'],
            'y1_p': ['mean', 'sem'],
            'y0_p': ['mean', 'sem'],
            'att_p': ['mean', 'sem']
        }).reset_index()
        term_summary.columns = ['term', 'Y_mean', 'Y_se', 'n_evaluated', 'y1_mean', 'y1_se', 'y0_mean', 'y0_se', 'att_mean', 'att_se']

        # n_evaluated should equal the number of T=1 bills in that term; terms with very few
        # bills (e.g. < 5) give unstable estimates simply because of sample size.
        print(f"\n提醒：改成 cross-fitting 之後，n_evaluated 應該等於該屆全部 T=1 筆數"
              f"（不再有之前 holdout 版本『某屆剛好只抽到 1、2 筆』的問題）。"
              f"但如果某個屆別本身筆數就很少，那一屆的估計值統計上仍然不穩定，"
              f"這是資料量本身的限制，不是方法論的問題。")

        report = (
            f"【Step 1 最終報告: {TARGET_WORD}】（{N_FOLDS}-fold cross-fitting 樣本外評估版本）\n"
            f"--------------------------------------------------\n"
            f"0. Overlap 診斷（用 R，訓練資料乾不乾淨）：Accuracy={diag_acc:.3f}, AUC={diag_auc:.3f}\n"
            f"0b. fr 診斷（模型有沒有真的學到關鍵字效應）：Accuracy={fr_diag_acc:.3f}, AUC={fr_diag_auc:.3f}\n"
            f"1. 總體亞群組校準（Cross-fitted 樣本外, 均值 ± 標準誤，n={result_holdout['n']}，"
            f"涵蓋全部 T=1 樣本，不是切一小塊 holdout）:\n"
            f"   - 實際通過率: {result_holdout['actual_mean']:.4f} ± {result_holdout['actual_se']:.4f}\n"
            f"   - 模型預測 y1: {result_holdout['y1_mean']:.4f} ± {result_holdout['y1_se']:.4f}\n"
            f"   - 假設無關鍵字 y0: {result_holdout['y0_mean']:.4f} ± {result_holdout['y0_se']:.4f}\n"
            f"   - naive plug-in ATT: {result_holdout['att_mean']:.4f} ± {result_holdout['att_se']:.4f}\n"
        )
        if aipw_att is not None:
            report += f"   - AIPW 校正後 ATE（全體，cross-fitted）: {aipw_att:.4f} ± {aipw_se:.4f}\n"
        report += f"\n2. 分屆詳細數據（全部 T=1 樣本，皆為 cross-fitted 樣本外預測）：\n"

        term_table_str = term_summary.to_string(index=False, formatters={
            'Y_mean': '{:.4f}'.format, 'Y_se': '±{:.4f}'.format,
            'y1_mean': '{:.4f}'.format, 'y1_se': '±{:.4f}'.format,
            'y0_mean': '{:.4f}'.format, 'y0_se': '±{:.4f}'.format,
            'att_mean': '{:.4f}'.format, 'att_se': '±{:.4f}'.format
        })
        report += term_table_str + "\n"
        report += "--------------------------------------------------\n"

        print("\n" + report)

        # Append to the global results log
        LOG_FILE = os.path.join(DESKTOP_PATH, "Centralized_Causal_Results.csv")
        log_rows = []
        for _, row in term_summary.iterrows():
            log_rows.append({
                'Topic': TARGET_WORD, 'Term': row['term'],
                'Actual_Pass_Rate': f"{row['Y_mean']:.4f}", 'Actual_SE': f"{row['Y_se']:.4f}",
                'Predicted_y1': f"{row['y1_mean']:.4f}", 'y1_SE': f"{row['y1_se']:.4f}",
                'Counterfactual_y0': f"{row['y0_mean']:.4f}", 'y0_SE': f"{row['y0_se']:.4f}",
                'ATT_naive': f"{row['att_mean']:.4f}", 'ATT_SE': f"{row['att_se']:.4f}",
                'Overlap_AUC_from_R': f"{diag_auc:.3f}",
                'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        log_rows.append({
            'Topic': TARGET_WORD, 'Term': 'Global(Holdout)',
            'Actual_Pass_Rate': f"{result_holdout['actual_mean']:.4f}", 'Actual_SE': f"{result_holdout['actual_se']:.4f}",
            'Predicted_y1': f"{result_holdout['y1_mean']:.4f}", 'y1_SE': f"{result_holdout['y1_se']:.4f}",
            'Counterfactual_y0': f"{result_holdout['y0_mean']:.4f}", 'y0_SE': f"{result_holdout['y0_se']:.4f}",
            'ATT_naive': f"{result_holdout['att_mean']:.4f}", 'ATT_SE': f"{result_holdout['att_se']:.4f}",
            'ATE_AIPW': f"{aipw_att:.4f}" if aipw_att is not None else "",
            'Overlap_AUC_from_R': f"{diag_auc:.3f}",
            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        log_df = pd.DataFrame(log_rows)
        if not os.path.exists(LOG_FILE):
            log_df.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
        else:
            log_df.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"✅ 分屆數據已累計至全域紀錄檔: {LOG_FILE}")

        with open(os.path.join(THEME_DIR, "Step1_Final_Report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        pd.DataFrame({
            'UID': np.array(uids)[result_holdout["idx1_local"]],
            'y1_factual': result_holdout["y1_p"],
            'y0_counter': result_holdout["y0_p"],
        }).to_csv(os.path.join(THEME_DIR, "Step1_Causal_Data_Holdout.csv"), index=False)

    print(f"所有結果已儲存至: {THEME_DIR}")

if __name__ == "__main__":
    main()
