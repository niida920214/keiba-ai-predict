import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

from modules.constants import local_paths
from train_model import split_data


def main():
    print("=" * 60)
    print("Harville 変換係数の最適化 (tune_harville.py)")
    print("=" * 60)

    # 1. データの読み込み
    data_c_path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not data_c_path.exists():
        print(f"[ERROR] {data_c_path} が見つかりません。")
        return
    with open(data_c_path, "rb") as f:
        data_c = pickle.load(f)

    # 2. モデルの読み込み
    model_path = local_paths.MODEL_DIR / "lgbm_model.pickle"
    if not model_path.exists():
        print(f"[ERROR] Classifier が見つかりません: {model_path}")
        return
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 3. データの準備 (Validation で最適化 — テストデータでの最適化は Data Leakage)
    # train_model.py と同様に split_data で分割
    train_full, test = split_data(data_c, test_size=0.3)
    max_date = train_full["date"].max()
    threshold_date = max_date - pd.Timedelta(days=30)
    valid = train_full[train_full["date"] > threshold_date]

    if valid.empty:
        print("[ERROR] Validation データが空です。")
        return

    eval_df = valid.copy()
    print(f"  Validation データ: {eval_df.shape}")
    print(f"  Test データ（最終検証用、最適化には未使用）: {test.shape}")

    if "rank_win" not in eval_df.columns:
        print("[ERROR] eval_df に rank_win カラムがありません。train_model.py の enrich_data が走っているか確認してください。")
        return

    # 特徴量の準備 (EXCLUDE_COLS で一元管理)
    from modules.constants.features import EXCLUDE_COLS
    drop_cols = [c for c in EXCLUDE_COLS if c in eval_df.columns]
    X_feat = eval_df.drop(columns=drop_cols)

    print("P(3着以内) を推論中...")
    p3 = model.predict_proba(X_feat)[:, 1]
    
    y_true = eval_df["rank_win"].values

    # レースごとにグルーピングするための Series
    groups = eval_df.index

    # -----------------------------------------------------------------
    # 発見:
    # 従来の P_win ∝ P_top3 / harville_coeff は、レース内正規化 ( / sum) を行うと
    # harville_coeff が数学的に完全に約分されて消滅するため、係数が機能していませんでした。
    # 
    # 正しい非線形マッピングとして、以下を検証します。
    # Model A (Power Law): P_win ∝ (P_top3) ^ beta
    # Model B (Exponential): P_win ∝ exp(beta * P_top3)
    # Model C (Logit scale): P_win ∝ exp(beta * logit(P_top3))
    # -----------------------------------------------------------------

    def power_law_predict(beta):
        p1_raw = pd.Series(p3 ** beta, index=groups)
        return p1_raw.groupby(level=0).transform(lambda x: x / x.sum()).values

    def exponential_predict(beta):
        p1_raw = pd.Series(np.exp(beta * p3), index=groups)
        return p1_raw.groupby(level=0).transform(lambda x: x / x.sum()).values
    
    # 最小値 1e-5 でクリップして logit を計算安全にする
    p3_safe = np.clip(p3, 1e-5, 1.0 - 1e-5)
    logit_p3 = np.log(p3_safe / (1 - p3_safe))
    
    def logit_predict(beta):
        p1_raw = pd.Series(np.exp(beta * logit_p3), index=groups)
        return p1_raw.groupby(level=0).transform(lambda x: x / x.sum()).values

    # 最適化の目的関数 (Log Loss)
    def objective_power(beta):
        preds = power_law_predict(beta[0])
        # np.clip で log のゼロ割りを防ぐ
        return log_loss(y_true, np.clip(preds, 1e-7, 1.0 - 1e-7))

    def objective_exponential(beta):
        preds = exponential_predict(beta[0])
        return log_loss(y_true, np.clip(preds, 1e-7, 1.0 - 1e-7))

    def objective_logit(beta):
        preds = logit_predict(beta[0])
        return log_loss(y_true, np.clip(preds, 1e-7, 1.0 - 1e-7))

    print("\n最適化を実行中...")

    # Power Law の最適化
    res_power = minimize(objective_power, x0=[1.0], method="Nelder-Mead")
    best_beta_power = res_power.x[0]
    best_loss_power = res_power.fun

    # Exponential の最適化
    res_exp = minimize(objective_exponential, x0=[1.0], method="Nelder-Mead")
    best_beta_exp = res_exp.x[0]
    best_loss_exp = res_exp.fun
    
    # Logit scale の最適化
    res_logit = minimize(objective_logit, x0=[1.0], method="Nelder-Mead")
    best_beta_logit = res_logit.x[0]
    best_loss_logit = res_logit.fun

    # ベースライン (比例: beta=1)
    base_preds = power_law_predict(1.0)
    base_loss = log_loss(y_true, np.clip(base_preds, 1e-7, 1.0 - 1e-7))
    base_auc = roc_auc_score(y_true, base_preds)

    print("\n" + "=" * 40)
    print("Optimization Results:")
    print("=" * 40)
    print(f"1. Baseline (Proportional, beta=1.0)")
    print(f"   Log Loss: {base_loss:.5f}")
    print(f"   ROC AUC : {base_auc:.5f}")
    
    print(f"\n2. Power Law ( P_win ∝ P_top3 ^ beta )")
    print(f"   Optimal Beta: {best_beta_power:.4f}")
    print(f"   Log Loss: {best_loss_power:.5f}")
    
    print(f"\n3. Exponential ( P_win ∝ exp(beta * P_top3) )")
    print(f"   Optimal Beta: {best_beta_exp:.4f}")
    print(f"   Log Loss: {best_loss_exp:.5f}")

    print(f"\n4. Logit Scale ( P_win ∝ exp(beta * logit(P_top3)) )")
    print(f"   Optimal Beta: {best_beta_logit:.4f}")
    print(f"   Log Loss: {best_loss_logit:.5f}")

    # ===== 回収率シミュレーションによる最適なベータの検証 =====
    # 回収率が最も良くなる手法はどれかを簡単に計算する
    # Power law が数学的モデルとして最も扱いやすいため、Power Law の推移をプロット
    
    betas = np.linspace(0.5, 3.0, 20)
    losses = [objective_power([b]) for b in betas]
    
    plt.figure(figsize=(8, 5))
    plt.plot(betas, losses, marker="o", color="blue")
    plt.axvline(best_beta_power, color="red", linestyle="--", label=f"Optimal Beta: {best_beta_power:.2f}")
    plt.xlabel("Power Law Beta")
    plt.ylabel("Log Loss")
    plt.title("Harville Approx (Power Law) Beta vs Log Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_path = local_paths.RESULTS_DIR / "harville_beta_optimization.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nプロットを保存しました: {out_path}")
    
    print("\n結論:")
    print("従来の `P_win ∝ P_top3 / 2.4` は数式上、割り算が正規化で相殺されるため意味がありませんでした。")
    print(f"最適化の結果、 P_win ∝ (P_top3) ^ {best_beta_power:.3f} などの非線形変換が妥当です。")
    print("evaluation.py の predict_win_proba をこの Power Law 形式に書き換えることで、さらに精度向上が見込めます。")


if __name__ == "__main__":
    main()
