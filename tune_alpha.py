"""
tune_alpha.py – Benter Alpha (Fundamental/Market ブレンド係数) の最適化
======================================================================
Validation データ上で Alpha を 0.0〜1.0 の範囲でグリッドサーチし、
Log Loss と回収率の両指標で最適値を決定する。

学術的根拠:
    Benter (1994) "Computer Based Horse Race Handicapping and Wagering Systems"
    では、Fundamental Model と Market Probability の線形結合係数を
    データから推定することが推奨されている。

Usage:
    python tune_alpha.py
"""

import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from modules.constants import local_paths
from modules.constants.features import EXCLUDE_COLS
from train_model import split_data

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = "MS Gothic"


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """温度付き softmax 変換。"""
    scaled = (x - np.max(x)) / temperature
    e_x = np.exp(scaled)
    return e_x / e_x.sum()


def compute_p_fundamental_per_race(ranker_model, X: pd.DataFrame) -> np.ndarray:
    """
    Ranker モデルからレース単位で softmax 正規化した P(Win) を計算する。
    """
    rfeat = ranker_model.feature_name_
    if callable(rfeat):
        rfeat = rfeat()

    X_r = X.copy()
    for f in rfeat:
        if f not in X_r.columns:
            X_r[f] = np.nan
    X_r = X_r[rfeat]
    for col in X_r.columns:
        if X_r[col].dtype == object:
            X_r[col] = pd.to_numeric(X_r[col], errors="coerce")

    scores = ranker_model.predict(X_r)

    # レース単位で softmax
    p_fund = np.zeros_like(scores, dtype=float)
    for race_id in X.index.unique():
        mask = X.index == race_id
        p_fund[mask] = softmax(scores[mask])

    return p_fund


def compute_p_market(odds: pd.Series) -> np.ndarray:
    """
    オッズから市場確率を計算する（オーバーラウンド除去済み）。

    Parameters
    ----------
    odds : pd.Series
        各馬の単勝オッズ（index = race_id）
    """
    p_raw = 1.0 / odds.replace(0, np.nan)
    # レース単位でオーバーラウンド除去
    p_market = p_raw.groupby(level=0).transform(lambda x: x / x.sum())
    return p_market.fillna(0).values


def main():
    print("=" * 60)
    print("Benter Alpha の最適化 (tune_alpha.py)")
    print("=" * 60)

    # 1. データ読み込み
    data_c_path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not data_c_path.exists():
        print(f"[ERROR] {data_c_path} が見つかりません。")
        return
    with open(data_c_path, "rb") as f:
        data_c = pickle.load(f)
    print(f"  data_c shape: {data_c.shape}")

    # 2. Ranker モデル読み込み
    ranker_path = local_paths.MODEL_DIR / "lgbm_ranker.pickle"
    if not ranker_path.exists():
        print("[ERROR] Ranker モデルが見つかりません。train_model.py を先に実行してください。")
        return
    with open(ranker_path, "rb") as f:
        ranker_model = pickle.load(f)
    print(f"  Ranker loaded: {ranker_path}")

    # 3. データ分割（train / valid / test）
    train_full, test = split_data(data_c, test_size=0.3)
    max_date = train_full["date"].max()
    threshold_date = max_date - pd.Timedelta(days=30)
    valid = train_full[train_full["date"] > threshold_date]

    if valid.empty:
        print("[ERROR] Validation データが空です。")
        return

    print(f"  Valid: {valid.shape}, Test: {test.shape}")

    # 4. rank_win の確認
    if "rank_win" not in valid.columns:
        # return_tables から生成
        from train_model import enrich_data
        valid = enrich_data(valid.copy())

    if "rank_win" not in valid.columns:
        print("[ERROR] rank_win がありません。")
        return

    # 5. 特徴量の準備
    drop_cols = [c for c in EXCLUDE_COLS if c in valid.columns]
    X_valid = valid.drop(columns=drop_cols)
    y_true = valid["rank_win"].values

    # 6. P_fundamental の計算
    print("\nP_fundamental を計算中...")
    p_fund = compute_p_fundamental_per_race(ranker_model, X_valid)
    print(f"  P_fund range: [{p_fund.min():.4f}, {p_fund.max():.4f}]")

    # 7. P_market の計算（単勝オッズから）
    if "単勝" not in valid.columns:
        print("[ERROR] 単勝カラムがありません。")
        return

    print("P_market を計算中...")
    p_market = compute_p_market(valid["単勝"])
    print(f"  P_market range: [{p_market.min():.4f}, {p_market.max():.4f}]")

    # 8. Alpha グリッドサーチ
    alpha_values = np.linspace(0.0, 1.0, 21)
    results = []

    print(f"\nAlpha グリッドサーチ ({len(alpha_values)} steps)...")
    for alpha in alpha_values:
        p_final = alpha * p_fund + (1 - alpha) * p_market

        # レース単位で再正規化
        p_final_series = pd.Series(p_final, index=valid.index)
        p_final_norm = p_final_series.groupby(level=0).transform(lambda x: x / x.sum())
        p_final_clipped = np.clip(p_final_norm.values, 1e-7, 1.0 - 1e-7)

        # Log Loss
        ll = log_loss(y_true, p_final_clipped)

        # 回収率（EV > 1.0 の馬に均等賭け）
        ev = p_final_norm.values * valid["単勝"].values
        bet_mask = ev >= 1.0
        if bet_mask.sum() > 0:
            # 単勝的中判定
            n_bets = bet_mask.sum()
            returns = []
            bet_indices = np.where(bet_mask)[0]
            for idx in bet_indices:
                if y_true[idx] == 1:
                    returns.append(float(valid["単勝"].iloc[idx]))
                else:
                    returns.append(0.0)
            roi = sum(returns) / n_bets
        else:
            n_bets = 0
            roi = 0.0

        results.append({
            "alpha": alpha,
            "log_loss": ll,
            "roi": roi,
            "n_bets": n_bets,
        })
        print(f"  α={alpha:.2f}: LogLoss={ll:.5f}, ROI={roi:.4f}, n_bets={n_bets}")

    results_df = pd.DataFrame(results)

    # 9. 最適 Alpha の決定
    # Log Loss が最小の Alpha
    best_ll_row = results_df.loc[results_df["log_loss"].idxmin()]
    # ROI が最大の Alpha（n_bets >= 50 のフィルタ付き）
    practical = results_df[results_df["n_bets"] >= 50]
    if not practical.empty:
        best_roi_row = practical.loc[practical["roi"].idxmax()]
    else:
        best_roi_row = results_df.loc[results_df["roi"].idxmax()]

    print("\n" + "=" * 50)
    print("最適化結果:")
    print("=" * 50)
    print(f"  Log Loss 最小: α={best_ll_row['alpha']:.2f} (LL={best_ll_row['log_loss']:.5f})")
    print(f"  ROI 最大:      α={best_roi_row['alpha']:.2f} (ROI={best_roi_row['roi']:.4f}, n_bets={best_roi_row['n_bets']:.0f})")

    # 実用的な Alpha: ROI 基準（実際の利益に直結）
    optimal_alpha = float(best_roi_row["alpha"])
    print(f"\n  ★ 採用する Alpha: {optimal_alpha:.2f}")

    # 10. 保存
    alpha_path = local_paths.MODEL_DIR / "optimal_alpha.pickle"
    with open(alpha_path, "wb") as f:
        pickle.dump(optimal_alpha, f)
    print(f"  保存完了: {alpha_path}")

    # 11. プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Log Loss
    ax1.plot(results_df["alpha"], results_df["log_loss"], "b-o", markersize=4)
    ax1.axvline(best_ll_row["alpha"], color="red", linestyle="--",
                label=f"Best α={best_ll_row['alpha']:.2f}")
    ax1.set_xlabel("Alpha (Fundamental weight)", fontsize=12)
    ax1.set_ylabel("Log Loss", fontsize=12)
    ax1.set_title("Log Loss vs Alpha", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ROI
    ax2.plot(results_df["alpha"], results_df["roi"], "g-o", markersize=4)
    ax2.axvline(optimal_alpha, color="red", linestyle="--",
                label=f"Best α={optimal_alpha:.2f}")
    ax2.axhline(1.0, color="orange", linestyle="--", alpha=0.5, label="Break-even")
    ax2.set_xlabel("Alpha (Fundamental weight)", fontsize=12)
    ax2.set_ylabel("ROI", fontsize=12)
    ax2.set_title("ROI vs Alpha (EV≥1.0 均等賭け)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = local_paths.RESULTS_DIR / "alpha_optimization.png"
    local_paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  プロット保存: {plot_path}")

    print("\n" + "=" * 60)
    print("Alpha 最適化完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
