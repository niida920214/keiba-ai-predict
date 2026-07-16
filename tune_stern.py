"""
tune_stern.py – Stern-Gamma パラメータ r の最適化
==================================================
Ranker モデルの Softmax 出力を用いて、Stern-Gamma パラメータ r を
テストデータの着順結果に対する対数尤度を最大化するように最適化する。

学術的根拠:
    Stern (1990): ガンマ分布モデルにおける形状パラメータ r は
    Harville (r=1) と Henery (r→∞) を連続的に補間する。
    r < 1 は人気馬の過大評価バイアスを緩和する効果がある。

使い方:
    python tune_stern.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from modules.constants import local_paths
from modules.constants.features import EXCLUDE_COLS
from train_model import split_data

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_log_likelihood(
    p_win_races: list[np.ndarray],
    finish_pos_races: list[np.ndarray],
    stern_r: float,
) -> float:
    """Stern-Gamma モデルの対数尤度を計算する。

    各レースの着順結果について、Stern-Gamma 補正付き Harville 確率の
    対数尤度を計算する。

    Parameters
    ----------
    p_win_races : list[np.ndarray]
        各レースの P(Win) 配列のリスト
    finish_pos_races : list[np.ndarray]
        各レースの着順配列のリスト（1-indexed）
    stern_r : float
        Stern-Gamma パラメータ

    Returns
    -------
    float
        全レースの平均対数尤度
    """
    total_ll = 0.0
    n_races = 0

    for p_win, finish_pos in zip(p_win_races, finish_pos_races):
        n = len(p_win)
        if n < 3:
            continue

        # 着順でソート（1着→2着→3着の順）
        order = np.argsort(finish_pos)
        sorted_p = p_win[order]

        # 上位3着の Harville-Stern 対数尤度
        ll = 0.0
        remaining_power = np.sum(sorted_p ** stern_r)

        for pos in range(min(3, n)):  # 1着〜3着
            p_r = sorted_p[pos] ** stern_r
            if remaining_power <= 0 or p_r <= 0:
                ll += -50.0  # ペナルティ
                break
            ll += np.log(p_r / remaining_power)
            remaining_power -= p_r

        total_ll += ll
        n_races += 1

    return total_ll / n_races if n_races > 0 else -np.inf


def main():
    print("=" * 60)
    print("Stern-Gamma パラメータ r の最適化")
    print("=" * 60)

    # ------------------------------------------------------------------
    # データ・モデルの読み込み
    # ------------------------------------------------------------------
    data_c_path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not data_c_path.exists():
        print(f"[ERROR] {data_c_path} が見つかりません。")
        return

    print("[1] データを読み込み中...")
    with open(data_c_path, "rb") as f:
        data_c = pickle.load(f)

    ranker_path = local_paths.MODEL_DIR / "lgbm_ranker.pickle"
    if not ranker_path.exists():
        print("[ERROR] Ranker モデルが見つかりません。train_model.py を実行してください。")
        return

    with open(ranker_path, "rb") as f:
        ranker_model = pickle.load(f)
    print("  Ranker loaded.")

    # ------------------------------------------------------------------
    # テストデータの準備
    # ------------------------------------------------------------------
    print("\n[2] テストデータを準備中...")
    _, test = split_data(data_c, test_size=0.3)

    if "finishing_position" not in test.columns:
        print("[ERROR] finishing_position カラムが存在しません。")
        return

    drop_cols = [c for c in EXCLUDE_COLS if c in test.columns]
    X_test = test.drop(columns=drop_cols)

    # Ranker 特徴量
    rfeat = ranker_model.feature_name_
    if callable(rfeat):
        rfeat = rfeat()

    X_r = X_test.copy()
    for f in rfeat:
        if f not in X_r.columns:
            X_r[f] = np.nan
    X_r = X_r[rfeat]
    for col in X_r.columns:
        if X_r[col].dtype == object:
            X_r[col] = pd.to_numeric(X_r[col], errors="coerce")

    # レースごとの P(Win) と着順を準備
    scores = ranker_model.predict(X_r)

    p_win_races = []
    finish_pos_races = []
    unique_races = X_test.index.unique()

    for race_id in unique_races:
        mask = X_test.index == race_id
        race_scores = scores[mask]
        race_finish = test.loc[mask, "finishing_position"].values

        # Softmax
        e_x = np.exp(race_scores - np.max(race_scores))
        p_win = e_x / e_x.sum()

        p_win_races.append(p_win)
        finish_pos_races.append(race_finish)

    print(f"  レース数: {len(p_win_races)}")

    # ------------------------------------------------------------------
    # Optuna 最適化
    # ------------------------------------------------------------------
    N_TRIALS = 200

    def objective(trial: optuna.Trial) -> float:
        stern_r = trial.suggest_float("stern_r", 0.4, 1.5)
        ll = compute_log_likelihood(p_win_races, finish_pos_races, stern_r)
        return ll  # 最大化

    print(f"\n[3] Optuna 最適化 ({N_TRIALS} trials) ...")
    study = optuna.create_study(
        study_name="stern_r_study",
        storage="sqlite:///optuna_study.db",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_r = study.best_params["stern_r"]
    best_ll = study.best_value

    print(f"\n  Best stern_r: {best_r:.4f}")
    print(f"  Best log-likelihood: {best_ll:.6f}")

    # Harville (r=1.0) との比較
    harville_ll = compute_log_likelihood(p_win_races, finish_pos_races, 1.0)
    print(f"  Harville (r=1.0) log-likelihood: {harville_ll:.6f}")
    print(f"  改善率: {((best_ll - harville_ll) / abs(harville_ll) * 100):.2f}%")

    # 保存
    save_path = local_paths.MODEL_DIR / "optimal_stern_r.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(best_r, f)
    print(f"\n  保存完了: {save_path}")

    # ------------------------------------------------------------------
    # r の感度分析
    # ------------------------------------------------------------------
    print("\n[4] Stern r の感度分析:")
    print(f"  {'r':>6s}  {'Log-Likelihood':>15s}")
    print("  " + "-" * 25)
    for r_test in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        ll = compute_log_likelihood(p_win_races, finish_pos_races, r_test)
        marker = " <-- best" if abs(r_test - best_r) < 0.05 else ""
        print(f"  {r_test:>6.2f}  {ll:>15.6f}{marker}")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
