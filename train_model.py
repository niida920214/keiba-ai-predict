"""
train_model.py – LightGBM モデル学習 & Optuna ハイパーパラメータチューニング
============================================================================
前処理済みデータ (data_c) を読み込み、時系列分割で train/valid/test に分け、
Optuna でハイパーパラメータを探索した後、最終モデルを学習・保存する。

Phase 3: Benter式オッズ分離 + ランキング学習 (LGBMRanker) を追加。
         log_odds/odds_rank は訓練から排除し、純粋な能力推定モデルを構築する。
"""

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.metrics import log_loss, roc_auc_score, ndcg_score

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
from modules.constants import local_paths
from modules.constants.features import EXCLUDE_COLS

# ---------------------------------------------------------------------------
# パス定数 (local_pathsを使用)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. 時系列データ分割
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    date 列に基づいて時系列で train / test に分割する。
    """
    sorted_df = df.sort_values("date")
    unique_indices = sorted_df.index.unique()

    split_point = int(len(unique_indices) * (1 - test_size))
    train_indices = unique_indices[:split_point]
    test_indices = unique_indices[split_point:]

    train_df = sorted_df.loc[sorted_df.index.isin(train_indices)]
    test_df = sorted_df.loc[sorted_df.index.isin(test_indices)]

    return train_df, test_df


# ---------------------------------------------------------------------------
# 2. データ拡充 (Phase 2/3)
# ---------------------------------------------------------------------------
def enrich_data(data_c: pd.DataFrame) -> pd.DataFrame:
    """
    data_c に不足するカラムを追加する（再スクレイピング不要）。

    追加カラム:
        - rank_win    : 1着=1, それ以外=0 (return_tables から取得)
    
    ※ Benter式オッズ分離: log_odds/odds_rank は訓練特徴量から排除。
      予測時にリアルタイムオッズを市場確率として統合する (predict.py).
    """
    # --- rank_win の追加 ---
    if "rank_win" not in data_c.columns:
        print("  [ENRICH] rank_win を return_tables から生成中...")
        return_path = local_paths.RAW_RETURN_TABLES_DIR / "return_tables.pickle"
        if return_path.exists():
            return_tables = pd.read_pickle(return_path)
            tansho = return_tables[return_tables[0] == "単勝"][[1]]
            tansho.columns = ["win_umaban"]
            tansho["win_umaban"] = pd.to_numeric(tansho["win_umaban"], errors="coerce")
            # 重複を除去（レースIDごとに1つ）
            win_umaban = tansho.groupby(level=0)["win_umaban"].first()
            # data_c にマッピング
            data_c["_win_umaban"] = data_c.index.map(win_umaban)
            data_c["rank_win"] = (data_c["馬番"] == data_c["_win_umaban"]).astype(int)
            data_c.drop("_win_umaban", axis=1, inplace=True)
            print(f"    rank_win 生成完了: 1着数={data_c['rank_win'].sum()}, "
                  f"全体={len(data_c)}, 比率={data_c['rank_win'].mean():.4f}")
        else:
            print("  [WARN] return_tables が見つかりません。rank_win をスキップします。")

    # ※ log_odds / odds_rank は Benter式オッズ分離により訓練から排除。
    # 予測時に predict.py でリアルタイムオッズから生成する。
    # 既存の log_odds/odds_rank カラムがあれば削除
    for col in ["log_odds", "odds_rank"]:
        if col in data_c.columns:
            data_c.drop(col, axis=1, inplace=True)
            print(f"  [BENTER] {col} を訓練特徴量から除外")

    return data_c


# ★ Benter式オッズ分離: オッズ関連特徴量を訓練から排除
# EXCLUDE_COLS は modules.constants.features から import 済み


def prepare_xy(df: pd.DataFrame, target_col: str = "rank"):
    """特徴量 X と目的変数 y を分離する。"""
    cols_to_drop = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    return X, y


def make_rank_label(finishing_pos: pd.Series) -> pd.Series:
    """着順を LGBMRanker 用の関連度スコアに変換する。
    
    NDCG 最適化では高いスコアが「良い」と判定される。
    LightGBM lambdarank は整数ラベル必須。
    1着=5, 2着=3, 3着=2, 4着以降=0
    
    改善: 旧スコア {4:1, 5:1} は3着と4-5着の区別が不十分で、
    三連系馬券に必要な「3着以内 vs 4着以下」の識別力が弱かった。
    学術的根拠: Burges et al. (2010) のランキング学習では、
    関連度スコアの差が大きいペアほど強く学習される。
    """
    mapping = {1: 5, 2: 3, 3: 2}
    return finishing_pos.map(lambda x: mapping.get(int(x), 0) if pd.notnull(x) else 0).astype(int)


def make_group_sizes(df: pd.DataFrame) -> np.ndarray:
    """レースID (インデックス) ごとのグループサイズを返す。
    
    LGBMRanker の group パラメータに使用する。
    データが race_id 順にソートされている必要がある。
    """
    return df.groupby(level=0).size().values


def compute_ndcg_per_race(y_true, y_pred, group_sizes, k=3) -> float:
    """レース単位の平均 NDCG@k を計算する。
    
    Parameters
    ----------
    y_true : array-like
        正解ラベル（関連度スコア）
    y_pred : array-like
        予測スコア
    group_sizes : array-like
        各レースの出走頭数
    k : int
        NDCG の評価位置
    
    Returns
    -------
    float
        全レースの平均 NDCG@k
    """
    ndcg_scores = []
    start_idx = 0
    for g_size in group_sizes:
        end_idx = start_idx + g_size
        if hasattr(y_true, 'iloc'):
            y_true_g = y_true.iloc[start_idx:end_idx].values.reshape(1, -1)
        else:
            y_true_g = np.array(y_true[start_idx:end_idx]).reshape(1, -1)
        y_pred_g = np.array(y_pred[start_idx:end_idx]).reshape(1, -1)
        if y_true_g.max() > 0:
            ndcg_k = min(k, g_size)
            ndcg_scores.append(ndcg_score(y_true_g, y_pred_g, k=ndcg_k))
        start_idx = end_idx
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# 4. Optuna objective
# ---------------------------------------------------------------------------


def create_objective(X_train, y_train, X_valid, y_valid, extra_params=None):
    """
    Optuna 用の objective 関数を返す（クロージャ）。
    extra_params があればモデルに追加する。
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,
            # --- 探索パラメータ ---
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": 1000,
        }
        if extra_params:
            params.update(extra_params)

        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        loss = log_loss(y_valid, y_pred_proba)
        trial.set_user_attr("best_iteration", model.best_iteration_)

        return loss

    return objective


def create_ranker_objective(X_train, y_train, group_train,
                             X_valid, y_valid, group_valid):
    """
    LGBMRanker 用の Optuna objective 関数を返す（NDCG@3 を最大化）。
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5, 10],
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,
            # --- 探索パラメータ ---
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": 1000,
        }

        ranker = LGBMRanker(**params)
        ranker.fit(
            X_train,
            y_train,
            group=group_train,
            eval_set=[(X_valid, y_valid)],
            eval_group=[group_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred = ranker.predict(X_valid)
        avg_ndcg = compute_ndcg_per_race(y_valid, y_pred, group_valid, k=3)
        trial.set_user_attr("best_iteration", ranker.best_iteration_)

        return avg_ndcg

    return objective


# ---------------------------------------------------------------------------
# 5. モデル学習+キャリブレーション の共通関数
# ---------------------------------------------------------------------------
def train_and_calibrate(
    X_train, y_train, X_valid, y_valid, X_test, y_test,
    best_params, best_iteration, model_name="model",
    calibration_method="isotonic",
):
    """
    ベストパラメータでモデルを学習 → キャリブレーション。

    Parameters
    ----------
    calibration_method : str
        'isotonic' (Isotonic Regression) or 'sigmoid' (Platt Scaling)
    """
    print(f"\n  [{model_name}] train で学習中...")
    final_params = best_params.copy()
    final_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_estimators": best_iteration,
    })

    final_model = LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)

    # 確率補正
    method = "sigmoid" if "sigmoid" in model_name.lower() else calibration_method
    print(f"  [{model_name}] キャリブレーション中 (method={method})...")
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(final_model), method=method
    )
    calibrated_model.fit(X_valid, y_valid)

    # テスト評価
    y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    test_logloss = log_loss(y_test, y_test_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"  [{model_name}] Test Log Loss: {test_logloss:.6f}")
    print(f"  [{model_name}] Test ROC AUC:  {test_auc:.6f}")

    return calibrated_model, final_params


# ---------------------------------------------------------------------------
# 6. メイン処理
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LightGBM モデル学習 & Optuna チューニング")
    parser.add_argument(
        "--trials", type=int, default=200,
        help="Optuna 試行回数の基準値（デフォルト 200。Win/Ranker モデルは比例配分）",
    )
    args = parser.parse_args()

    local_paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # データ読み込み & 拡充
    # ------------------------------------------------------------------
    data_c_path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not data_c_path.exists():
        print(f"[ERROR] {data_c_path} が見つかりません。")
        return

    print("[1] データを読み込み中...")
    with open(data_c_path, "rb") as f:
        data_c = pickle.load(f)
    print(f"  data_c shape: {data_c.shape}")

    # Phase 2/3: データ拡充（Benter式: log_odds/odds_rank を除外）
    print("\n[1.5] データ拡充中...")
    data_c = enrich_data(data_c)
    print(f"  拡充後 data_c shape: {data_c.shape}")

    # ------------------------------------------------------------------
    # 時系列分割: train_full / test
    # ------------------------------------------------------------------
    print("\n[2] 時系列分割 (train_full / test) ...")
    train_full, test = split_data(data_c, test_size=0.3)
    print(f"  train_full: {train_full.shape}")
    print(f"  test:       {test.shape}")

    # ------------------------------------------------------------------
    # 時系列分割: train / valid (1ヶ月)
    # ------------------------------------------------------------------
    print("\n[3] 時系列分割 (train / valid) ...")
    max_date = train_full["date"].max()
    threshold_date = max_date - pd.Timedelta(days=30)

    train = train_full[train_full["date"] <= threshold_date]
    valid = train_full[train_full["date"] > threshold_date]

    print(f"  train: {train.shape}")
    print(f"  valid: {valid.shape}")

    # ==================================================================
    # Model 1: 3着以内予測（既存互換）
    # ==================================================================
    print("\n" + "=" * 60)
    print("Model 1: 3着以内予測 (rank = top3)")
    print("=" * 60)

    X_train, y_train = prepare_xy(train, target_col="rank")
    X_valid, y_valid = prepare_xy(valid, target_col="rank")
    X_test, y_test = prepare_xy(test, target_col="rank")

    print(f"\n[4] 特徴量: {X_train.shape[1]} 個")
    print(f"  X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")

    # --- Optuna ハイパーパラメータチューニング ---
    N_TRIALS = args.trials
    print(f"\n[5] Optuna チューニング開始 ({N_TRIALS} trials) ...")

    study = optuna.create_study(
        study_name="top3_study",
        storage="sqlite:///optuna_study.db",
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(
        create_objective(X_train, y_train, X_valid, y_valid),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best log_loss: {study.best_value:.6f}")
    best_iteration = study.best_trial.user_attrs.get("best_iteration", 500)
    print(f"  Best iteration: {best_iteration}")

    # 学習 + キャリブレーション
    calibrated_top3, best_params_top3 = train_and_calibrate(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        study.best_params, best_iteration, model_name="Top3"
    )

    # 保存
    model_path = local_paths.MODEL_DIR / "lgbm_model.pickle"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_top3, f)
    print(f"\n  Top3 モデル保存: {model_path}")

    params_path = local_paths.MODEL_DIR / "best_params.pickle"
    with open(params_path, "wb") as f:
        pickle.dump(best_params_top3, f)

    # 特徴量重要度 (top 20)
    print(f"\n  [Top3 特徴量重要度 Top 20]")
    importances_top3 = np.mean([
        clf.estimator.feature_importances_ for clf in calibrated_top3.calibrated_classifiers_
    ], axis=0)
    importance = pd.Series(
        importances_top3, index=X_train.columns
    ).sort_values(ascending=False)
    for i, (feat, imp) in enumerate(importance.head(20).items()):
        print(f"    {i+1:>2}. {feat}: {imp}")

    # ==================================================================
    # Model 2: 1着予測（単勝回収率最適化用）
    # ==================================================================
    print("\n" + "=" * 60)
    print("Model 2: 1着予測 (rank_win = 1着のみ)")
    print("=" * 60)

    if "rank_win" not in data_c.columns:
        print("  [SKIP] rank_win がデータに存在しません。")
    else:
        _, y_train_win = prepare_xy(train, target_col="rank_win")
        _, y_valid_win = prepare_xy(valid, target_col="rank_win")
        _, y_test_win = prepare_xy(test, target_col="rank_win")

        # クラス不均衡の計算
        n_neg = (y_train_win == 0).sum()
        n_pos = (y_train_win == 1).sum()
        spw = n_neg / n_pos
        print(f"\n  不均衡比率: {n_neg}:{n_pos} (scale_pos_weight={spw:.2f})")

        # Top3モデルのパラメータを流用 + scale_pos_weight
        win_extra_params = {"scale_pos_weight": spw}

        N_TRIALS_WIN = max(1, args.trials // 2)
        print(f"\n[6] Win Model Optuna ({N_TRIALS_WIN} trials) ...")

        study_win = optuna.create_study(
            study_name="win_study",
            storage="sqlite:///optuna_study.db",
            direction="minimize",
            load_if_exists=True
        )
        study_win.optimize(
            create_objective(X_train, y_train_win, X_valid, y_valid_win,
                             extra_params=win_extra_params),
            n_trials=N_TRIALS_WIN,
            show_progress_bar=True,
        )

        print(f"\n  Best trial: #{study_win.best_trial.number}")
        print(f"  Best log_loss: {study_win.best_value:.6f}")
        best_iteration_win = study_win.best_trial.user_attrs.get("best_iteration", 500)

        win_params = study_win.best_params.copy()
        win_params["scale_pos_weight"] = spw

        # --- キャリブレーション方式比較 (isotonic vs sigmoid) ---
        # クラス不均衡が大きい場合、sigmoid (Platt Scaling) が安定する場合がある
        calibrated_win_iso, best_params_win = train_and_calibrate(
            X_train, y_train_win, X_valid, y_valid_win, X_test, y_test_win,
            win_params, best_iteration_win, model_name="Win(isotonic)"
        )
        calibrated_win_sig, _ = train_and_calibrate(
            X_train, y_train_win, X_valid, y_valid_win, X_test, y_test_win,
            win_params, best_iteration_win, model_name="Win(sigmoid)",
        )

        # テストデータで比較
        y_iso = calibrated_win_iso.predict_proba(X_test)[:, 1]
        y_sig = calibrated_win_sig.predict_proba(X_test)[:, 1]
        ll_iso = log_loss(y_test_win, y_iso)
        ll_sig = log_loss(y_test_win, y_sig)
        print(f"\n  キャリブレーション比較:")
        print(f"    Isotonic: {ll_iso:.6f}")
        print(f"    Sigmoid:  {ll_sig:.6f}")

        if ll_sig < ll_iso:
            calibrated_win = calibrated_win_sig
            print(f"    → Sigmoid を採用")
        else:
            calibrated_win = calibrated_win_iso
            print(f"    → Isotonic を採用")

        win_model_path = local_paths.MODEL_DIR / "lgbm_model_win.pickle"
        with open(win_model_path, "wb") as f:
            pickle.dump(calibrated_win, f)
        print(f"\n  Win モデル保存: {win_model_path}")

        win_params_path = local_paths.MODEL_DIR / "best_params_win.pickle"
        with open(win_params_path, "wb") as f:
            pickle.dump(best_params_win, f)

        # 特徴量重要度 (top 20)
        print(f"\n  [Win 特徴量重要度 Top 20]")
        importances_win = np.mean([
            clf.estimator.feature_importances_ for clf in calibrated_win.calibrated_classifiers_
        ], axis=0)
        importance_win = pd.Series(
            importances_win, index=X_train.columns
        ).sort_values(ascending=False)
        for i, (feat, imp) in enumerate(importance_win.head(20).items()):
            print(f"    {i+1:>2}. {feat}: {imp}")

    # ==================================================================
    # Model 3: LGBMRanker (Fundamental Model — Benter式 Stage 1)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Model 3: LGBMRanker (ランキング学習 — Benter式 Fundamental Model)")
    print("=" * 60)

    if "finishing_position" not in data_c.columns:
        print("  [SKIP] finishing_position がデータに存在しません。")
        print("         main.py で data_c を再生成してください。")
    else:
        # ランキング学習用のラベルを生成
        y_rank_train = make_rank_label(train["finishing_position"])
        y_rank_valid = make_rank_label(valid["finishing_position"])
        y_rank_test = make_rank_label(test["finishing_position"])

        # グループサイズ（各レースの出走頭数）
        # データが race_id (インデックス) でグルーピングされていることを確認
        group_train = make_group_sizes(train)
        group_valid = make_group_sizes(valid)
        group_test = make_group_sizes(test)

        print(f"\n  ランキングラベル分布:")
        print(f"    train: スコア平均={y_rank_train.mean():.3f}, グループ数={len(group_train)}")
        print(f"    valid: スコア平均={y_rank_valid.mean():.3f}, グループ数={len(group_valid)}")
        print(f"    test:  スコア平均={y_rank_test.mean():.3f}, グループ数={len(group_test)}")

        # --- Optuna チューニング (Ranker) ---
        N_TRIALS_RANK = max(1, args.trials * 3 // 4)
        print(f"\n[7] Ranker Optuna ({N_TRIALS_RANK} trials) ...")

        study_rank = optuna.create_study(
            study_name="ranker_study",
            storage="sqlite:///optuna_study.db",
            direction="maximize",  # NDCG は最大化
            load_if_exists=True
        )
        study_rank.optimize(
            create_ranker_objective(
                X_train, y_rank_train, group_train,
                X_valid, y_rank_valid, group_valid
            ),
            n_trials=N_TRIALS_RANK,
            show_progress_bar=True,
        )

        print(f"\n  Best trial: #{study_rank.best_trial.number}")
        print(f"  Best NDCG@3: {study_rank.best_value:.6f}")
        best_iteration_rank = study_rank.best_trial.user_attrs.get("best_iteration", 500)

        # 最終 Ranker モデルの学習
        print(f"\n  [Ranker] 最終モデルを学習中...")
        final_rank_params = study_rank.best_params.copy()
        final_rank_params.update({
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5, 10],
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_estimators": best_iteration_rank,
        })

        final_ranker = LGBMRanker(**final_rank_params)
        final_ranker.fit(
            X_train, y_rank_train, group=group_train,
            eval_set=[(X_valid, y_rank_valid)],
            eval_group=[group_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Test NDCG@3
        y_rank_test_pred = final_ranker.predict(X_test)
        avg_ndcg = compute_ndcg_per_race(y_rank_test, y_rank_test_pred, group_test, k=3)
        print(f"  [Ranker] Test NDCG@3: {avg_ndcg:.6f}")

        # 保存
        ranker_path = local_paths.MODEL_DIR / "lgbm_ranker.pickle"
        with open(ranker_path, "wb") as f:
            pickle.dump(final_ranker, f)
        print(f"\n  Ranker モデル保存: {ranker_path}")

        ranker_params_path = local_paths.MODEL_DIR / "best_params_ranker.pickle"
        with open(ranker_params_path, "wb") as f:
            pickle.dump(final_rank_params, f)

        # 特徴量重要度 (top 20)
        print(f"\n  [Ranker 特徴量重要度 Top 20]")
        importance_rank = pd.Series(
            final_ranker.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        for i, (feat, imp) in enumerate(importance_rank.head(20).items()):
            print(f"    {i+1:>2}. {feat}: {imp}")

    # ------------------------------------------------------------------
    # シミュレーション用データ保存（三連系対応で拡充）
    # ------------------------------------------------------------------
    test_tansho = test["単勝"].copy() if "単勝" in test.columns else None
    test_date = test["date"].copy() if "date" in test.columns else None
    test_umaban = test["馬番"].copy() if "馬番" in test.columns else None
    test_finish = test["finishing_position"].copy() if "finishing_position" in test.columns else None
    sim_data = {
        "test_tansho": test_tansho,
        "test_date": test_date,
        "test_umaban": test_umaban,
        "test_finishing_position": test_finish,
        "y_test": y_test,
    }
    sim_path = local_paths.MODEL_DIR / "simulation_data.pickle"
    with open(sim_path, "wb") as f:
        pickle.dump(sim_data, f)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("学習パイプライン完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
