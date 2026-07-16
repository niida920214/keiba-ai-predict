"""
predict.py - 出馬表から予測を行うスクリプト (Benter式 2段階予測)
============================================
Stage 1: Fundamental Model (LGBMRanker) でオッズに依存しない能力推定
Stage 2: 市場確率 (リアルタイムオッズ) との統合
Stage 3: Stern-Gamma 補正による着順確率 (複勝率・馬券種別確率) の算出

出力:
    results/predictions/prediction_{race_id}.csv        … 各馬の予測一覧
    results/predictions/prediction_{race_id}_combos.csv … 馬連/ワイド/三連複の確率上位

Usage:
    python predict.py 202605010811 --date 2026/02/22
"""

import sys
import io

# Windows ターミナル (cp932) で日本語・Unicode 記号を正しく出力するため
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import pickle
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from modules.constants import local_paths
from position_probability import PositionProbabilityEstimator
from preprocessing import ShutubaTable, HorseResults, Peds
from scraper import HorseResults as ScraperHorseResults
from scraper import Peds as ScraperPeds
from tabulate import tabulate

PREDICTIONS_DIR = local_paths.RESULTS_DIR / "predictions"


# =====================================================================
# データ取得・前処理
# =====================================================================
def fetch_and_preprocess(race_id: str, date: str) -> ShutubaTable:
    """出馬表と依存データ（過去成績・血統）を取得し、特徴量まで加工する。"""
    print("Fetching Shutuba Table...")
    st = ShutubaTable.scrape([race_id], date)

    print("Preprocessing ShutubaTable and fetching dependent data...")
    st.preprocessing()

    horse_id_list = (
        st.data_p["horse_id"].tolist() if "horse_id" in st.data_p.columns else []
    )

    # 少数なのでレート制限なし
    hr_df = ScraperHorseResults.scrape(horse_id_list, skip_wait=True)
    hr = HorseResults(hr_df)

    peds_df = ScraperPeds.scrape(horse_id_list, skip_wait=True)
    p = Peds(peds_df)
    p.encode()

    st.merge_horse_results(hr)
    st.merge_peds(p.peds_e)

    # 学習フェーズで保存した LabelEncoder・カテゴリ一覧を引き継ぐ。
    # 軽量な predict_meta.pickle を優先し、なければ旧 results_obj.pickle にフォールバック。
    meta_path = local_paths.PROCESSED_DIR / "predict_meta.pickle"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        cats = meta["categories"]
        maxlen = max(len(v) for v in cats.values())
        # unique() で元のカテゴリ集合が復元できるよう、値をタイル状に敷き詰めた参照 DataFrame を作る
        ref_df = pd.DataFrame({
            col: np.resize(np.asarray(vals, dtype=object), maxlen)
            for col, vals in cats.items()
        })
        st.process_categorical(meta["le_horse"], meta["le_jockey"], ref_df)
    else:
        results_obj_path = local_paths.PROCESSED_DIR / "results_obj.pickle"
        with open(results_obj_path, "rb") as f:
            results_obj = pickle.load(f)
        st.process_categorical(
            results_obj.le_horse, results_obj.le_jockey, results_obj.data_pe
        )
    return st


# =====================================================================
# モデル読み込み・予測
# =====================================================================
def load_pickle_if_exists(path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def align_features(X: pd.DataFrame, feature_names) -> pd.DataFrame:
    """モデルの学習時特徴量に合わせて列を整列し、object 列を数値化する。"""
    if callable(feature_names):
        feature_names = feature_names()
    X_aligned = X.copy()
    for f in feature_names:
        if f not in X_aligned.columns:
            X_aligned[f] = np.nan
    X_aligned = X_aligned[list(feature_names)]
    for col in X_aligned.columns:
        if X_aligned[col].dtype == object:
            X_aligned[col] = pd.to_numeric(
                X_aligned[col], errors="coerce"
            ).astype(np.float64)
    return X_aligned


def unwrap_calibrated(model):
    """CalibratedClassifierCV の場合は内部の推定器を返す。"""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def predict_fundamental(X: pd.DataFrame, ranker_model, classifier) -> np.ndarray | None:
    """Stage 1: Fundamental Model による勝率推定（Ranker 優先）。"""
    if ranker_model is not None:
        X_ranker = align_features(X, ranker_model.feature_name_)
        rank_scores = ranker_model.predict(X_ranker)

        # Softmax で確率化
        e_x = np.exp(rank_scores - np.max(rank_scores))
        p_fundamental = e_x / e_x.sum()

        print(f"\n  === Benter Stage 1 (Fundamental: Ranker) ===")
        print(f"  Ranker score range: [{rank_scores.min():.3f}, {rank_scores.max():.3f}]")
        print(f"  P_Fund range: [{p_fundamental.min():.4f}, {p_fundamental.max():.4f}]")
        return p_fundamental

    if classifier is not None:
        base = unwrap_calibrated(classifier)
        X_cls = (
            align_features(X, base.feature_name_)
            if hasattr(base, "feature_name_")
            else X
        )
        y_pred = base.predict_proba(X_cls)[:, 1]
        p_fundamental = y_pred / y_pred.sum()
        print(f"\n  === Benter Stage 1 (Fundamental: Classifier) ===")
        print(f"  P_Fund range: [{p_fundamental.min():.4f}, {p_fundamental.max():.4f}]")
        return p_fundamental

    return None


def predict_top3(X: pd.DataFrame, classifier) -> np.ndarray | None:
    """分類器による P(3着以内)。キャリブレーション済みモデルを優先する。"""
    if classifier is None:
        return None
    base = unwrap_calibrated(classifier)
    if not hasattr(base, "feature_name_"):
        return None
    X_cls = align_features(X, base.feature_name_)
    try:
        return classifier.predict_proba(X_cls)[:, 1]
    except Exception:
        try:
            return base.predict_proba(X_cls)[:, 1]
        except Exception:
            return None


def fetch_market_odds(race_id: str) -> dict[int, float]:
    """リアルタイム単勝オッズを取得する（失敗時は空 dict）。"""
    try:
        from jra_playwright_async import fetch_win_odds_map
        odds_map = fetch_win_odds_map(race_id)
        if odds_map:
            print(f"\nOdds fetched ({len(odds_map)} horses) -> Benter Stage 2 integration")
        else:
            print("\nWARN: Could not fetch odds (using Fundamental only)")
        return odds_map or {}
    except Exception as e:
        print(f"\nWARN: Odds fetch error: {e} (using Fundamental only)")
        return {}


def compute_market_probs(umaban: np.ndarray, odds_map: dict[int, float]) -> np.ndarray | None:
    """Stage 2: オッズから市場確率（オーバーラウンド除去済み）を計算する。"""
    if not odds_map:
        return None
    inv = np.array([
        1.0 / odds_map[u] if odds_map.get(u, 0) and odds_map[u] > 0 else 0.0
        for u in umaban
    ])
    total = inv.sum()
    if total <= 0:
        return None
    print(f"\n  === Benter Stage 2 (Market) ===")
    print(f"  Overround: {total:.3f}")
    p_market = inv / total
    print(f"  P_market range: [{p_market.min():.4f}, {p_market.max():.4f}]")
    return p_market


def load_alpha() -> float:
    """tune_alpha.py で最適化済みの統合係数を読み込む。"""
    alpha = load_pickle_if_exists(local_paths.MODEL_DIR / "optimal_alpha.pickle")
    if alpha is not None:
        print(f"\n  Optimized Alpha loaded: {alpha:.3f}")
        return alpha
    print("\n  [INFO] optimal_alpha.pickle not found, using default Alpha=0.7")
    return 0.7


def integrate_benter(
    p_fundamental: np.ndarray | None, p_market: np.ndarray | None, alpha: float
) -> np.ndarray | None:
    """Benter (1994): P_final = alpha * P_fund + (1-alpha) * P_market"""
    if p_fundamental is not None and p_market is not None:
        p_final = alpha * p_fundamental + (1 - alpha) * p_market
        p_final = p_final / p_final.sum()
        print(f"\n  === Benter integrated (alpha={alpha}) ===")
        print(f"  P_final range: [{p_final.min():.4f}, {p_final.max():.4f}]")
        return p_final
    if p_fundamental is not None:
        print("\n  [NOTE] No odds -> Fundamental only")
        return p_fundamental
    if p_market is not None:
        print("\n  [NOTE] No Fundamental -> Market only")
        return p_market
    return None


# =====================================================================
# 出力生成
# =====================================================================
def get_horse_names(st: ShutubaTable, n: int) -> list[str]:
    """出馬表から馬名リストを取得する。"""
    for c in st.data.columns:
        if "馬名" in str(c):
            names = st.data[c].tolist()
            if len(names) >= n:
                return names[:n]
            return names + [f"Unknown-{i}" for i in range(len(names), n)]
    return [f"Horse {i}" for i in range(n)]


def get_umaban(st: ShutubaTable, n: int) -> np.ndarray:
    """前処理済みデータから馬番配列を取得する（欠損時は連番）。"""
    if "馬番" in st.data_p.columns:
        umaban = pd.to_numeric(st.data_p["馬番"], errors="coerce")
        if umaban.notna().all():
            return umaban.astype(int).to_numpy()[:n]
    return np.arange(1, n + 1)


def build_result_table(
    race_id: str,
    date: str,
    umaban: np.ndarray,
    horse_names: list[str],
    p_win_final: np.ndarray,
    p_place: np.ndarray,
    p_top3_model: np.ndarray | None,
    p_fundamental: np.ndarray | None,
    p_market: np.ndarray | None,
    odds_map: dict[int, float],
) -> pd.DataFrame:
    """各馬 1 行の予測結果テーブルを構築する。"""
    n = len(p_win_final)
    real_odds = np.array([odds_map.get(int(u), np.nan) for u in umaban], dtype=float)
    # 取得失敗・発売前などオッズ 0 以下は欠損として扱う
    real_odds = np.where(real_odds > 0, real_odds, np.nan)

    # 市場確率（オーバーラウンド除去済み）
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_odds = np.where(real_odds > 0, 1.0 / real_odds, np.nan)
    total_market = np.nansum(inv_odds)
    market_prob = inv_odds / total_market if total_market > 0 else np.full(n, np.nan)

    ev = p_win_final * real_odds
    edge = p_win_final - market_prob
    with np.errstate(divide="ignore", invalid="ignore"):
        kelly = np.where(real_odds > 1, edge / (real_odds - 1), np.nan)

    # 人気（オッズ昇順の順位）
    odds_series = pd.Series(real_odds)
    ninki = odds_series.rank(method="min").astype("Int64")

    df = pd.DataFrame({
        "race_id": race_id,
        "日付": date,
        "馬番": umaban.astype(int),
        "馬名": horse_names[:n],
        "P(Win)": np.round(p_win_final, 4),
        "P(Place)": np.round(p_place, 4),
        "単勝オッズ": real_odds,
        "人気": ninki,
        "市場確率": np.round(market_prob, 4),
        "EV": np.round(ev, 4),
        "Edge": np.round(edge, 4),
        "Kelly": np.round(kelly, 4),
    })
    if p_top3_model is not None:
        df["P_Top3_model"] = np.round(p_top3_model, 4)
    if p_fundamental is not None:
        df["P_Fund"] = np.round(p_fundamental, 4)
    if p_market is not None:
        df["P_Market"] = np.round(p_market, 4)

    df = df.sort_values("P(Win)", ascending=False).reset_index(drop=True)
    df.insert(2, "予測順位", np.arange(1, n + 1))
    return df


def build_combo_table(
    estimator: PositionProbabilityEstimator,
    race_id: str,
    date: str,
    top_k: int = 6,
    top_n: int = 10,
) -> pd.DataFrame:
    """馬連・ワイド・三連複の確率上位組み合わせを計算する。"""
    rows = []

    top_indices = np.argsort(-estimator.p_win)[:top_k]
    top_umaban = estimator.umaban[top_indices]

    # 馬連
    for u1, u2 in combinations(sorted(int(u) for u in top_umaban), 2):
        rows.append(("馬連", f"{u1}-{u2}", estimator.p_quinella(u1, u2)))
    # ワイド
    for u1, u2 in combinations(sorted(int(u) for u in top_umaban), 2):
        rows.append(("ワイド", f"{u1}-{u2}", estimator.p_wide(u1, u2)))
    # 三連複
    for combo in combinations(sorted(int(u) for u in top_umaban), 3):
        rows.append(("三連複", "-".join(map(str, combo)), estimator.p_trio(*combo)))

    df = pd.DataFrame(rows, columns=["券種", "買い目", "確率"])
    df["確率"] = df["確率"].round(4)
    df = (
        df.sort_values(["券種", "確率"], ascending=[True, False])
        .groupby("券種", sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df.insert(0, "race_id", race_id)
    df.insert(1, "日付", date)
    return df


def print_report(df_res: pd.DataFrame, df_combo: pd.DataFrame) -> None:
    """コンソールに予測結果・推奨買い目を表示する。"""
    display_cols = [
        "予測順位", "馬番", "馬名", "P(Win)", "P(Place)",
        "単勝オッズ", "人気", "EV", "Kelly",
    ]
    display_cols = [c for c in display_cols if c in df_res.columns]

    print("\nPrediction Results:")
    print(tabulate(
        df_res[display_cols], headers="keys", tablefmt="grid",
        floatfmt=".4f", showindex=False,
    ))

    # 単勝の推奨買い目 (EV > 1.0 かつ Kelly > 0)
    if not df_res["EV"].isna().all():
        bets = df_res[(df_res["EV"] > 1.0) & (df_res["Kelly"] > 0)].sort_values(
            "EV", ascending=False
        )
        if not bets.empty:
            print("\n** Bet Recommendations (単勝 EV > 1.0, Kelly > 0):")
            print(tabulate(
                bets[["馬番", "馬名", "P(Win)", "単勝オッズ", "EV", "Kelly"]],
                headers="keys", tablefmt="grid", floatfmt=".4f", showindex=False,
            ))
        else:
            print("\n  [INFO] No horses with EV > 1.0 found.")

    # 馬券種別の確率上位（各券種 上位5点のみ表示、全量は CSV 参照）
    print("\n** 着順確率モデルによる買い目候補 (各券種 上位5点):")
    top5 = df_combo.groupby("券種", sort=False).head(5)
    print(tabulate(
        top5[["券種", "買い目", "確率"]],
        headers="keys", tablefmt="grid", floatfmt=".4f", showindex=False,
    ))


# =====================================================================
# メイン
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Predict Race Outcome")
    parser.add_argument("race_id", type=str, help="Target Race ID (e.g. 202605010811)")
    parser.add_argument(
        "--date", type=str, default=datetime.now().strftime("%Y/%m/%d"),
        help="Race Date (YYYY/MM/DD)  ※省略時は本日",
    )
    args = parser.parse_args()

    race_id = args.race_id
    date = args.date
    print(f"Predicting Race ID: {race_id} (Date: {date})")

    # 1. 出馬表の取得と前処理
    try:
        st = fetch_and_preprocess(race_id, date)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during data fetch/preprocessing: {e}")
        return

    # 2. モデル読み込み
    classifier = load_pickle_if_exists(local_paths.MODEL_DIR / "lgbm_model.pickle")
    if classifier is not None:
        print("Classifier loaded (P(Top3) model)")
    ranker_model = load_pickle_if_exists(local_paths.MODEL_DIR / "lgbm_ranker.pickle")
    if ranker_model is not None:
        print("Ranker loaded (Benter Fundamental Model)")
    else:
        print("  [INFO] Ranker not found. Using Classifier only.")

    if classifier is None and ranker_model is None:
        print("Error: No model found. Run train_model.py first.")
        return

    # 3. リアルタイムオッズ
    odds_map = fetch_market_odds(race_id)

    # 4. 予測 (Benter 2-Stage)
    print("Predicting...")
    drop_cols = ["date", "target", "finishing_position", "time_sec"]
    X = st.data_c.drop([c for c in drop_cols if c in st.data_c.columns], axis=1)

    umaban = get_umaban(st, len(X))

    p_fundamental = predict_fundamental(X, ranker_model, classifier)
    p_market = compute_market_probs(umaban, odds_map)
    p_win_final = integrate_benter(p_fundamental, p_market, load_alpha())

    if p_win_final is None:
        print("Error: No model available for prediction and no market odds found.")
        return

    # 分類器による P(3着以内) — Harville 近似とは独立したモデル予測
    p_top3_model = predict_top3(X, classifier)

    # 5. 着順確率 (Stern-Gamma 補正) — 複勝率・馬券種別確率
    stern_r = PositionProbabilityEstimator.load_stern_r()
    estimator = PositionProbabilityEstimator(p_win_final, umaban=umaban, stern_r=stern_r)
    p_place = np.array([estimator.p_place(int(u)) for u in umaban])

    # ----- Diagnostics -----
    print("\n===== Diagnostics =====")
    print(f"  P(Win) sum: {p_win_final.sum():.4f}  (expected: 1.0)")
    print(f"  P(Win) range: [{p_win_final.min():.4f}, {p_win_final.max():.4f}]")
    model_type = (
        "Benter (Ranker + Market)" if ranker_model is not None else "Classifier"
    )
    print(f"  Model: {model_type}, Stern r: {stern_r:.3f}")
    print("=" * 40)

    # 6. 出力
    horse_names = get_horse_names(st, len(p_win_final))
    df_res = build_result_table(
        race_id, date, umaban, horse_names,
        p_win_final, p_place, p_top3_model, p_fundamental, p_market, odds_map,
    )
    df_combo = build_combo_table(estimator, race_id, date)

    print_report(df_res, df_combo)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    res_path = PREDICTIONS_DIR / f"prediction_{race_id}.csv"
    combo_path = PREDICTIONS_DIR / f"prediction_{race_id}_combos.csv"
    df_res.to_csv(res_path, index=False, encoding="utf-8-sig")
    df_combo.to_csv(combo_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {res_path}")
    print(f"Saved to {combo_path}")


if __name__ == "__main__":
    main()
