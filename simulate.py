"""
simulate.py -- 回収率シミュレーション実行スクリプト (Phase 1-4 統合版)
====================================================================
学習済みモデルとテストデータを読み込み、複数の戦略 (z-score / EV-Harville / EV-WinModel / Kelly / Benter(Ranker)) を比較する。
"""

import pickle
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import ModelEvaluator, gain, plot
from modules.constants import local_paths
from modules.constants.features import EXCLUDE_COLS
from train_model import split_data

# Windows は MS Gothic、Linux (GitHub Actions等) は Noto Sans CJK にフォールバック
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "MS Gothic", "Noto Sans CJK JP", "IPAGothic", "DejaVu Sans",
]


# ---------------------------------------------------------------------------
# EVベース戦略用の gain 関数
# ---------------------------------------------------------------------------
def gain_ev(return_func, X, n_samples=50, ev_range=None):
    """min_ev を走査して回収率を計算する。"""
    if ev_range is None:
        ev_range = [0.8, 2.0]

    from tqdm import tqdm

    gain_dict = {}
    for i in tqdm(range(n_samples), desc="gain_ev"):
        min_ev = ev_range[0] + (ev_range[1] - ev_range[0]) * i / n_samples
        n_bets, return_rate, n_hits, std = return_func(X, min_ev=min_ev)
        if n_bets > 2:
            gain_dict[min_ev] = {
                "return_rate": return_rate,
                "n_hits": n_hits,
                "std": std,
                "n_bets": n_bets,
            }
    return pd.DataFrame(gain_dict).T


def checkpoint_upload(*paths) -> None:
    """成果物を保存した直後にクラウドへ逐次アップロードする。

    環境変数 CHECKPOINT_UPLOAD=1 のとき有効（GitHub Actions用）。
    タイムアウトで打ち切られても、完成済みの成果物は失われない。
    """
    import os
    if os.environ.get("CHECKPOINT_UPLOAD") != "1":
        return
    try:
        import cloud_storage
        files = {f"results/{Path(p).name}": Path(p) for p in paths if Path(p).exists()}
        if files:
            cloud_storage.upload_files(files, log=lambda _: None)
            print(f"  [CKPT] {', '.join(Path(p).name for p in paths)} をクラウドへ保存")
    except Exception as e:
        print(f"  [CKPT] アップロード失敗（処理は継続）: {e}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="回収率シミュレーション")
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="各戦略の閾値スキャンの段階数（デフォルト50。実行時間はほぼ比例）",
    )
    args = parser.parse_args()

    local_paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. データ・モデルの読み込み
    # ==================================================================
    print("[1] Loading data and models...")

    model_path = local_paths.MODEL_DIR / "lgbm_model.pickle"
    model = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"  Classifier loaded: {model_path}")
    else:
        print("  [WARN] Classifier model not found.")

    # Win model (Phase 2)
    win_model_path = local_paths.MODEL_DIR / "lgbm_model_win.pickle"
    win_model = None
    if win_model_path.exists():
        with open(win_model_path, "rb") as f:
            win_model = pickle.load(f)
        print(f"  Win Model loaded: {win_model_path}")
    else:
        print("  [INFO] Win Model not found -> Harville only")

    # Ranker model (Phase 3: Benter Fundamental)
    ranker_path = local_paths.MODEL_DIR / "lgbm_ranker.pickle"
    ranker_model = None
    if ranker_path.exists():
        with open(ranker_path, "rb") as f:
            ranker_model = pickle.load(f)
        print(f"  Ranker loaded: {ranker_path}")
    else:
        print("  [INFO] Ranker model not found -> Benter strategy skipped")

    data_c_path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not data_c_path.exists():
        print(f"[ERROR] {data_c_path} not found.")
        return
    with open(data_c_path, "rb") as f:
        data_c = pickle.load(f)

    return_tables_path = local_paths.RAW_RETURN_TABLES_DIR / "return_tables.pickle"
    if not return_tables_path.exists():
        print(f"[ERROR] {return_tables_path} not found.")
        return

    print(f"  data_c shape: {data_c.shape}")

    if model is None and ranker_model is None:
        print("[ERROR] No model found. Run train_model.py first.")
        return

    # ==================================================================
    # 2. Prepare test data
    # ==================================================================
    print("\n[2] Preparing test data...")
    _, test = split_data(data_c, test_size=0.3)

    # EXCLUDE_COLS から「単勝」「馬番」はシミュレーション（bet判定・EV計算）で必要なので残す
    sim_keep = {"単勝", "馬番"}
    drop_cols = [c for c in EXCLUDE_COLS if c in test.columns and c not in sim_keep]
    X_test = test.drop(columns=drop_cols)
    print(f"  X_test shape: {X_test.shape}")

    # ==================================================================
    # 3. ModelEvaluator
    # ==================================================================
    print("\n[3] Building ModelEvaluator...")
    if model is not None:
        me = ModelEvaluator(model, [str(return_tables_path)], win_model=win_model)
        print("  ModelEvaluator built")
    else:
        me = None
        print("  [INFO] No classifier -> skipping z-score/Harville/WinModel strategies")

    # ==================================================================
    # 4. シミュレーション実行
    N_SAMPLES = args.n_samples
    EV_RANGE = [0.8, 2.0]
    T_RANGE = [0.5, 3.5]
    MAX_ODDS_LIST = [10, 20, 30, 50]

    # --- 4-1. z-score (requires classifier) ---
    g_tansho = pd.DataFrame()
    if me is not None:
        print(f"\n[4-1] z-score strategy (n_samples={N_SAMPLES}) ...")
        g_tansho = gain(me.tansho_return, X_test, n_samples=N_SAMPLES, t_range=T_RANGE)

    # --- 4-2. EV-Harville (requires classifier) ---
    ev_harville_results = {}
    if me is not None:
        for max_odds in MAX_ODDS_LIST:
            print(f"\n[4-2] EV-Harville (odds<={max_odds}) ...")
            func = partial(me.tansho_return_ev, max_odds=max_odds)
            g = gain_ev(func, X_test, n_samples=N_SAMPLES, ev_range=EV_RANGE)
            ev_harville_results[max_odds] = g
            if not g.empty:
                print(f"  Max: {g['return_rate'].max():.4f} @ {g['return_rate'].idxmax():.3f}")

    # --- 4-3. EV-WinModel (requires Win model + classifier) ---
    ev_win_results = {}
    if win_model is not None and me is not None:
        for max_odds in [20, 30]:
            print(f"\n[4-3] EV-WinModel (odds<={max_odds}) ...")
            # _build_ev_table に use_harville=False を渡すため、
            # tansho_return_ev_win を定義
            def win_return_func(X, min_ev=1.0, _max_odds=max_odds):
                ev_table = me._build_ev_table(X, max_odds=_max_odds, use_harville=False)
                bet_table = ev_table[ev_table["ev"] >= min_ev]
                n_bets = len(bet_table)
                if n_bets == 0:
                    return 0, 0.0, 0, 0.0
                return_list = []
                for race_id, preds in bet_table.groupby(level=0):
                    return_list.append(
                        np.sum([me.bet(race_id, "tansho", umaban, 1)
                                for umaban in preds["馬番"]])
                    )
                if not return_list:
                    return 0, 0.0, 0, 0.0
                std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
                n_hits = np.sum([x > 0 for x in return_list])
                return_rate = np.sum(return_list) / n_bets
                return n_bets, return_rate, n_hits, std

            g = gain_ev(win_return_func, X_test, n_samples=N_SAMPLES, ev_range=EV_RANGE)
            ev_win_results[max_odds] = g
            if not g.empty:
                print(f"  Max: {g['return_rate'].max():.4f} @ {g['return_rate'].idxmax():.3f}")

    # --- 4-4. Kelly (WinModel + classifier) ---
    g_kelly_win = pd.DataFrame()
    if win_model is not None and me is not None:
        print(f"\n[4-4] Kelly-WinModel (odds<=20) ...")

        def kelly_win_func(X, min_ev=1.0, kelly_fraction=0.25):
            ev_table = me._build_ev_table(X, max_odds=20, use_harville=False)
            bet_table = ev_table[ev_table["ev"] >= min_ev].copy()
            n_bets = len(bet_table)
            if n_bets == 0:
                return 0, 0.0, 0, 0.0
            bet_table["kelly_f"] = (
                (bet_table["p_win"] * bet_table["単勝"] - 1)
                / (bet_table["単勝"] - 1)
            ).clip(lower=0) * kelly_fraction
            bet_table = bet_table[bet_table["kelly_f"] > 0]
            n_bets = len(bet_table)
            if n_bets == 0:
                return 0, 0.0, 0, 0.0
            total_bet = bet_table["kelly_f"].sum()
            return_list = []
            for race_id, preds in bet_table.groupby(level=0):
                return_list.append(
                    np.sum(preds.apply(
                        lambda x: me.bet(race_id, "tansho", x["馬番"], x["kelly_f"]),
                        axis=1,
                    ))
                )
            if not return_list:
                return 0, 0.0, 0, 0.0
            std = np.std(return_list) * np.sqrt(len(return_list)) / total_bet
            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / total_bet
            return n_bets, return_rate, n_hits, std

        g_kelly_win = gain_ev(kelly_win_func, X_test, n_samples=N_SAMPLES, ev_range=EV_RANGE)
        if not g_kelly_win.empty:
            print(f"  Max: {g_kelly_win['return_rate'].max():.4f}")

    # --- 4-5. Benter (Ranker) Strategy ---
    ev_benter_results = {}
    if ranker_model is not None and me is not None:
        for max_odds in [20, 30]:
            print(f"\n[4-5] Benter-Ranker (odds<={max_odds}) ...")

            def benter_return_func(X, min_ev=1.0, _max_odds=max_odds):
                """Ranker softmax -> EV -> bet."""
                # Get ranker features
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
                        X_r[col] = pd.to_numeric(X_r[col], errors='coerce')

                scores = ranker_model.predict(X_r)

                # Per-race softmax + EV computation
                unique_races = X.index.unique()
                bet_data = []
                for race_id in unique_races:
                    mask = X.index == race_id
                    race_scores = scores[mask]
                    e_x = np.exp(race_scores - np.max(race_scores))
                    p_win = e_x / e_x.sum()

                    race_X = X[mask].copy()
                    race_X["p_win"] = p_win
                    if "馬番" in race_X.columns:
                        race_X["umaban"] = race_X["馬番"]
                    else:
                        race_X["umaban"] = range(1, len(race_X) + 1)
                    if "単勝" in X.columns:
                        race_X["odds"] = X.loc[mask, "単勝"].values
                    else:
                        continue

                    race_X = race_X[race_X["odds"] <= _max_odds]
                    race_X = race_X[race_X["odds"] > 0]
                    race_X["ev"] = race_X["p_win"] * race_X["odds"]
                    race_X = race_X[race_X["ev"] >= min_ev]
                    if len(race_X) > 0:
                        for _, row in race_X.iterrows():
                            bet_data.append({
                                "race_id": race_id,
                                "umaban": int(row["umaban"]),
                                "ev": row["ev"],
                            })

                n_bets = len(bet_data)
                if n_bets == 0:
                    return 0, 0.0, 0, 0.0

                return_list = []
                for bd in bet_data:
                    ret = me.bet(bd["race_id"], "tansho", bd["umaban"], 1)
                    return_list.append(ret)
                if not return_list:
                    return 0, 0.0, 0, 0.0
                std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
                n_hits = np.sum([x > 0 for x in return_list])
                return_rate = np.sum(return_list) / n_bets
                return n_bets, return_rate, n_hits, std

            g = gain_ev(benter_return_func, X_test, n_samples=N_SAMPLES, ev_range=EV_RANGE)
            ev_benter_results[max_odds] = g
            if not g.empty:
                print(f"  Max: {g['return_rate'].max():.4f} @ {g['return_rate'].idxmax():.3f}")

    # --- 4-6〜4-9. 三連複/三連単/馬単/ワイド (Stern-Gamma 補正付き) ---
    # 高速版: レース×組み合わせのEVを1回だけ計算し、min_ev のスイープは
    # 軽量な集計で済ませる（従来は gain_ev が n_samples 回すべて再計算しており、
    # 三連単がタイムアウトする主因だった）。数式・分岐ロジックは元のまま。
    g_trio = pd.DataFrame()
    if me is not None:
        print(f"\n[4-6] 三連複 EV (Stern-Gamma, top_k=6) ...")
        try:
            rows = me.trio_ev_rows(X_test, top_k=6, ranker_model=ranker_model)
            g_trio = me.sweep_ev_rows(rows, n_samples=N_SAMPLES, ev_range=EV_RANGE)
            if not g_trio.empty:
                print(f"  Max: {g_trio['return_rate'].max():.4f} @ {g_trio['return_rate'].idxmax():.3f}")
        except Exception as e:
            print(f"  [ERROR] 三連複: {e}")

    g_trifecta = pd.DataFrame()
    if me is not None:
        print(f"\n[4-7] 三連単 EV (Stern-Gamma, top_k=5) ...")
        try:
            rows = me.trifecta_ev_rows(X_test, top_k=5, ranker_model=ranker_model)
            g_trifecta = me.sweep_ev_rows(rows, n_samples=N_SAMPLES, ev_range=EV_RANGE)
            if not g_trifecta.empty:
                print(f"  Max: {g_trifecta['return_rate'].max():.4f} @ {g_trifecta['return_rate'].idxmax():.3f}")
        except Exception as e:
            print(f"  [ERROR] 三連単: {e}")

    g_exacta = pd.DataFrame()
    if me is not None:
        print(f"\n[4-8] 馬単 EV (Stern-Gamma) ...")
        try:
            rows = me.exacta_ev_rows(X_test, ranker_model=ranker_model)
            g_exacta = me.sweep_ev_rows(rows, n_samples=N_SAMPLES, ev_range=EV_RANGE)
            if not g_exacta.empty:
                print(f"  Max: {g_exacta['return_rate'].max():.4f} @ {g_exacta['return_rate'].idxmax():.3f}")
        except Exception as e:
            print(f"  [ERROR] 馬単: {e}")

    g_wide = pd.DataFrame()
    if me is not None:
        print(f"\n[4-9] ワイド EV (Stern-Gamma, top_k=8) ...")
        try:
            rows = me.wide_ev_rows(X_test, top_k=8, ranker_model=ranker_model)
            g_wide = me.sweep_ev_rows(rows, n_samples=N_SAMPLES, ev_range=EV_RANGE)
            if not g_wide.empty:
                print(f"  Max: {g_wide['return_rate'].max():.4f} @ {g_wide['return_rate'].idxmax():.3f}")
        except Exception as e:
            print(f"  [ERROR] ワイド: {e}")

    # ==================================================================
    # 5. Graphs
    # ==================================================================
    print("\n[5] グラフを作成・保存中...")

    colors_odds = {"10": "#2196F3", "20": "#4CAF50", "30": "#FF9800", "50": "#E91E63"}

    # --- 5-1. 全戦略比較: 回収率 vs min_ev ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Harville
    for max_odds, g in ev_harville_results.items():
        if not g.empty:
            ax.plot(g.index, g["return_rate"],
                    label=f"Harville (odds<={max_odds})",
                    color=colors_odds[str(max_odds)], linewidth=1.5, linestyle="--", alpha=0.6)

    # WinModel
    for max_odds, g in ev_win_results.items():
        if not g.empty:
            ax.plot(g.index, g["return_rate"],
                    label=f"WinModel (odds<={max_odds})",
                    color=colors_odds[str(max_odds)], linewidth=2.5)
            ax.fill_between(g.index,
                            g["return_rate"] - g["std"],
                            g["return_rate"] + g["std"],
                            alpha=0.12, color=colors_odds[str(max_odds)])

    # Kelly
    if not g_kelly_win.empty:
        ax.plot(g_kelly_win.index, g_kelly_win["return_rate"],
                label="Kelly 1/4 WinModel (odds<=20)",
                color="#9C27B0", linewidth=2.5)
        ax.fill_between(g_kelly_win.index,
                        g_kelly_win["return_rate"] - g_kelly_win["std"],
                        g_kelly_win["return_rate"] + g_kelly_win["std"],
                        alpha=0.12, color="#9C27B0")

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="100%")
    ax.set_xlabel("min EV", fontsize=12)
    ax.set_ylabel("Return Rate", fontsize=12)
    ax.set_title("All Strategies: Return Rate vs min_ev", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = local_paths.RESULTS_DIR / "all_strategies_return_rate.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    checkpoint_upload(path1)
    print(f"  -> {path1}")

    # --- 5-2. Return Rate vs n_bets ---
    fig, ax = plt.subplots(figsize=(14, 8))

    if not g_tansho.empty:
        ax.plot(g_tansho["n_bets"], g_tansho["return_rate"],
                label="z-score", color="gray", linestyle="--", linewidth=1.5, alpha=0.6)

    for max_odds, g in ev_harville_results.items():
        if not g.empty:
            ax.plot(g["n_bets"], g["return_rate"],
                    label=f"Harv (odds<={max_odds})",
                    color=colors_odds[str(max_odds)], linestyle="--", linewidth=1.2, alpha=0.5)

    for max_odds, g in ev_win_results.items():
        if not g.empty:
            ax.plot(g["n_bets"], g["return_rate"],
                    label=f"WinModel (odds<={max_odds})",
                    color=colors_odds[str(max_odds)], linewidth=2.5)

    # Benter
    for max_odds, g in ev_benter_results.items():
        if not g.empty:
            ax.plot(g["n_bets"], g["return_rate"],
                    label=f"Benter (odds<={max_odds})",
                    color="#00BCD4", linewidth=2.5, linestyle="-.")

    if not g_kelly_win.empty:
        ax.plot(g_kelly_win["n_bets"], g_kelly_win["return_rate"],
                label="Kelly WinModel",
                color="#9C27B0", linewidth=2.5)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("n_bets", fontsize=12)
    ax.set_ylabel("Return Rate", fontsize=12)
    ax.set_title("Return Rate vs n_bets (All Strategies)", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = local_paths.RESULTS_DIR / "all_strategies_vs_nbets.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    checkpoint_upload(path2)
    print(f"  -> {path2}")

    # --- 5-3. WinModel vs Harville 直接比較 (odds<=20) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    if 20 in ev_harville_results and not ev_harville_results[20].empty:
        g_h20 = ev_harville_results[20]
        ax.plot(g_h20.index, g_h20["return_rate"],
                label="Harville (odds<=20)", color="#4CAF50", linestyle="--", linewidth=2)
        ax.fill_between(g_h20.index, g_h20["return_rate"] - g_h20["std"],
                        g_h20["return_rate"] + g_h20["std"], alpha=0.12, color="#4CAF50")
    if 20 in ev_win_results and not ev_win_results[20].empty:
        g_w20 = ev_win_results[20]
        ax.plot(g_w20.index, g_w20["return_rate"],
                label="WinModel (odds<=20)", color="#FF5722", linewidth=2.5)
        ax.fill_between(g_w20.index, g_w20["return_rate"] - g_w20["std"],
                        g_w20["return_rate"] + g_w20["std"], alpha=0.12, color="#FF5722")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="100%")
    ax.set_xlabel("min_ev", fontsize=12)
    ax.set_ylabel("Return Rate", fontsize=12)
    ax.set_title("Harville vs WinModel (odds<=20)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = local_paths.RESULTS_DIR / "harville_vs_winmodel.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    checkpoint_upload(path3)
    print(f"  -> {path3}")

    # --- 5-4. 三連系券種比較 ---
    fig, ax = plt.subplots(figsize=(14, 8))
    exotic_results = [
        ("三連複", g_trio, "#E91E63", 2.5, "-"),
        ("三連単", g_trifecta, "#FF5722", 2.5, "--"),
        ("馬単", g_exacta, "#795548", 2.0, "-."),
        ("ワイド", g_wide, "#009688", 2.0, ":"),
    ]
    for label, g, color, lw, ls in exotic_results:
        if not g.empty:
            ax.plot(g.index, g["return_rate"],
                    label=label, color=color, linewidth=lw, linestyle=ls)
            if "std" in g.columns:
                ax.fill_between(g.index,
                                g["return_rate"] - g["std"],
                                g["return_rate"] + g["std"],
                                alpha=0.1, color=color)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="100%")
    ax.set_xlabel("min EV", fontsize=12)
    ax.set_ylabel("Return Rate", fontsize=12)
    ax.set_title("三連系券種 EV戦略 Return Rate vs min_ev (Stern-Gamma)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path4 = local_paths.RESULTS_DIR / "exotic_strategies_return_rate.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    checkpoint_upload(path4)
    print(f"  -> {path4}")

    # ==================================================================
    # 6. 数値サマリ
    # ==================================================================
    summary = {}
    if not g_tansho.empty:
        summary["z-score"] = {
            "max_return_rate": g_tansho["return_rate"].max(),
            "best_threshold": g_tansho["return_rate"].idxmax(),
            "n_bets_at_best": g_tansho.loc[g_tansho["return_rate"].idxmax(), "n_bets"],
        }
    for max_odds, g in ev_harville_results.items():
        if not g.empty:
            summary[f"Harville(odds<={max_odds})"] = {
                "max_return_rate": g["return_rate"].max(),
                "best_threshold": g["return_rate"].idxmax(),
                "n_bets_at_best": g.loc[g["return_rate"].idxmax(), "n_bets"],
            }
    for max_odds, g in ev_win_results.items():
        if not g.empty:
            summary[f"WinModel(odds<={max_odds})"] = {
                "max_return_rate": g["return_rate"].max(),
                "best_threshold": g["return_rate"].idxmax(),
                "n_bets_at_best": g.loc[g["return_rate"].idxmax(), "n_bets"],
            }
    for max_odds, g in ev_benter_results.items():
        if not g.empty:
            summary[f"Benter(odds<={max_odds})"] = {
                "max_return_rate": g["return_rate"].max(),
                "best_threshold": g["return_rate"].idxmax(),
                "n_bets_at_best": g.loc[g["return_rate"].idxmax(), "n_bets"],
            }
    if not g_kelly_win.empty:
        summary["Kelly-WinModel(odds<=20)"] = {
            "max_return_rate": g_kelly_win["return_rate"].max(),
            "best_threshold": g_kelly_win["return_rate"].idxmax(),
            "n_bets_at_best": g_kelly_win.loc[g_kelly_win["return_rate"].idxmax(), "n_bets"],
        }

    # 三連系のサマリ追加
    exotic_summary = [
        ("三連複-EV", g_trio),
        ("三連単-EV", g_trifecta),
        ("馬単-EV", g_exacta),
        ("ワイド-EV", g_wide),
    ]
    for label, g in exotic_summary:
        if not g.empty:
            summary[label] = {
                "max_return_rate": g["return_rate"].max(),
                "best_threshold": g["return_rate"].idxmax(),
                "n_bets_at_best": g.loc[g["return_rate"].idxmax(), "n_bets"],
            }

    summary_df = pd.DataFrame(summary).T
    summary_path = local_paths.RESULTS_DIR / "simulation_summary.csv"
    summary_df.to_csv(summary_path, encoding="utf-8-sig")
    checkpoint_upload(summary_path)
    print(f"\n  Summary saved: {summary_path}")
    print(summary_df.to_string())

    # Practical range
    print("\n  === Practical Range (500<=n_bets<=5000) ===")
    all_results = [("z-score", g_tansho)]
    all_results += [(f"Harv(odds<={k})", v) for k, v in ev_harville_results.items()]
    all_results += [(f"Win(odds<={k})", v) for k, v in ev_win_results.items()]
    all_results += [(f"Benter(odds<={k})", v) for k, v in ev_benter_results.items()]
    if not g_kelly_win.empty:
        all_results.append(("Kelly-Win", g_kelly_win))
    all_results += [(l, g) for l, g in exotic_summary]

    for label, g in all_results:
        if not g.empty:
            practical = g[(g["n_bets"] >= 500) & (g["n_bets"] <= 5000)]
            if not practical.empty:
                best_row = practical.loc[practical["return_rate"].idxmax()]
                print(f"    {label}: rate={best_row['return_rate']:.4f}, "
                      f"n_bets={best_row['n_bets']:.0f}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Results: {local_paths.RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

