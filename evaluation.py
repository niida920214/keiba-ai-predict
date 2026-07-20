"""
evaluation.py – 回収率シミュレーション & モデル評価
====================================================
Return        : 払い戻し表データの加工・各券種プロパティ
ModelEvaluator: モデルの予測 → 各券種シミュレーション
gain          : threshold を走査して回収率を計算
plot          : 標準偏差帯付き回収率プロット
"""

from itertools import combinations, permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from scraper import REQUEST_INTERVAL, USER_AGENTS
from scraper import Return as ScraperReturn
from scraper import update_data


# =====================================================================
# Return（払い戻し表の加工）
# =====================================================================
class Return:
    """
    払い戻し表データを（レースID, 勝った馬番, 払戻金）の形に加工する。
    各券種は property としてアクセス可能。
    """

    def __init__(self, return_tables: pd.DataFrame):
        self.return_tables = return_tables

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(race_id_list):
        """scraper.Return.scrape に委譲（requests.get + USER_AGENTS 対応済み）。"""
        return ScraperReturn.scrape(race_id_list)

    # -----------------------------------------------------------------
    # 券種プロパティ
    # -----------------------------------------------------------------
    @property
    def fukusho(self):
        fukusho = self.return_tables[self.return_tables[0] == "複勝"][[1, 2]]
        wins = fukusho[1].str.split("br", expand=True)[[0, 1, 2]]
        wins.columns = ["win_0", "win_1", "win_2"]
        returns = fukusho[2].str.split("br", expand=True)[[0, 1, 2]]
        returns.columns = ["return_0", "return_1", "return_2"]
        df = pd.concat([wins, returns], axis=1)
        for column in df.columns:
            df[column] = df[column].str.replace(",", "")
        return df.fillna(0).astype(int)

    @property
    def tansho(self):
        tansho = self.return_tables[self.return_tables[0] == "単勝"][[1, 2]]
        tansho.columns = ["win", "return"]
        for column in tansho.columns:
            tansho[column] = pd.to_numeric(tansho[column], errors="coerce")
        return tansho

    @property
    def umaren(self):
        umaren = self.return_tables[self.return_tables[0] == "馬連"][[1, 2]]
        wins = umaren[1].str.split("-", expand=True)[[0, 1]].add_prefix("win_")
        return_ = umaren[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

    @property
    def umatan(self):
        umatan = self.return_tables[self.return_tables[0] == "馬単"][[1, 2]]
        wins = umatan[1].str.split("→", expand=True)[[0, 1]].add_prefix("win_")
        return_ = umatan[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

    @property
    def wide(self):
        wide = self.return_tables[self.return_tables[0] == "ワイド"][[1, 2]]
        wins = wide[1].str.split("br", expand=True)[[0, 1, 2]]
        wins = wins.stack().str.split("-", expand=True).add_prefix("win_")
        return_ = wide[2].str.split("br", expand=True)[[0, 1, 2]]
        return_ = return_.stack().rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(
            lambda x: pd.to_numeric(x.str.replace(",", ""), errors="coerce")
        )

    @property
    def sanrentan(self):
        rentan = self.return_tables[self.return_tables[0] == "三連単"][[1, 2]]
        wins = rentan[1].str.split("→", expand=True)[[0, 1, 2]].add_prefix("win_")
        return_ = rentan[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

    @property
    def sanrenpuku(self):
        renpuku = self.return_tables[self.return_tables[0] == "三連複"][[1, 2]]
        wins = renpuku[1].str.split("-", expand=True)[[0, 1, 2]].add_prefix("win_")
        return_ = renpuku[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))


# =====================================================================
# ModelEvaluator
# =====================================================================
class ModelEvaluator:
    """学習済みモデルを使って各券種の回収率をシミュレーションする。"""

    def __init__(self, model, return_tables_path_list, win_model=None):
        self.model = model
        self.win_model = win_model  # Phase 2: 1着予測専用モデル (optional)
        self.rt = Return.read_pickle(return_tables_path_list)
        self.fukusho = self.rt.fukusho
        self.tansho = self.rt.tansho
        self.umaren = self.rt.umaren
        self.umatan = self.rt.umatan
        self.wide = self.rt.wide
        self.sanrentan = self.rt.sanrentan
        self.sanrenpuku = self.rt.sanrenpuku
        self.proba = None
        self.sample = None

    # -----------------------------------------------------------------
    # 予測
    # -----------------------------------------------------------------
    def predict_proba(self, X, train=True, std=True, minmax=False):
        """3 着以内に入る確率を予測する。"""
        # CalibratedClassifierCV の場合はキャリブレーション済みモデルをそのまま使用
        # （以前はベースモデルに戻していたが、キャリブレーション結果を活用する）
        if train:
            X_feat = X.drop(["単勝"], axis=1)
        else:
            X_feat = X

        proba = pd.Series(
            self.model.predict_proba(X_feat)[:, 1],
            index=X.index,
        )
        if std:
            # レース内で標準化（レース内偏差値）
            standard_scaler = lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0.0
            proba = proba.groupby(level=0).transform(standard_scaler)
        if minmax:
            proba = (proba - proba.min()) / (proba.max() - proba.min())
        return proba

    def predict_proba_calibrated(self, X):
        """
        キャリブレーション済み P(3着以内) を返す（標準化なし、生確率）。
        EVベース戦略で使用する。
        """
        X_feat = X.drop(["単勝"], axis=1) if "単勝" in X.columns else X
        proba = pd.Series(
            self.model.predict_proba(X_feat)[:, 1],
            index=X.index,
        )
        return proba

    def predict_win_proba(self, X, harville_coeff=1.414):
        """
        Harville 近似 (Power Law) で P(1着) を推定する。

        従来の単純な割り算 (P3 / c) はレース内正規化によって係数が相殺されるため、
        数学的に有効な非線形変換 (Power Law) を用いて推定を行う。
        tune_harville.py で算出された最適化指数 beta=1.414 をデフォルトとする。

        Parameters
        ----------
        X : pd.DataFrame
            テストデータ（単勝列を含む）
        harville_coeff : float
            3着以内確率→1着確率の変換指数 (デフォルト: 1.414)

        Returns
        -------
        pd.Series
            レース内正規化済み P(1着) の推定値
        """
        p3 = self.predict_proba_calibrated(X)
        # Power Law 変換: P(1着) ∝ P(3着以内) ^ harville_coeff
        p1_raw = p3 ** harville_coeff
        # レース内で正規化: 合計確率を 1 にする
        p1_normalized = p1_raw.groupby(level=0).transform(lambda x: x / x.sum())
        return p1_normalized

    def predict_win_proba_direct(self, X):
        """
        Win モデルで直接 P(1着) を予測する（Harville近似不要）。

        win_model が設定されていない場合は None を返す。
        """
        if self.win_model is None:
            return None
        X_feat = X.drop(["単勝"], axis=1) if "単勝" in X.columns else X
        proba = pd.Series(
            self.win_model.predict_proba(X_feat)[:, 1],
            index=X.index,
        )
        # レース内正規化: 合計 = 1
        proba = proba.groupby(level=0).transform(lambda x: x / x.sum())
        return proba

    def predict(self, X, threshold=0.5):
        """threshold 以上なら 1, 未満なら 0 を返す。"""
        y_pred = self.predict_proba(X)
        self.proba = y_pred
        return [0 if p < threshold else 1 for p in y_pred]

    def score(self, y_true, X):
        return roc_auc_score(y_true, self.predict_proba(X))

    def feature_importance(self, X, n_display=20):
        importances = pd.DataFrame(
            {
                "features": X.columns,
                "importance": self.model.feature_importances_,
            }
        )
        return importances.sort_values("importance", ascending=False)[:n_display]

    def pred_table(self, X, threshold=0.5, bet_only=True):
        pred_table = X.copy()[["馬番", "単勝"]]
        pred_table["pred"] = self.predict(X, threshold)
        pred_table["score"] = self.proba
        if bet_only:
            return pred_table[pred_table["pred"] == 1][["馬番", "単勝", "score"]]
        else:
            return pred_table[["馬番", "単勝", "score", "pred"]]

    # -----------------------------------------------------------------
    # bet（的中判定）
    # -----------------------------------------------------------------
    def bet(self, race_id, kind, umaban, amount):
        """
        的中判定を行い、払い戻し金額を返す。
        的中しなかった場合は賭け金（amount）がマイナスとして返る。
        """
        return_ = 0
        if kind == "fukusho":
            rt_1R = self.fukusho.loc[race_id]
            is_win = (rt_1R[["win_0", "win_1", "win_2"]] == umaban)
            # handle pandas boolean dataframe
            if isinstance(is_win, pd.DataFrame):
                is_win = is_win.values
            return_ = (
                is_win
                * rt_1R[["return_0", "return_1", "return_2"]].values
                * amount
                / 100
            )
            return_ = np.sum(return_)
        elif kind == "tansho":
            rt_1R = self.tansho.loc[race_id]
            return_ = (rt_1R["win"] == umaban) * rt_1R["return"] * amount / 100
        elif kind == "umaren":
            rt_1R = self.umaren.loc[race_id]
            return_ = (
                (set(rt_1R[["win_0", "win_1"]]) == set(umaban))
                * rt_1R["return"]
                / 100
                * amount
            )
        elif kind == "umatan":
            rt_1R = self.umatan.loc[race_id]
            return_ = (
                (list(rt_1R[["win_0", "win_1"]]) == list(umaban))
                * rt_1R["return"]
                / 100
                * amount
            )
        elif kind == "wide":
            rt_1R = self.wide.loc[race_id]
            return_ = (
                (
                    rt_1R[["win_0", "win_1"]].apply(
                        lambda x: set(x) == set(umaban), axis=1
                    )
                )
                * rt_1R["return"]
                / 100
                * amount
            )
            return_ = return_.sum()
        elif kind == "sanrentan":
            rt_1R = self.sanrentan.loc[race_id]
            return_ = (
                (list(rt_1R[["win_0", "win_1", "win_2"]]) == list(umaban))
                * rt_1R["return"]
                / 100
                * amount
            )
        elif kind == "sanrenpuku":
            rt_1R = self.sanrenpuku.loc[race_id]
            return_ = (
                (set(rt_1R[["win_0", "win_1", "win_2"]]) == set(umaban))
                * rt_1R["return"]
                / 100
                * amount
            )
        # NaN・負値は「外れ」扱い（賭け金は没収 → 払い戻し 0）
        try:
            return_ = float(return_)
            if not (return_ >= 0):  # NaN と負値の両方をキャッチ
                return_ = 0.0
        except (TypeError, ValueError):
            return_ = 0.0
        return return_

    # -----------------------------------------------------------------
    # 各券種の回収率計算
    # -----------------------------------------------------------------
    def fukusho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum(
                    [
                        self.bet(race_id, "fukusho", umaban, 1)
                        for umaban in preds["馬番"]
                    ]
                )
            )
        return_rate = np.sum(return_list) / n_bets
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return n_bets, return_rate, n_hits, std

    def tansho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        self.sample = pred_table
        n_bets = len(pred_table)
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum(
                    [self.bet(race_id, "tansho", umaban, 1) for umaban in preds["馬番"]]
                )
            )
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def tansho_return_proper(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum(
                    preds.apply(
                        lambda x: self.bet(race_id, "tansho", x["馬番"], 1 / x["単勝"]),
                        axis=1,
                    )
                )
            )
        bet_money = (1 / pred_table["単勝"]).sum()
        std = np.std(return_list) * np.sqrt(len(return_list)) / bet_money
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / bet_money
        return n_bets, return_rate, n_hits, std

    # -----------------------------------------------------------------
    # EVベース単勝戦略 (Phase 1)
    # -----------------------------------------------------------------
    def _build_ev_table(self, X, max_odds=30.0, harville_coeff=2.4, use_harville=False):
        """
        EVベース戦略用のテーブルを構築する。

        Returns
        -------
        pd.DataFrame
            columns: 馬番, 単勝(=odds), p_win, ev
        """
        # Win モデルが利用可能なら直接予測、なければ Harville 近似
        p_win_direct = self.predict_win_proba_direct(X)
        if p_win_direct is not None and not use_harville:
            p_win = p_win_direct
        else:
            p_win = self.predict_win_proba(X, harville_coeff=harville_coeff)
        ev_table = X[["馬番", "単勝"]].copy()
        ev_table["p_win"] = p_win
        ev_table["ev"] = ev_table["p_win"] * ev_table["単勝"]
        # オッズ上限フィルタ: 大穴を排除
        ev_table = ev_table[ev_table["単勝"] <= max_odds]
        return ev_table

    def tansho_return_ev(self, X, min_ev=1.0, max_odds=30.0, harville_coeff=2.4):
        """
        EVベース単勝戦略: EV > min_ev かつ odds ≤ max_odds の馬に均等額を賭ける。

        Parameters
        ----------
        X : pd.DataFrame
        min_ev : float
            期待値の下限閾値（1.0 = 期待値プラス）
        max_odds : float
            オッズ上限（これを超えるロングショットは除外）
        harville_coeff : float
            Harville 近似の係数

        Returns
        -------
        (n_bets, return_rate, n_hits, std)
        """
        ev_table = self._build_ev_table(X, max_odds, harville_coeff)
        # EV > min_ev の馬を選出
        bet_table = ev_table[ev_table["ev"] >= min_ev]

        n_bets = len(bet_table)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0

        return_list = []
        for race_id, preds in bet_table.groupby(level=0):
            return_list.append(
                np.sum(
                    [self.bet(race_id, "tansho", umaban, 1) for umaban in preds["馬番"]]
                )
            )
        if not return_list:
            return 0, 0.0, 0, 0.0

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def tansho_return_ev_kelly(self, X, min_ev=1.0, max_odds=30.0,
                               harville_coeff=2.4, kelly_fraction=0.25):
        """
        Kelly基準ベースの単勝戦略:
        EV > min_ev の馬にフラクショナル Kelly で賭け金を配分する。

        Kelly 基準:
            f* = (p * odds - 1) / (odds - 1)
            実際の賭け金 = f* × kelly_fraction （過学習防止）

        Parameters
        ----------
        kelly_fraction : float
            Kelly基準の縮小率（0.25 = 1/4 Kelly、保守的推奨値）
        """
        ev_table = self._build_ev_table(X, max_odds, harville_coeff)
        bet_table = ev_table[ev_table["ev"] >= min_ev].copy()

        n_bets = len(bet_table)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0

        # Kelly基準による賭け金計算
        bet_table["kelly_f"] = (
            (bet_table["p_win"] * bet_table["単勝"] - 1)
            / (bet_table["単勝"] - 1)
        ).clip(lower=0) * kelly_fraction

        # 賭け金 0 の馬は除外
        bet_table = bet_table[bet_table["kelly_f"] > 0]
        n_bets = len(bet_table)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0

        total_bet = bet_table["kelly_f"].sum()
        return_list = []
        for race_id, preds in bet_table.groupby(level=0):
            return_list.append(
                np.sum(
                    preds.apply(
                        lambda x: self.bet(race_id, "tansho", x["馬番"], x["kelly_f"]),
                        axis=1,
                    )
                )
            )
        if not return_list:
            return 0, 0.0, 0, 0.0

        std = np.std(return_list) * np.sqrt(len(return_list)) / total_bet
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / total_bet
        return n_bets, return_rate, n_hits, std

    def umaren_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku["馬番"], 2):
                    return_ += self.bet(race_id, "umaren", umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def umatan_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in permutations(preds_jiku["馬番"], 2):
                    return_ += self.bet(race_id, "umatan", umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def wide_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku["馬番"], 2):
                    return_ += self.bet(race_id, "wide", umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def sanrentan_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) < 3:
                continue
            else:
                for umaban in permutations(preds_jiku["馬番"], 3):
                    return_ += self.bet(race_id, "sanrentan", umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def sanrenpuku_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) < 3:
                continue
            else:
                for umaban in combinations(preds_jiku["馬番"], 3):
                    return_ += self.bet(race_id, "sanrenpuku", umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        if n_bets == 0:
            return 0, 0.0, 0, 0.0
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    # -----------------------------------------------------------------
    # 流し
    # -----------------------------------------------------------------
    def umaren_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values("score", ascending=False).iloc[
                    1 : (n_aite + 1)
                ]["馬番"]
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id,
                        "umaren",
                        [preds_jiku["馬番"].values[0], x],
                        1,
                    )
                ).sum()
                n_bets += n_aite
                return_list.append(return_)
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku["馬番"], 2):
                    return_ += self.bet(race_id, "umaren", umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def umatan_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values("score", ascending=False).iloc[
                    1 : (n_aite + 1)
                ]["馬番"]
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id,
                        "umatan",
                        [preds_jiku["馬番"].values[0], x],
                        1,
                    )
                ).sum()
                n_bets += n_aite
            elif len(preds_jiku) >= 2:
                for umaban in permutations(preds_jiku["馬番"], 2):
                    return_ += self.bet(race_id, "umatan", umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def wide_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values("score", ascending=False).iloc[
                    1 : (n_aite + 1)
                ]["馬番"]
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id,
                        "wide",
                        [preds_jiku["馬番"].values[0], x],
                        1,
                    )
                ).sum()
                n_bets += len(preds_aite)
                return_list.append(return_)
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku["馬番"], 2):
                    return_ += self.bet(race_id, "wide", umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def sanrentan_nagashi(self, X, threshold=1.5, n_aite=7):
        pred_table = self.pred_table(X, threshold, bet_only=False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            preds_jiku = preds.query("pred == 1")
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) == 2:
                preds_aite = preds.sort_values("score", ascending=False).iloc[
                    2 : (n_aite + 2)
                ]["馬番"]
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id,
                        "sanrentan",
                        np.append(preds_jiku["馬番"].values, x),
                        1,
                    )
                ).sum()
                n_bets += len(preds_aite)
                return_list.append(return_)
            elif len(preds_jiku) >= 3:
                return_ = 0
                for umaban in permutations(preds_jiku["馬番"], 3):
                    return_ += self.bet(race_id, "sanrentan", umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    # -----------------------------------------------------------------
    # EVベース三連系戦略 (Phase 5: Stern-Gamma 補正付き)
    # -----------------------------------------------------------------
    def _compute_race_p_win(self, X_race, use_ranker=True, ranker_model=None):
        """レース単位で P(Win) を算出する。

        Parameters
        ----------
        X_race : pd.DataFrame
            1レース分のテストデータ
        use_ranker : bool
            True: Ranker → Softmax, False: Win/Classifier → Harville
        ranker_model : LGBMRanker, optional
            外部から渡すランカーモデル

        Returns
        -------
        p_win : np.ndarray
        umaban : np.ndarray
        """
        from position_probability import (
            compute_race_p_win_from_ranker,
            compute_race_p_win_from_classifier,
        )

        if use_ranker and ranker_model is not None:
            return compute_race_p_win_from_ranker(ranker_model, X_race)
        elif self.win_model is not None:
            return compute_race_p_win_from_classifier(self.win_model, X_race)
        else:
            return compute_race_p_win_from_classifier(self.model, X_race)

    def trio_return_ev(
        self, X, min_ev=1.0, top_k=6, ranker_model=None, stern_r=None
    ):
        """三連複 EVベース戦略。

        Stern-Gamma 補正付き Harville で三連複の確率を推定し、
        EV > min_ev の組み合わせに均等額を賭ける。

        学術的根拠:
            Stern (1990): ガンマ分布モデルによる着順確率推定
            Lo & Bacon-Shone (1994): Harville バイアスの実証的補正

        Parameters
        ----------
        X : pd.DataFrame
        min_ev : float
            EV下限閾値
        top_k : int
            レースあたり上位何頭を候補にするか
        ranker_model : LGBMRanker, optional
        stern_r : float, optional

        Returns
        -------
        (n_bets, return_rate, n_hits, std)
        """
        from position_probability import PositionProbabilityEstimator

        if stern_r is None:
            stern_r = PositionProbabilityEstimator.load_stern_r()

        unique_races = X.index.unique()
        n_bets = 0
        return_list = []

        for race_id in unique_races:
            race_X = X.loc[[race_id]]
            if len(race_X) < 3:
                continue

            try:
                p_win, umaban = self._compute_race_p_win(
                    race_X,
                    use_ranker=(ranker_model is not None),
                    ranker_model=ranker_model,
                )
            except Exception:
                continue

            ppe = PositionProbabilityEstimator(p_win, umaban, stern_r=stern_r)

            # 上位 top_k 頭の組み合わせ
            top_idx = np.argsort(-p_win)[:top_k]
            top_uma = umaban[top_idx]

            race_return = 0.0
            race_bets = 0
            for combo in combinations(top_uma, 3):
                prob = ppe.p_trio(int(combo[0]), int(combo[1]), int(combo[2]))
                # 払戻表からオッズを逆算 (配当/100)
                try:
                    rt = self.sanrenpuku.loc[race_id]
                    actual_return = self.bet(
                        race_id, "sanrenpuku", set(int(c) for c in combo), 1
                    )
                    if actual_return > 0:
                        implied_odds = actual_return * 100
                    else:
                        implied_odds = 0
                    ev = prob * implied_odds if implied_odds > 0 else 0
                except Exception:
                    ev = 0

                if ev >= min_ev:
                    ret = self.bet(
                        race_id, "sanrenpuku",
                        set(int(c) for c in combo), 1
                    )
                    race_return += ret
                    race_bets += 1

            if race_bets > 0:
                n_bets += race_bets
                return_list.append(race_return)

        if n_bets == 0:
            return 0, 0.0, 0, 0.0

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets if n_bets > 0 else 0.0
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def trifecta_return_ev(
        self, X, min_ev=1.0, top_k=6, ranker_model=None, stern_r=None
    ):
        """三連単 EVベース戦略。

        Stern-Gamma 補正で三連単の確率を推定し、EV > min_ev に賭ける。

        Returns
        -------
        (n_bets, return_rate, n_hits, std)
        """
        from position_probability import PositionProbabilityEstimator

        if stern_r is None:
            stern_r = PositionProbabilityEstimator.load_stern_r()

        unique_races = X.index.unique()
        n_bets = 0
        return_list = []

        for race_id in unique_races:
            race_X = X.loc[[race_id]]
            if len(race_X) < 3:
                continue

            try:
                p_win, umaban = self._compute_race_p_win(
                    race_X,
                    use_ranker=(ranker_model is not None),
                    ranker_model=ranker_model,
                )
            except Exception:
                continue

            ppe = PositionProbabilityEstimator(p_win, umaban, stern_r=stern_r)

            top_idx = np.argsort(-p_win)[:top_k]
            top_uma = umaban[top_idx]

            race_return = 0.0
            race_bets = 0
            for perm in permutations(top_uma, 3):
                prob = ppe.p_trifecta(int(perm[0]), int(perm[1]), int(perm[2]))
                try:
                    actual_return = self.bet(
                        race_id, "sanrentan", list(int(p) for p in perm), 1
                    )
                    if actual_return > 0:
                        implied_odds = actual_return * 100
                    else:
                        implied_odds = 0
                    ev = prob * implied_odds if implied_odds > 0 else 0
                except Exception:
                    ev = 0

                if ev >= min_ev:
                    ret = self.bet(
                        race_id, "sanrentan",
                        list(int(p) for p in perm), 1
                    )
                    race_return += ret
                    race_bets += 1

            if race_bets > 0:
                n_bets += race_bets
                return_list.append(race_return)

        if n_bets == 0:
            return 0, 0.0, 0, 0.0

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets if n_bets > 0 else 0.0
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    # -----------------------------------------------------------------
    # 高速版: EVテーブルを1度だけ計算し、閾値スイープを使い回す
    #
    # 元の trio/trifecta/exacta/wide_return_ev は gain_ev から n_samples 回
    # （既定25〜50回）呼ばれ、そのたびに全レース×全組み合わせのStern-Gamma確率を
    # 計算し直していた（min_ev はフィルタ条件にしか使わないため本来不要な再計算）。
    # ここでは同じ数式・同じ分岐ロジックを保ったまま、min_ev に依存しない部分
    # （race_id, ev, actual_return）だけを1回計算してテーブル化し、
    # 閾値スイープは軽量な集計のみで済ませる。
    # -----------------------------------------------------------------
    _EXOTIC_BET_ARG = {
        "sanrenpuku": lambda combo: set(int(c) for c in combo),
        "sanrentan": lambda combo: list(int(c) for c in combo),
        "umatan": lambda combo: list(int(c) for c in combo),
        "wide": lambda combo: [int(combo[0]), int(combo[1])],
    }

    def _build_exotic_ev_rows(
        self, X, bet_kind, prob_fn, combo_size, ordered,
        top_k=None, ranker_model=None, stern_r=None,
    ) -> pd.DataFrame:
        """(race_id, ev, actual_return) のテーブルを1回だけ計算する。"""
        from position_probability import PositionProbabilityEstimator

        if stern_r is None:
            stern_r = PositionProbabilityEstimator.load_stern_r()
        combo_iter = permutations if ordered else combinations
        arg_fn = self._EXOTIC_BET_ARG[bet_kind]

        rows = []
        for race_id in X.index.unique():
            race_X = X.loc[[race_id]]
            if len(race_X) < combo_size:
                continue
            try:
                p_win, umaban = self._compute_race_p_win(
                    race_X, use_ranker=(ranker_model is not None),
                    ranker_model=ranker_model,
                )
            except Exception:
                continue

            ppe = PositionProbabilityEstimator(p_win, umaban, stern_r=stern_r)

            if top_k is not None:
                candidates = umaban[np.argsort(-p_win)[:top_k]]
            else:
                candidates = umaban

            for combo in combo_iter(candidates, combo_size):
                prob = prob_fn(ppe, *[int(c) for c in combo])
                try:
                    actual_return = self.bet(race_id, bet_kind, arg_fn(combo), 1)
                    ev = prob * actual_return * 100 if actual_return > 0 else 0.0
                except Exception:
                    actual_return, ev = 0.0, 0.0
                rows.append((race_id, ev, actual_return))

        return pd.DataFrame(rows, columns=["race_id", "ev", "actual_return"])

    @staticmethod
    def sweep_ev_rows(ev_rows: pd.DataFrame, n_samples=50, ev_range=None) -> pd.DataFrame:
        """事前計算済みのEVテーブルを閾値でスイープする（gain_evと同じ集計式）。"""
        if ev_range is None:
            ev_range = [0.8, 2.0]
        lo, hi = ev_range
        gain_dict = {}
        if ev_rows.empty:
            return pd.DataFrame()
        for i in range(n_samples):
            min_ev = lo + (hi - lo) * i / n_samples
            sub = ev_rows[ev_rows["ev"] >= min_ev]
            if sub.empty:
                continue
            grp = sub.groupby("race_id")["actual_return"].agg(["sum", "count"])
            n_bets = int(grp["count"].sum())
            if n_bets <= 2:
                continue
            return_list = grp["sum"].to_numpy()
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
            n_hits = int(np.sum(return_list > 0))
            return_rate = float(np.sum(return_list) / n_bets)
            gain_dict[min_ev] = {
                "return_rate": return_rate, "n_hits": n_hits,
                "std": std, "n_bets": n_bets,
            }
        return pd.DataFrame(gain_dict).T

    @staticmethod
    def sweep_ev_rows_per_bet(ev_rows: pd.DataFrame, n_samples=50, ev_range=None) -> pd.DataFrame:
        """事前計算済みのEVテーブルを閾値でスイープする（ベット単位の集計版）。

        sweep_ev_rows はレースごとに return を合算してから std を取るが
        （trio/trifecta/exacta/wide の元実装がそうだったため）、
        Benter-Ranker戦略の元実装は同一レース内の複数ベットを合算せず、
        ベット1件ごとに独立した値として std を計算していた。集計単位が
        違うだけで数式は同じなので、専用にこちらを用意する。
        """
        if ev_range is None:
            ev_range = [0.8, 2.0]
        lo, hi = ev_range
        gain_dict = {}
        if ev_rows.empty:
            return pd.DataFrame()
        for i in range(n_samples):
            min_ev = lo + (hi - lo) * i / n_samples
            sub = ev_rows[ev_rows["ev"] >= min_ev]
            n_bets = len(sub)
            if n_bets <= 2:
                continue
            returns = sub["actual_return"].to_numpy()
            std = np.std(returns) * np.sqrt(n_bets) / n_bets
            n_hits = int(np.sum(returns > 0))
            return_rate = float(np.sum(returns) / n_bets)
            gain_dict[min_ev] = {
                "return_rate": return_rate, "n_hits": n_hits,
                "std": std, "n_bets": n_bets,
            }
        return pd.DataFrame(gain_dict).T

    def trio_ev_rows(self, X, top_k=6, ranker_model=None, stern_r=None) -> pd.DataFrame:
        return self._build_exotic_ev_rows(
            X, "sanrenpuku", lambda ppe, a, b, c: ppe.p_trio(a, b, c),
            combo_size=3, ordered=False, top_k=top_k,
            ranker_model=ranker_model, stern_r=stern_r,
        )

    def trifecta_ev_rows(self, X, top_k=5, ranker_model=None, stern_r=None) -> pd.DataFrame:
        return self._build_exotic_ev_rows(
            X, "sanrentan", lambda ppe, a, b, c: ppe.p_trifecta(a, b, c),
            combo_size=3, ordered=True, top_k=top_k,
            ranker_model=ranker_model, stern_r=stern_r,
        )

    def exacta_ev_rows(self, X, top_k=None, ranker_model=None, stern_r=None) -> pd.DataFrame:
        return self._build_exotic_ev_rows(
            X, "umatan", lambda ppe, a, b: ppe.p_exacta(a, b),
            combo_size=2, ordered=True, top_k=top_k,
            ranker_model=ranker_model, stern_r=stern_r,
        )

    def wide_ev_rows(self, X, top_k=8, ranker_model=None, stern_r=None) -> pd.DataFrame:
        return self._build_exotic_ev_rows(
            X, "wide", lambda ppe, a, b: ppe.p_wide(a, b),
            combo_size=2, ordered=False, top_k=top_k,
            ranker_model=ranker_model, stern_r=stern_r,
        )

    def exacta_return_ev(
        self, X, min_ev=1.0, max_odds=100, ranker_model=None, stern_r=None
    ):
        """馬単 EVベース戦略。

        Returns
        -------
        (n_bets, return_rate, n_hits, std)
        """
        from position_probability import PositionProbabilityEstimator

        if stern_r is None:
            stern_r = PositionProbabilityEstimator.load_stern_r()

        unique_races = X.index.unique()
        n_bets = 0
        return_list = []

        for race_id in unique_races:
            race_X = X.loc[[race_id]]
            if len(race_X) < 2:
                continue

            try:
                p_win, umaban = self._compute_race_p_win(
                    race_X,
                    use_ranker=(ranker_model is not None),
                    ranker_model=ranker_model,
                )
            except Exception:
                continue

            ppe = PositionProbabilityEstimator(p_win, umaban, stern_r=stern_r)

            race_return = 0.0
            race_bets = 0
            for perm in permutations(umaban, 2):
                prob = ppe.p_exacta(int(perm[0]), int(perm[1]))
                try:
                    actual_return = self.bet(
                        race_id, "umatan", list(int(p) for p in perm), 1
                    )
                    if actual_return > 0:
                        implied_odds = actual_return * 100
                    else:
                        implied_odds = 0
                    ev = prob * implied_odds if implied_odds > 0 else 0
                except Exception:
                    ev = 0

                if ev >= min_ev:
                    ret = self.bet(
                        race_id, "umatan",
                        list(int(p) for p in perm), 1
                    )
                    race_return += ret
                    race_bets += 1

            if race_bets > 0:
                n_bets += race_bets
                return_list.append(race_return)

        if n_bets == 0:
            return 0, 0.0, 0, 0.0

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets if n_bets > 0 else 0.0
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def wide_return_ev(
        self, X, min_ev=1.0, top_k=8, ranker_model=None, stern_r=None
    ):
        """ワイド EVベース戦略。

        Returns
        -------
        (n_bets, return_rate, n_hits, std)
        """
        from position_probability import PositionProbabilityEstimator

        if stern_r is None:
            stern_r = PositionProbabilityEstimator.load_stern_r()

        unique_races = X.index.unique()
        n_bets = 0
        return_list = []

        for race_id in unique_races:
            race_X = X.loc[[race_id]]
            if len(race_X) < 3:
                continue

            try:
                p_win, umaban = self._compute_race_p_win(
                    race_X,
                    use_ranker=(ranker_model is not None),
                    ranker_model=ranker_model,
                )
            except Exception:
                continue

            ppe = PositionProbabilityEstimator(p_win, umaban, stern_r=stern_r)

            top_idx = np.argsort(-p_win)[:top_k]
            top_uma = umaban[top_idx]

            race_return = 0.0
            race_bets = 0
            for combo in combinations(top_uma, 2):
                prob = ppe.p_wide(int(combo[0]), int(combo[1]))
                try:
                    actual_return = self.bet(
                        race_id, "wide",
                        [int(combo[0]), int(combo[1])], 1
                    )
                    if actual_return > 0:
                        implied_odds = actual_return * 100
                    else:
                        implied_odds = 0
                    ev = prob * implied_odds if implied_odds > 0 else 0
                except Exception:
                    ev = 0

                if ev >= min_ev:
                    ret = self.bet(
                        race_id, "wide",
                        [int(combo[0]), int(combo[1])], 1
                    )
                    race_return += ret
                    race_bets += 1

            if race_bets > 0:
                n_bets += race_bets
                return_list.append(race_return)

        if n_bets == 0:
            return 0, 0.0, 0, 0.0

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets if n_bets > 0 else 0.0
        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std


# =====================================================================
# gain / plot 関数
# =====================================================================


def gain(return_func, X, n_samples=100, t_range=None):
    """
    threshold を走査して回収率・標準偏差を計算する。

    Parameters
    ----------
    return_func : callable
        ModelEvaluator の *_return / *_box メソッド
    X : pd.DataFrame
        テストデータ（単勝列を含む）
    n_samples : int
        threshold の分割数
    t_range : list[float]
        [min_threshold, max_threshold]

    Returns
    -------
    pd.DataFrame
        index=threshold, columns=[return_rate, n_hits, std, n_bets]
    """
    if t_range is None:
        t_range = [0.5, 3.5]

    gain_dict = {}
    for i in tqdm(range(n_samples), desc="gain"):
        threshold = t_range[1] * i / n_samples + t_range[0] * (1 - (i / n_samples))
        n_bets, return_rate, n_hits, std = return_func(X, threshold)
        if n_bets > 2:
            gain_dict[threshold] = {
                "return_rate": return_rate,
                "n_hits": n_hits,
                "std": std,
                "n_bets": n_bets,
            }
    return pd.DataFrame(gain_dict).T


def plot(df, label=" "):
    """標準偏差帯付き回収率プロット。"""
    plt.fill_between(
        df.index,
        y1=df["return_rate"] - df["std"],
        y2=df["return_rate"] + df["std"],
        alpha=0.3,
    )
    plt.plot(df.index, df["return_rate"], label=label)
    plt.legend()
    plt.grid(True)
