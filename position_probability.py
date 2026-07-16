"""
position_probability.py – 着順確率推定モジュール
==================================================
Plackett-Luce / Harville / Stern-Gamma に基づく着順同時確率を計算する。

学術的根拠:
    - Harville (1973): P(i,j,k) = P(i) × P(j|i won) × P(k|i,j won)
      条件付き確率は残りの馬の勝率を再正規化して算出。
      走行時間に指数分布を仮定。人気馬の2-3着確率を過大評価するバイアスあり。
    
    - Henery (1981): 走行時間に正規分布を仮定し、Harville のバイアスを緩和。
      解析的な閉形式解が存在しないため、計算コストが高い。
    
    - Stern (1990): ガンマ分布を仮定。形状パラメータ r で Harville (r=1) と
      Henery (r→∞) を連続的に補間する。
      P_corrected(i wins from S) = p_i^r / Σ_{j∈S} p_j^r
    
    - Lo & Bacon-Shone (1994): Harville バイアスのロジスティック回帰補正。
    
    - Benter (1994): 市場確率と基本モデルの統合。

本モジュールでは Stern-Gamma 補正を採用し、パラメータ r を tune_stern.py で最適化する。
"""

import pickle
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from modules.constants import local_paths


# =====================================================================
# PositionProbabilityEstimator
# =====================================================================
class PositionProbabilityEstimator:
    """各馬の着順位置確率を推定し、馬券種ごとの確率・EVを計算する。

    Parameters
    ----------
    p_win : np.ndarray
        各馬の勝率（Softmax 正規化済み、合計 ≈ 1.0）。
        インデックスは馬のレース内順序（0-indexed）。
    umaban : np.ndarray or list
        各馬の馬番（1-indexed）。p_win と同じ順序。
    stern_r : float
        Stern-Gamma 補正パラメータ。
        r = 1.0 → 標準 Harville
        r < 1.0 → 人気馬のバイアス緩和（推奨: 0.7〜0.9）
        r > 1.0 → 人気馬のバイアス強化
    """

    def __init__(
        self,
        p_win: np.ndarray,
        umaban: np.ndarray = None,
        stern_r: float = 0.82,
    ):
        self.p_win = np.asarray(p_win, dtype=np.float64)
        self.n_horses = len(self.p_win)
        self.stern_r = stern_r

        if umaban is not None:
            self.umaban = np.asarray(umaban, dtype=int)
        else:
            self.umaban = np.arange(1, self.n_horses + 1)

        # p_win を正規化（丸め誤差対策）
        total = self.p_win.sum()
        if total > 0:
            self.p_win = self.p_win / total

        # 馬番→内部インデックスのマッピング
        self._umaban_to_idx = {int(u): i for i, u in enumerate(self.umaban)}

    # -----------------------------------------------------------------
    # Stern-Gamma 補正付き条件付き確率
    # -----------------------------------------------------------------
    def _conditional_win_prob(self, idx: int, excluded: set) -> float:
        """Stern-Gamma 補正付きの条件付き勝率を計算する。

        P_Stern(i wins | excluded) = p_i^r / Σ_{j ∉ excluded} p_j^r

        Parameters
        ----------
        idx : int
            馬の内部インデックス
        excluded : set
            既に着順が確定した馬のインデックス集合

        Returns
        -------
        float
            Stern-Gamma 補正付き条件付き勝率
        """
        r = self.stern_r
        remaining_indices = [
            j for j in range(self.n_horses) if j not in excluded
        ]
        if idx not in remaining_indices:
            return 0.0

        numerator = self.p_win[idx] ** r
        denominator = sum(self.p_win[j] ** r for j in remaining_indices)

        if denominator <= 0:
            return 0.0
        return numerator / denominator

    # -----------------------------------------------------------------
    # 基本的な着順確率
    # -----------------------------------------------------------------
    def p_win_single(self, umaban: int) -> float:
        """単勝: P(umaban が1着)"""
        idx = self._umaban_to_idx.get(umaban)
        if idx is None:
            return 0.0
        return self.p_win[idx]

    def p_place(self, umaban: int) -> float:
        """複勝: P(umaban が3着以内)

        3着以内に入る確率は、全ての可能な (1着, 2着, 3着) の組み合わせで
        umaban がいずれかに含まれる確率の合計。

        計算量削減のため、補完事象 1 - P(4着以下) で計算する。
        P(4着以下) = Π_{pos=1}^{3} P(他馬が pos 着 | 上位が確定)
        を全組み合わせで合計する…は計算量が爆発するため、
        Harville 近似で P(i が k 着) を直接計算する。
        """
        idx = self._umaban_to_idx.get(umaban)
        if idx is None:
            return 0.0

        # P(1着)
        p1 = self._conditional_win_prob(idx, set())

        # P(2着) = Σ_{j≠i} P(j=1着) × P(i=2着|j won)
        p2 = 0.0
        for j in range(self.n_horses):
            if j == idx:
                continue
            p_j_wins = self._conditional_win_prob(j, set())
            p_i_second = self._conditional_win_prob(idx, {j})
            p2 += p_j_wins * p_i_second

        # P(3着) = Σ_{j≠i} Σ_{k≠i,j} P(j=1着) × P(k=2着|j) × P(i=3着|j,k)
        p3 = 0.0
        for j in range(self.n_horses):
            if j == idx:
                continue
            p_j_wins = self._conditional_win_prob(j, set())
            for k in range(self.n_horses):
                if k == idx or k == j:
                    continue
                p_k_second = self._conditional_win_prob(k, {j})
                p_i_third = self._conditional_win_prob(idx, {j, k})
                p3 += p_j_wins * p_k_second * p_i_third

        return p1 + p2 + p3

    def p_exacta(self, uma1: int, uma2: int) -> float:
        """馬単: P(uma1=1着, uma2=2着)

        Harville-Stern:
            P(uma1, uma2) = P_s(uma1 wins) × P_s(uma2 wins | uma1 won)
        """
        i = self._umaban_to_idx.get(uma1)
        j = self._umaban_to_idx.get(uma2)
        if i is None or j is None or i == j:
            return 0.0

        p_i_first = self._conditional_win_prob(i, set())
        p_j_second = self._conditional_win_prob(j, {i})
        return p_i_first * p_j_second

    def p_quinella(self, uma1: int, uma2: int) -> float:
        """馬連: P({uma1, uma2} が 1-2着、順不同)"""
        return self.p_exacta(uma1, uma2) + self.p_exacta(uma2, uma1)

    def p_wide(self, uma1: int, uma2: int) -> float:
        """ワイド: P({uma1, uma2} がともに3着以内)

        P(both in top3) = P(uma1∈top3, uma2∈top3)
        = Σ over all (a,b,c) in top3 where {uma1,uma2} ⊂ {a,b,c}
        """
        i = self._umaban_to_idx.get(uma1)
        j = self._umaban_to_idx.get(uma2)
        if i is None or j is None or i == j:
            return 0.0

        prob = 0.0
        # {i, j} が top3 に含まれる全パターンを列挙
        # 3番目の馬 k は i,j 以外の全馬
        for k in range(self.n_horses):
            if k == i or k == j:
                continue
            # (i,j,k) の全順列
            for perm in permutations([i, j, k]):
                p_first = self._conditional_win_prob(perm[0], set())
                p_second = self._conditional_win_prob(perm[1], {perm[0]})
                p_third = self._conditional_win_prob(perm[2], {perm[0], perm[1]})
                prob += p_first * p_second * p_third

        return prob

    def p_trifecta(self, uma1: int, uma2: int, uma3: int) -> float:
        """三連単: P(uma1=1着, uma2=2着, uma3=3着)

        Harville-Stern:
            P(i,j,k) = P_s(i wins) × P_s(j wins|i) × P_s(k wins|i,j)
        """
        i = self._umaban_to_idx.get(uma1)
        j = self._umaban_to_idx.get(uma2)
        k = self._umaban_to_idx.get(uma3)
        if i is None or j is None or k is None:
            return 0.0
        if len({i, j, k}) < 3:
            return 0.0

        p_i = self._conditional_win_prob(i, set())
        p_j = self._conditional_win_prob(j, {i})
        p_k = self._conditional_win_prob(k, {i, j})
        return p_i * p_j * p_k

    def p_trio(self, uma1: int, uma2: int, uma3: int) -> float:
        """三連複: P({uma1, uma2, uma3} が 1-2-3着、順不同)

        = Σ (全 3! = 6 通りの順列) P_trifecta(perm)
        """
        uma_list = [uma1, uma2, uma3]
        total = 0.0
        for perm in permutations(uma_list):
            total += self.p_trifecta(*perm)
        return total

    # -----------------------------------------------------------------
    # EV (期待値) 計算
    # -----------------------------------------------------------------
    def all_exacta_ev(self, odds_map: dict = None) -> pd.DataFrame:
        """全馬単組み合わせの確率と EV を計算する。

        Parameters
        ----------
        odds_map : dict, optional
            {(uma1, uma2): odds} の辞書。None の場合は確率のみ返す。

        Returns
        -------
        pd.DataFrame
            columns: uma1, uma2, prob, odds, ev
        """
        results = []
        for i, j in permutations(self.umaban, 2):
            prob = self.p_exacta(int(i), int(j))
            row = {"uma1": int(i), "uma2": int(j), "prob": prob}
            if odds_map and (int(i), int(j)) in odds_map:
                odds = odds_map[(int(i), int(j))]
                row["odds"] = odds
                row["ev"] = prob * odds
            results.append(row)

        df = pd.DataFrame(results)
        return df.sort_values("prob", ascending=False).reset_index(drop=True)

    def all_trifecta_ev(
        self, odds_map: dict = None, top_k: int = 8
    ) -> pd.DataFrame:
        """三連単の確率と EV を計算する。

        計算量削減のため、P(Win) 上位 top_k 頭に限定する。

        Parameters
        ----------
        odds_map : dict, optional
            {(uma1, uma2, uma3): odds} の辞書
        top_k : int
            確率上位何頭に絞るか（デフォルト: 8）

        Returns
        -------
        pd.DataFrame
            columns: uma1, uma2, uma3, prob, odds, ev
        """
        # 上位 top_k 頭の馬番を取得
        top_indices = np.argsort(-self.p_win)[:top_k]
        top_umaban = self.umaban[top_indices]

        results = []
        for perm in permutations(top_umaban, 3):
            prob = self.p_trifecta(int(perm[0]), int(perm[1]), int(perm[2]))
            row = {
                "uma1": int(perm[0]),
                "uma2": int(perm[1]),
                "uma3": int(perm[2]),
                "prob": prob,
            }
            if odds_map:
                key = (int(perm[0]), int(perm[1]), int(perm[2]))
                if key in odds_map:
                    odds = odds_map[key]
                    row["odds"] = odds
                    row["ev"] = prob * odds
            results.append(row)

        df = pd.DataFrame(results)
        return df.sort_values("prob", ascending=False).reset_index(drop=True)

    def all_trio_ev(
        self, odds_map: dict = None, top_k: int = 8
    ) -> pd.DataFrame:
        """三連複の確率と EV を計算する。

        Parameters
        ----------
        odds_map : dict, optional
            {frozenset({uma1, uma2, uma3}): odds} または {(uma1,uma2,uma3): odds}
        top_k : int
            確率上位何頭に絞るか

        Returns
        -------
        pd.DataFrame
            columns: uma1, uma2, uma3, prob, odds, ev
        """
        top_indices = np.argsort(-self.p_win)[:top_k]
        top_umaban = self.umaban[top_indices]

        results = []
        for combo in combinations(top_umaban, 3):
            prob = self.p_trio(int(combo[0]), int(combo[1]), int(combo[2]))
            row = {
                "uma1": int(combo[0]),
                "uma2": int(combo[1]),
                "uma3": int(combo[2]),
                "prob": prob,
            }
            if odds_map:
                # frozenset キーまたは sorted tuple キーで検索
                key_set = frozenset(int(c) for c in combo)
                key_tuple = tuple(sorted(int(c) for c in combo))
                odds = odds_map.get(key_set) or odds_map.get(key_tuple)
                if odds:
                    row["odds"] = odds
                    row["ev"] = prob * odds
            results.append(row)

        df = pd.DataFrame(results)
        return df.sort_values("prob", ascending=False).reset_index(drop=True)

    def all_wide_ev(
        self, odds_map: dict = None, top_k: int = 10
    ) -> pd.DataFrame:
        """ワイドの確率と EV を計算する。

        Parameters
        ----------
        odds_map : dict, optional
            {frozenset({uma1, uma2}): odds}
        top_k : int
            確率上位何頭に絞るか
        """
        top_indices = np.argsort(-self.p_win)[:top_k]
        top_umaban = self.umaban[top_indices]

        results = []
        for combo in combinations(top_umaban, 2):
            prob = self.p_wide(int(combo[0]), int(combo[1]))
            row = {
                "uma1": int(combo[0]),
                "uma2": int(combo[1]),
                "prob": prob,
            }
            if odds_map:
                key_set = frozenset(int(c) for c in combo)
                key_tuple = tuple(sorted(int(c) for c in combo))
                odds = odds_map.get(key_set) or odds_map.get(key_tuple)
                if odds:
                    row["odds"] = odds
                    row["ev"] = prob * odds
            results.append(row)

        df = pd.DataFrame(results)
        return df.sort_values("prob", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------
    # ユーティリティ
    # -----------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """各馬の P(Win), P(Place=Top3) を一覧表示する。"""
        rows = []
        for u in self.umaban:
            rows.append({
                "馬番": int(u),
                "P(Win)": self.p_win_single(int(u)),
                "P(Place)": self.p_place(int(u)),
            })
        return pd.DataFrame(rows).set_index("馬番")

    @staticmethod
    def load_stern_r() -> float:
        """最適化済み Stern パラメータを読み込む。"""
        path = local_paths.MODEL_DIR / "optimal_stern_r.pickle"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return 0.82  # デフォルト値

    def __repr__(self) -> str:
        return (
            f"PositionProbabilityEstimator("
            f"n_horses={self.n_horses}, stern_r={self.stern_r:.3f})"
        )


# =====================================================================
# ユーティリティ関数
# =====================================================================
def compute_race_p_win_from_ranker(
    ranker_model, X_race: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """LGBMRanker のスコアから Softmax で P(Win) を算出する。

    Parameters
    ----------
    ranker_model : LGBMRanker
        学習済みランカーモデル
    X_race : pd.DataFrame
        1レース分の特徴量データ

    Returns
    -------
    p_win : np.ndarray
        各馬の勝率
    umaban : np.ndarray
        馬番配列
    """
    # 特徴量の整列
    rfeat = ranker_model.feature_name_
    if callable(rfeat):
        rfeat = rfeat()

    X_r = X_race.copy()
    for f in rfeat:
        if f not in X_r.columns:
            X_r[f] = np.nan
    X_r = X_r[rfeat]

    for col in X_r.columns:
        if X_r[col].dtype == object:
            X_r[col] = pd.to_numeric(X_r[col], errors="coerce")

    scores = ranker_model.predict(X_r)

    # Softmax
    e_x = np.exp(scores - np.max(scores))
    p_win = e_x / e_x.sum()

    # 馬番
    if "馬番" in X_race.columns:
        umaban = X_race["馬番"].values.astype(int)
    else:
        umaban = np.arange(1, len(X_race) + 1)

    return p_win, umaban


def compute_race_p_win_from_classifier(
    model, X_race: pd.DataFrame, harville_coeff: float = 1.414
) -> tuple[np.ndarray, np.ndarray]:
    """分類器の P(Top3) から Harville 近似で P(Win) を算出する。

    Parameters
    ----------
    model : CalibratedClassifierCV or LGBMClassifier
        学習済み分類器
    X_race : pd.DataFrame
        1レース分の特徴量データ
    harville_coeff : float
        Power Law 変換指数

    Returns
    -------
    p_win : np.ndarray
        各馬の勝率
    umaban : np.ndarray
        馬番配列
    """
    X_feat = X_race.drop(columns=["単勝"], errors="ignore")

    p_top3 = model.predict_proba(X_feat)[:, 1]

    # Power Law: P(1着) ∝ P(Top3) ^ harville_coeff
    p_win_raw = p_top3 ** harville_coeff
    total = p_win_raw.sum()
    if total > 0:
        p_win = p_win_raw / total
    else:
        p_win = np.ones(len(p_top3)) / len(p_top3)

    if "馬番" in X_race.columns:
        umaban = X_race["馬番"].values.astype(int)
    else:
        umaban = np.arange(1, len(X_race) + 1)

    return p_win, umaban
