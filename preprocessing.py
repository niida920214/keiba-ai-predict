"""
preprocessing.py – 競馬データ前処理パイプライン
================================================
スクレイピング済みの生データを LightGBM に入力可能な形式に加工する。

クラス構成:
    DataProcessor   … Results / ShutubaTable の抽象親クラス
    Results         … 訓練データ（レース結果）の加工
    ShutubaTable    … 予測データ（出馬表）の加工
    HorseResults    … 馬の過去成績の保持・集計
    Peds            … 血統データの保持・エンコーディング
"""

import random
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from modules.constants import master
from modules.constants.master import RACE_CLASS_DICT
# scraper モジュールからスクレイピング機能と共通定数をインポート
from scraper import REQUEST_INTERVAL, USER_AGENTS
from scraper import HorseResults as ScraperHorseResults
from scraper import Peds as ScraperPeds
from scraper import Results as ScraperResults
from scraper import update_data

# ---------------------------------------------------------------------------
# 定数 (modules.constants.master から参照)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unicode 安全対策
# ---------------------------------------------------------------------------
def normalize_col_name(name: str) -> str:
    """カラム名を NFKC 正規化し、空白を除去する。

    netkeiba/JRA のデータは EUC-JP → UTF-8 変換時に
    全角スペース (U+3000) や半角スペースが混入することがある。
    NFKC 正規化で全角英数字も半角に統一する。
    """
    s = str(name)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(" ", "").replace("\u3000", "").replace("\t", "")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame のカラム名を正規化する。"""
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def parse_time_to_seconds(time_str) -> float:
    """走破タイム文字列 (例: '1:34.5') を秒数に変換する。"""
    try:
        s = str(time_str).strip()
        if ":" in s:
            parts = s.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        else:
            return float(s)
    except (ValueError, TypeError, IndexError):
        return np.nan


# =====================================================================
# DataProcessor（抽象親クラス）
# =====================================================================
class DataProcessor:
    """
    Results と ShutubaTable の親クラス。
    共通処理（馬の過去成績追加、血統データ追加、カテゴリ変数処理）を担う。

    Attributes
    ----------
    data    : pd.DataFrame  — raw データ
    data_p  : pd.DataFrame  — preprocessing() 後
    data_h  : pd.DataFrame  — merge_horse_results() 後
    data_pe : pd.DataFrame  — merge_peds() 後
    data_c  : pd.DataFrame  — process_categorical() 後
    no_peds : np.ndarray    — 血統データが存在しなかった horse_id 一覧
    """

    def __init__(self):
        self.data = pd.DataFrame()
        self.data_p = pd.DataFrame()
        self.data_h = pd.DataFrame()
        self.data_pe = pd.DataFrame()
        self.data_c = pd.DataFrame()
        self.no_peds = np.array([])

    # -----------------------------------------------------------------
    # 馬の過去成績を追加
    # -----------------------------------------------------------------
    def merge_horse_results(self, hr, n_samples_list=None):
        """
        Parameters
        ----------
        hr : HorseResults
            前処理済みの馬の過去成績オブジェクト
        n_samples_list : list, default [5, 9, 'all']
            過去何レース分の成績を追加するか
        """
        if n_samples_list is None:
            n_samples_list = [1, 5, 9, "all"]

        self.data_h = self.data_p.copy()
        
        # Ensure we don't accidentally drop columns if a merge fails silently or recreates df
        original_columns = self.data_h.columns.tolist()

        for n_samples in n_samples_list:
            try:
                self.data_h = hr.merge_all(self.data_h, n_samples=n_samples)
            except Exception as e:
                import traceback
                print(f"Failed at n_samples={n_samples}: {e}")
                traceback.print_exc()
                raise e
            if "date" not in self.data_h.columns:
                print(f"CRITICAL: 'date' column lost after n_samples={n_samples}! Columns: {self.data_h.columns.tolist()}")
                raise KeyError("date lost")

        # デフラグ（DataFrame の断片化による PerformanceWarning 回避）
        self.data_h = self.data_h.copy()

        # 馬の出走間隔を追加
        if "latest" in self.data_h.columns:
            self.data_h["interval"] = (self.data_h["date"] - self.data_h["latest"]).dt.days
        else:
            self.data_h["interval"] = np.nan
        # 不要カラムを安全に削除（存在しない場合は無視）
        self.data_h.drop(
            [c for c in ["開催", "latest"] if c in self.data_h.columns],
            axis=1, inplace=True
        )

        # 前走との距離差分（距離短縮フラグ）など
        if "course_len_1R" in self.data_h.columns:
            self.data_h["dist_diff"] = self.data_h["course_len"] - self.data_h["course_len_1R"]
            self.data_h["is_distance_shortened"] = (self.data_h["dist_diff"] < 0).astype(int)

        # 展開予測特徴量を追加
        self.add_pace_features()

    # -----------------------------------------------------------------
    # レース内相対特徴量の追加
    # -----------------------------------------------------------------
    def add_intra_race_features(self):
        """レース内での相対値（百分位順位・偏差）を特徴量に追加する。

        学術的根拠:
            ランキング学習 (Burges et al., 2010) では、絶対値よりも
            グループ内での相対値が予測精度を向上させることが知られている。
        """
        df = self.data_p.copy()

        # 体重のレース内百分位順位
        if "体重" in df.columns:
            df["weight_rank"] = df.groupby(level=0)["体重"].rank(pct=True)

        # 年齢のレース内偏差
        if "年齢" in df.columns:
            df["age_diff"] = df["年齢"] - df.groupby(level=0)["年齢"].transform("mean")

        # 斤量のレース内百分位順位
        if "斤量" in df.columns:
            df["impost_rank"] = df.groupby(level=0)["斤量"].rank(pct=True)

        self.data_p = df

    # -----------------------------------------------------------------
    # 展開予測特徴量の追加
    # -----------------------------------------------------------------
    def add_pace_features(self):
        """出走馬の脚質分布からペース圧を予測する特徴量を追加する。

        ペースプレッシャー (pace_pressure) が高いほどハイペースになりやすく、
        差し・追込み脚質が有利になる傾向がある。
        """
        df = self.data_h.copy()

        if "legtype_allR" in df.columns:
            # レース内の逃げ馬・先行馬の数を集計
            df["n_nige"] = df.groupby(level=0)["legtype_allR"].transform(
                lambda x: (x < 0.5).sum()
            )
            df["n_senkou"] = df.groupby(level=0)["legtype_allR"].transform(
                lambda x: ((x >= 0.5) & (x < 1.5)).sum()
            )
            # ペースプレッシャー指標
            df["pace_pressure"] = df["n_nige"] + df["n_senkou"] * 0.5
            # 脚質 × ペース交互作用
            df["legtype_pace_interaction"] = df["legtype_allR"] * df["pace_pressure"]
        else:
            # legtype_allR がまだ存在しない場合（merge前など）はスキップ
            pass

        self.data_h = df

    # -----------------------------------------------------------------
    # コース特徴量の付与
    # -----------------------------------------------------------------
    def add_course_features(self):
        """開催コードに基づいて各コースの設計特徴量を付与する。"""
        df = self.data_p.copy()
        
        # 01:札幌, 02:函館, 03:福島, 04:新潟, 05:東京, 06:中山, 07:中京, 08:京都, 09:阪神, 10:小倉
        straight_dict = {
            "01": 266, "02": 262, "03": 292, "04": 659, "05": 525, 
            "06": 310, "07": 412, "08": 328, "09": 473, "10": 293
        }
        df["straight_len"] = df["開催"].map(straight_dict).fillna(300).astype(int)
        
        # 急坂フラグ (中山, 中京, 阪神, 東京)
        steep_hill_venues = ["05", "06", "07", "09"]
        df["has_steep_hill"] = df["開催"].isin(steep_hill_venues).astype(int)
        
        # 小回りフラグ (札幌, 函館, 福島, 小倉, 中山など)
        small_circle_venues = ["01", "02", "03", "06", "10"]
        df["is_small_circle"] = df["開催"].isin(small_circle_venues).astype(int)
        
        self.data_p = df

    # -----------------------------------------------------------------
    # 血統データを追加
    # -----------------------------------------------------------------
    def merge_peds(self, peds):
        """
        Parameters
        ----------
        peds : pd.DataFrame
            Peds.peds_e（エンコード済み血統データ）
        """
        self.data_pe = self.data_h.merge(
            peds, left_on="horse_id", right_index=True, how="left"
        )
        self.no_peds = self.data_pe[self.data_pe["peds_0"].isnull()][
            "horse_id"
        ].unique()
        if len(self.no_peds) > 0:
            print('scrape peds at horse_id_list "no_peds"')

    # -----------------------------------------------------------------
    # カテゴリ変数の処理
    # -----------------------------------------------------------------
    def process_categorical(self, le_horse, le_jockey, results_m):
        """
        Parameters
        ----------
        le_horse  : LabelEncoder  — horse_id 用
        le_jockey : LabelEncoder  — jockey_id 用
        results_m : pd.DataFrame  — ダミー変数の列を合わせるための参照データ
        """
        df = self.data_pe.copy()

        # --- horse_id のラベルエンコーディング ---
        mask_horse = df["horse_id"].isin(le_horse.classes_)
        new_horse_id = df["horse_id"].mask(mask_horse).dropna().unique()
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        df["horse_id"] = le_horse.transform(df["horse_id"])

        # --- jockey_id のラベルエンコーディング ---
        mask_jockey = df["jockey_id"].isin(le_jockey.classes_)
        new_jockey_id = df["jockey_id"].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate([le_jockey.classes_, new_jockey_id])
        df["jockey_id"] = le_jockey.transform(df["jockey_id"])

        # category 型に変換
        df["horse_id"] = df["horse_id"].astype("category")
        df["jockey_id"] = df["jockey_id"].astype("category")

        # ダミー変数化（列を一定にするため Categorical を使用）
        weathers = results_m["weather"].unique()
        race_types = results_m["race_type"].unique()
        ground_states = results_m["ground_state"].unique()
        sexes = results_m["性"].unique()

        df["weather"] = pd.Categorical(df["weather"], weathers)
        df["race_type"] = pd.Categorical(df["race_type"], race_types)
        df["ground_state"] = pd.Categorical(df["ground_state"], ground_states)
        df["性"] = pd.Categorical(df["性"], sexes)
        df = pd.get_dummies(df, columns=["weather", "race_type", "ground_state", "性"])

        self.data_c = df


# =====================================================================
# Results（訓練データ加工）
# =====================================================================
class Results(DataProcessor):
    """レース結果データを加工するクラス（訓練用）。"""

    def __init__(self, results: pd.DataFrame):
        super().__init__()
        self.data = results

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(race_id_list):
        """scraper.Results.scrape に委譲。"""
        return ScraperResults.scrape(race_id_list)

    # -----------------------------------------------------------------
    # 前処理
    # -----------------------------------------------------------------
    def preprocessing(self):
        df = self.data.copy()

        # Unicode 正規化（EUC-JP→UTF-8 変換由来の不整合を解消）
        df = normalize_columns(df)

        # 着順に数字以外の文字列が含まれているものを除去
        df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
        df.dropna(subset=["着順"], inplace=True)
        df["着順"] = df["着順"].astype(int)
        df["rank"] = df["着順"].map(lambda x: 1 if x < 4 else 0)
        df["rank_win"] = (df["着順"] == 1).astype(int)  # 1着予測用

        # ★ ランキング学習用に着順を保持（LGBMRanker の label として使用）
        df["finishing_position"] = df["着順"].copy()

        # ★ スピード指数用に走破タイムを秒数に変換して保持
        if "タイム" in df.columns:
            df["time_sec"] = df["タイム"].map(parse_time_to_seconds)
        else:
            df["time_sec"] = np.nan

        # 性齢 → 性 + 年齢
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重 → 体重 + 体重変化
        df["体重"] = df["馬体重"].str.split("(", expand=True)[0]
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1]
        df["体重"] = pd.to_numeric(df["体重"], errors="coerce")
        df["体重変化"] = pd.to_numeric(df["体重変化"], errors="coerce")

        # 単勝を float に変換
        df["単勝"] = df["単勝"].astype(float)
        # 距離は 10 の位を切り捨て
        df["course_len"] = df["course_len"].astype(float) // 100

        # 不要な列を削除（タイム・着順は上で保持済み）
        drop_cols = [
            "タイム",
            "着差",
            "調教師",
            "性齢",
            "馬体重",
            "馬名",
            "騎手",
            "人気",
            "着順",
        ]
        df.drop(
            [c for c in drop_cols if c in df.columns],
            axis=1,
            inplace=True,
        )

        df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")

        # 開催場所
        df["開催"] = df.index.map(lambda x: str(x)[4:6])

        # 出走頭数
        df["n_horses"] = df.index.map(df.index.value_counts())

        self.data_p = df
        self.add_course_features()
        self.add_intra_race_features()

    # -----------------------------------------------------------------
    # カテゴリ変数の処理（オーバーライド）
    # -----------------------------------------------------------------
    def process_categorical(self):
        """
        Results 側で LabelEncoder を fit してから親クラスに委譲。

        注意（学術的制約）:
            LabelEncoder は data_pe 全体（train+test）で fit されている。
            LightGBM の category 型ではエンコーディング値自体に意味はなく、
            各ユニーク値が独立したカテゴリとして扱われるため、実害は限定的だが、
            厳密にはテストデータの ID 分布情報が漏洩している（Data Leakage）。
            Kaggle / 論文投稿レベルでは train のみで fit すべき。
        """
        self.le_horse = LabelEncoder().fit(self.data_pe["horse_id"])
        self.le_jockey = LabelEncoder().fit(self.data_pe["jockey_id"])
        super().process_categorical(self.le_horse, self.le_jockey, self.data_pe)


# =====================================================================
# ShutubaTable（予測データ加工）
# =====================================================================
class ShutubaTable(DataProcessor):
    """出馬表データを加工するクラス（予測用）。"""

    def __init__(self, shutuba_tables: pd.DataFrame):
        super().__init__()
        self.data = shutuba_tables

    @classmethod
    def scrape(cls, race_id_list, date):
        """
        出馬表をスクレイピングする。

        Parameters
        ----------
        race_id_list : list[str]
        date : str  — 開催日（例: "2019/01/05"）
        """
        dfs_list = []
        for race_id in tqdm(race_id_list, desc="ShutubaTable"):
            time.sleep(REQUEST_INTERVAL)
            url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + race_id
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            html = requests.get(url, headers=headers)

            # Try UTF-8 first, then EUC-JP, with error tolerance
            from io import BytesIO
            raw = html.content
            for enc in ["utf-8", "euc-jp", "cp932"]:
                try:
                    raw.decode(enc)
                    html.encoding = enc
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                html.encoding = "utf-8"  # fallback with errors handled by pandas

            df = pd.read_html(BytesIO(raw), encoding=html.encoding)[0]
            df = df.rename(columns=lambda x: str(x).replace(" ", ""))
            df = df.T.reset_index(level=0, drop=True).T

            # Initialize with defaults to avoid KeyErrors if parsing fails
            df["weather"] = "晴"
            df["ground_state"] = "良"
            df["race_type"] = "ダート"
            df["course_len"] = 1600


            soup = BeautifulSoup(html.text, "html.parser")
            texts = soup.find("div", attrs={"class": "RaceData01"}).text
            texts = re.findall(r"\w+", texts)

            # --- レース名の取得 (race_class 抽出用) ---
            race_name_elem = soup.find(class_="RaceName")
            race_name = race_name_elem.text.strip() if race_name_elem else ""
            
            # G1, G2, G3 などはアイコンクラスで表現される場合があるため補完
            grade_str = ""
            if race_name_elem:
                grade_icon = race_name_elem.find(class_=re.compile("Icon_GradeType"))
                if grade_icon:
                    cls_list = grade_icon.get("class", [])
                    grade_match = re.search(r"Icon_GradeType(\d)", str(cls_list))
                    if grade_match:
                        grade_num = grade_match.group(1)
                        if grade_num in ["1", "2", "3"]:
                            grade_str = f"G{grade_num}"
            
            # 予測フェーズ等で未勝利戦のタイトルが取れない場合のフォールバック（RaceData01内の情報も統合）
            full_title = f"{race_name} {grade_str} {' '.join(texts)}"
            
            # 正規表現で race_class を抽出 (RACE_CLASS_DICT のキーを使用)
            regex_race_class = "|".join([re.escape(k) for k in RACE_CLASS_DICT.keys()])
            match = re.search(rf"({regex_race_class})", full_title)
            if match:
                df["race_class"] = RACE_CLASS_DICT[match.group(1)]
            else:
                df["race_class"] = np.nan

            for text in texts:
                if "m" in text:
                    df["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(df)
                if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                    df["weather"] = [text] * len(df)
                if text in ["良", "稍重", "重"]:
                    df["ground_state"] = [text] * len(df)
                if "不" in text:
                    df["ground_state"] = ["不良"] * len(df)
                if "稍" in text:
                    df["ground_state"] = ["稍重"] * len(df)
                if "芝" in text:
                    df["race_type"] = ["芝"] * len(df)
                if "障" in text:
                    df["race_type"] = ["障害"] * len(df)
                if "ダ" in text:
                    df["race_type"] = ["ダート"] * len(df)
            df["date"] = [date] * len(df)

            # horse_id
            horse_id_list = []
            horse_td_list = soup.find_all("td", attrs={"class": "HorseInfo"})
            for td in horse_td_list:
                horse_id = re.findall(r"\d+", td.find("a")["href"])[0]
                horse_id_list.append(horse_id)

            # jockey_id
            jockey_id_list = []
            jockey_td_list = soup.find_all("td", attrs={"class": "Jockey"})
            for td in jockey_td_list:
                jockey_id = re.findall(r"\d+", td.find("a")["href"])[0]
                jockey_id_list.append(jockey_id)

            df["horse_id"] = horse_id_list
            df["jockey_id"] = jockey_id_list
            df.index = [race_id] * len(df)
            dfs_list.append(df)

        return cls(pd.concat(dfs_list) if dfs_list else pd.DataFrame())

    # -----------------------------------------------------------------
    # 前処理
    # -----------------------------------------------------------------
    def preprocessing(self):
        df = self.data.copy()

        # Unicode 正規化
        df = normalize_columns(df)

        # 性齢 → 性 + 年齢
        df["性"] = df["性齢"].map(lambda x: str(x)[0] if pd.notnull(x) else "")
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:] if pd.notnull(x) else "0").astype(int)

        # 馬体重 → 体重 + 体重変化
        # 馬体重(増減) カラム名を正規化後のバリエーションに対応
        weight_col = None
        for c in df.columns:
            if "馬体重" in c:
                weight_col = c
                break
        if weight_col is None:
            weight_col = "馬体重(増減)"

        splitted = df[weight_col].str.split("(", expand=True)
        
        # 体重: 数値変換できない場合は480(平均)で埋める
        df["体重"] = pd.to_numeric(splitted[0], errors='coerce').fillna(480).astype(int)
        
        # 体重変化: カラムがある場合のみ処理
        if len(splitted.columns) > 1:
            df["体重変化"] = splitted[1].str[:-1]
            df["体重変化"] = pd.to_numeric(df["体重変化"], errors="coerce").fillna(0).astype(int)
        else:
            df["体重変化"] = 0

        df["date"] = pd.to_datetime(df["date"])
        
        # 安全な型変換
        df["枠"] = pd.to_numeric(df["枠"], errors='coerce').fillna(0).astype(int)
        df["馬番"] = pd.to_numeric(df["馬番"], errors='coerce').fillna(0).astype(int)
        df["斤量"] = pd.to_numeric(df["斤量"], errors='coerce').fillna(57).astype(int) # Default weight
        
        df["開催"] = df.index.map(lambda x: str(x)[4:6] if pd.notnull(x) else "")

        # 出走頭数
        df["n_horses"] = df.index.map(df.index.value_counts())

        # 距離は 10 の位を切り捨て
        df["course_len"] = df["course_len"].astype(float) // 100

        # 使用する列を選択
        df = df[
            [
                "枠",
                "馬番",
                "斤量",
                "course_len",
                "weather",
                "race_type",
                "ground_state",
                "date",
                "horse_id",
                "jockey_id",
                "性",
                "年齢",
                "体重",
                "体重変化",
                "開催",
                "n_horses",
                "race_class",
            ]
        ]

        self.data_p = df.rename(columns={"枠": "枠番"})
        self.add_course_features()
        self.add_intra_race_features()


# =====================================================================
# HorseResults（馬の過去成績の保持・集計）
# =====================================================================
class HorseResults:
    """
    馬の過去成績データを保持し、集計・マージ機能を提供するクラス。
    """

    def __init__(self, horse_results: pd.DataFrame):
        # Unicode 正規化（Ajax API / EUC-JP 由来のカラム名のブレを統一）
        horse_results = horse_results.copy()
        horse_results = normalize_columns(horse_results)

        target_cols = ["日付", "着順", "賞金", "着差", "通過", "開催", "距離", "上り"]
        
        try:
            if horse_results.empty:
                self.horse_results = pd.DataFrame(columns=target_cols + ["頭数"])
            else:
                self.horse_results = horse_results[target_cols].copy()
                # 「頭数」があれば追加（脚質判定の精度向上のため）
                if "頭数" in horse_results.columns:
                    self.horse_results["頭数"] = horse_results["頭数"].values
        except KeyError:
            # カラム名の完全一致に失敗 → インデックスベースで抽出
            n_cols = len(horse_results.columns)
            if n_cols >= 33:
                # 33 columns (Ajax API 形式):
                # 日付(0), 着順(11), 賞金(28), 着差(19), 通過(21), 開催(1), 距離(14), 上り(23)
                self.horse_results = horse_results.iloc[:, [0, 11, 28, 19, 21, 1, 14, 23]]
            elif n_cols == 29:
                self.horse_results = horse_results.iloc[:, [0, 11, 28, 19, 21, 1, 14, 23]]
            elif n_cols == 28:
                self.horse_results = horse_results.iloc[:, [0, 11, 27, 18, 20, 1, 14, 22]]
            else:
                self.horse_results = pd.DataFrame(columns=target_cols)
            self.horse_results.columns = target_cols
            # インデックスベース抽出後に「頭数」を追加
            if n_cols >= 33 and 6 < n_cols:
                self.horse_results["頭数"] = horse_results.iloc[:, 6].values
        self.target_list = []
        self.average_dict = {}
        self.latest = pd.Series(name="latest", dtype="datetime64[ns]")
        self.preprocessing()

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(horse_id_list):
        """scraper.HorseResults.scrape に委譲。"""
        return ScraperHorseResults.scrape(horse_id_list)

    # -----------------------------------------------------------------
    # 前処理
    # -----------------------------------------------------------------
    def preprocessing(self):
        df = self.horse_results.copy()

        if df.empty:
            # Create an empty dataframe with correct types for further processing
            df = pd.DataFrame(columns=[
                "日付", "着順", "賞金", "着差", "first_corner", "final_corner",
                "final_to_rank", "first_to_rank", "first_to_final", "開催", "race_type", "course_len", "上り", "date"
            ])
            df.index.name = "horse_id"
            self.horse_results = df
            self.target_list = [
                "着順", "賞金", "着差", "first_corner", "final_corner",
                "first_to_rank", "first_to_final", "final_to_rank", "course_len", "上り"
            ]
            return

        # 着順: 数字以外を除去
        df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
        df.dropna(subset=["着順"], inplace=True)
        df["着順"] = df["着順"].astype(int)

        df["date"] = pd.to_datetime(df["日付"])
        df.drop(["日付"], axis=1, inplace=True)

        # 賞金の NaN/空文字列 → 0（BeautifulSoup直パースでは空セルが '' になる）
        df["賞金"] = pd.to_numeric(df["賞金"], errors="coerce").fillna(0)

        # 1 着の着差を 0 にする
        df["着差"] = pd.to_numeric(df["着差"], errors="coerce").fillna(0.0)
        df["着差"] = df["着差"].map(lambda x: 0 if x < 0 else x)

        # コーナー通過順位
        def corner(x, n):
            if not isinstance(x, str):
                return x
            matches = re.findall(r"\d+", x)
            if not matches:
                return 0
            elif n == 4:
                return int(matches[-1])
            elif n == 1:
                return int(matches[0])
            return 0

        df["first_corner"] = df["通過"].map(lambda x: corner(x, 1))
        df["final_corner"] = df["通過"].map(lambda x: corner(x, 4))

        df["final_to_rank"] = df["final_corner"] - df["着順"]
        df["first_to_rank"] = df["first_corner"] - df["着順"]
        df["first_to_final"] = df["first_corner"] - df["final_corner"]

        # 脚質判定 (JRA-VAN NEXT基準)
        def classify_legtype(row):
            passage = row["通過"]
            n_horses = pd.to_numeric(row["n_horses_past"], errors="coerce")
            if pd.isna(n_horses):
                n_horses = 18 # デフォルト
                
            if not isinstance(passage, str):
                return np.nan
            
            matches = re.findall(r"\d+", passage)
            if not matches:
                return np.nan
            
            corners = [int(x) for x in matches]
            
            if 1 in corners[:-1]:
                return 0.0  # 逃げ
            elif corners[-1] <= 4:
                return 1.0  # 先行
            elif corners[-1] <= n_horses * 2 / 3 and n_horses >= 8:
                return 2.0  # 差し
            else:
                return 3.0  # 追い込み
                
        # n_horses は今回レースの頭数であり、過去レースの頭数(=出走頭数)が必要なため着順/頭数などから取れない場合デフォルトにする
        # 正確な頭数が取れるように過去成績の「頭数」を活用（データにあれば）
        # なければとりあえず18で代用するか、rank等を使う
        if "頭数" in df.columns:
            df["n_horses_past"] = df["頭数"]
        else:
            # 取得していなければデフォルトで18
            df["n_horses_past"] = 18

        df["legtype"] = df.apply(classify_legtype, axis=1)

        # 開催場所
        df["開催"] = (
            df["開催"].str.extract(r"(\D+)")[0].map(master.PLACE_DICT).fillna("11")
        )
        # race_type
        df["race_type"] = df["距離"].str.extract(r"(\D+)")[0].map(master.RACE_TYPE_DICT)
        # 距離（10 の位を切り捨て）
        df["course_len"] = df["距離"].str.extract(r"(\d+)").astype(float) // 100
        df.drop(["距離"], axis=1, inplace=True)

        # 上り
        df["上り"] = pd.to_numeric(df["上り"], errors="coerce").fillna(0.0)

        df.index.name = "horse_id"

        self.horse_results = df
        self.target_list = [
            "着順",
            "賞金",
            "着差",
            "first_corner",
            "final_corner",
            "first_to_rank",
            "first_to_final",
            "final_to_rank",
            "course_len",
            "上り",
            "legtype",
        ]

    # -----------------------------------------------------------------
    # 集計
    # -----------------------------------------------------------------
    def average(self, horse_id_list, date, n_samples="all"):
        target_df = self.horse_results.query("index in @horse_id_list")

        if n_samples == "all":
            filtered_df = target_df[target_df["date"] < date]
        elif n_samples > 0:
            filtered_df = (
                target_df[target_df["date"] < date]
                .sort_values("date", ascending=False)
                .groupby(level=0)
                .head(n_samples)
            )
        else:
            raise ValueError("n_samples must be > 0")

        self.average_dict = {}
        
        if filtered_df.empty:
            # Handle empty data
            empty_non_category = pd.DataFrame(columns=[f"{c}_{n_samples}R" for c in self.target_list])
            empty_non_category.index.name = "horse_id"
            self.average_dict["non_category"] = empty_non_category
            for column in ["course_len", "race_type", "開催"]:
                empty_cat = pd.DataFrame(
                    columns=[f"{c}_{column}_{n_samples}R" for c in self.target_list],
                    index=pd.MultiIndex.from_arrays([[], []], names=["horse_id", column])
                )
                self.average_dict[column] = empty_cat
            # 空データでも latest を初期化（どの n_samples でも一貫して保持）
            self.latest = pd.Series(name="latest", index=pd.Index([], name="horse_id"), dtype='datetime64[ns]')
            return

        # 平均値集計
        self.average_dict["non_category"] = (
            filtered_df.groupby(level=0)[self.target_list]
            .mean()
            .add_suffix("_{}R".format(n_samples))
        )
        
        # legtypeなど特定の列について、標準偏差（ブレ）も計算して結合する (動画内での精度向上アプローチ)
        if "legtype" in self.target_list and n_samples in [3, 5]:
            std_df = (
                filtered_df.groupby(level=0)[["legtype"]]
                .std(ddof=0)
                .add_suffix(f"_std_{n_samples}R")
            )
            self.average_dict["non_category"] = self.average_dict["non_category"].join(std_df)

        for column in ["course_len", "race_type", "開催"]:
            self.average_dict[column] = (
                filtered_df.groupby(["horse_id", column])[self.target_list]
                .mean()
                .add_suffix("_{}_{}R".format(column, n_samples))
            )

        # 馬の出走間隔用に最新日付を保持（全 n_samples で計算 — max date は
        # n_samples に依存しないため、どの値でも同じ結果になる）
        self.latest = filtered_df.groupby("horse_id")["date"].max().rename("latest")

    def merge(self, results, date, n_samples="all"):
        df = results[results["date"] == date]
        horse_id_list = df["horse_id"]
        self.average(horse_id_list, date, n_samples)
        merged_df = df.merge(
            self.average_dict["non_category"],
            left_on="horse_id",
            right_index=True,
            how="left",
        )
        for column in ["course_len", "race_type", "開催"]:
            merged_df = merged_df.merge(
                self.average_dict[column],
                left_on=["horse_id", column],
                right_index=True,
                how="left",
            )

        # 出走間隔
        if n_samples == 5:
            merged_df = merged_df.merge(
                self.latest,
                left_on="horse_id",
                right_index=True,
                how="left",
            )
        return merged_df

    def merge_all(self, results, n_samples="all"):
        if "date" not in results.columns:
            raise KeyError(f"date not in results columns! Columns: {results.columns.tolist()}")
        date_list = results["date"].unique()
        merged_dfs = []
        for date in tqdm(date_list):
            merged_dfs.append(self.merge(results, date, n_samples))
        if not merged_dfs:
            return results.copy() 
        merged_df = pd.concat(merged_dfs)
        return merged_df


# =====================================================================
# Peds（血統データの保持・エンコーディング）
# =====================================================================
class Peds:
    """5 世代分の血統データを保持し、LabelEncoding を行うクラス。"""

    def __init__(self, peds: pd.DataFrame):
        self.peds = peds
        self.peds_e = pd.DataFrame()

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(horse_id_list):
        """scraper.Peds.scrape に委譲。"""
        return ScraperPeds.scrape(horse_id_list)

    def encode(self):
        """血統データを 0 始まりの整数にエンコードし category 型に変換。"""
        df = self.peds.copy()
        for column in df.columns:
            df[column] = LabelEncoder().fit_transform(df[column].fillna("Na"))
        self.peds_e = df.astype("category")
