"""
scraper.py – netkeiba スクレイピングモジュール
================================================
Results, HorseResults, Peds, Return の 4 クラスと
差分更新ユーティリティ update_data を提供する。
"""

import pickle
import random
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from modules.constants import local_paths, url_paths

# ---------------------------------------------------------------------------
# User-Agent ローテーション用リスト
# ---------------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
]


# ---------------------------------------------------------------------------
# 適応型レートリミッター（旧グローバル変数をクラスに封じ込め）
# ---------------------------------------------------------------------------
class AdaptiveRateLimiter:
    """
    リクエスト間隔を動的に管理するクラス。

    - 成功が続くと間隔を短縮
    - 400/429/503 発生時に間隔を延長 + クールダウン
    - ランダムジッターで人間らしいアクセスパターンを再現
    - テスト可能・リセット可能な設計
    """

    def __init__(
        self,
        base_interval: float = 4.0,
        step: float = 1.0,
        max_interval: float = 8.0,
        shrink_after: int = 200,
        jitter: float = 2.0,
    ):
        self.base_interval = base_interval
        self.step = step
        self.max_interval = max_interval
        self.shrink_after = shrink_after
        self.jitter = jitter
        self.reset()

    def reset(self):
        """状態を初期化する。"""
        self.interval = self.base_interval
        self.success_streak = 0
        self.consecutive_fails = 0

    def on_success(self):
        """リクエスト成功時のコールバック。"""
        self.success_streak += 1
        if self.success_streak >= self.shrink_after and self.interval > self.base_interval:
            self.interval = max(self.base_interval, self.interval - self.step)
            self.success_streak = 0
            print(f"\n  ✓ 安定 — 間隔を {self.interval:.1f}s に短縮")
        self.consecutive_fails = 0

    def on_rate_limit(self, status_code: int):
        """レートリミット発生時のコールバック。"""
        old = self.interval
        self.interval = min(self.max_interval, self.interval + self.step)
        self.success_streak = 0
        self.consecutive_fails += 1
        # 連続失敗に応じて長めにクールダウン（最低120秒、最大600秒）
        wait = min(120 * self.consecutive_fails, 600)
        print(
            f"\n  ⚠ HTTP {status_code} — "
            f"間隔 {old:.1f}s→{self.interval:.1f}s に延長、"
            f"{wait}秒クールダウン後にリトライ (連続{self.consecutive_fails}回目)"
        )
        time.sleep(wait)

    def on_error(self, error: Exception):
        """接続エラー時のコールバック。"""
        self.success_streak = 0
        print(f"\n  ⚠ 接続エラー: {error} — 60秒待機してリトライ")
        time.sleep(60)

    def wait(self):
        """現在のインターバル + ランダムジッター分だけ待機する。"""
        actual = self.interval + random.uniform(0, self.jitter)
        time.sleep(actual)


# シングルトンインスタンス（後方互換性のため）
rate_limiter = AdaptiveRateLimiter()

# Ajax API 用の軽量レートリミッター（JSON のみでサーバー負荷が低い）
ajax_rate_limiter = AdaptiveRateLimiter(
    base_interval=1.0,
    step=1.0,
    max_interval=6.0,
    shrink_after=100,
    jitter=0.5,
)

# 後方互換性: 他モジュールが REQUEST_INTERVAL を参照している
REQUEST_INTERVAL = rate_limiter.base_interval


def safe_get(url: str, limiter: AdaptiveRateLimiter | None = None) -> requests.Response:
    """
    HTTP GET を実行し、必ず成功するまでリトライする（スキップしない）。
    400/429 発生時はインターバルを段階的に延長してリトライ。
    """
    _limiter = limiter or rate_limiter
    while True:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        try:
            res = requests.get(url, headers=headers, timeout=30)
            if res.status_code == 200:
                _limiter.on_success()
                return res
            elif res.status_code in (400, 429, 503):
                _limiter.on_rate_limit(res.status_code)
            else:
                print(
                    f"\n  ⚠ HTTP {res.status_code} (予期せぬステータス) — 30秒待機してリトライ"
                )
                time.sleep(30)
        except KeyboardInterrupt:
            raise
        except requests.exceptions.RequestException as e:
            _limiter.on_error(e)


# ===================================================================
# レース結果
# ===================================================================
class Results:
    """netkeiba のレース結果ページをスクレイピングする。"""

    @staticmethod
    def download_html(
        race_id_list: list[str],
        save_dir: str = str(local_paths.HTML_RACE_DIR),
        skip: bool = True,
    ) -> list[str]:
        """
        race ページの HTML をスクレイピングして保存する。
        """
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        html_path_list = []

        for race_id in tqdm(race_id_list, desc="Download HTML (Race)"):
            file_path = save_dir_path / f"{race_id}.bin"
            html_path_list.append(str(file_path))

            if skip and file_path.exists():
                continue

            try:
                url = url_paths.RACE_URL + race_id
                res = safe_get(url)
                # ファイル書き込み (バイナリ)
                with open(file_path, "wb") as f:
                    f.write(res.content)
                rate_limiter.wait()
            except Exception as e:
                print(f"[Results] Download Error at {race_id}: {e}")

        return html_path_list

    @staticmethod
    def scrape_from_html(html_path_list: list[str]) -> pd.DataFrame:
        """
        保存された HTML ファイルからレース結果を解析する。
        """
        import numpy as np
        from modules.constants.master import RACE_CLASS_DICT

        race_results: dict[str, pd.DataFrame] = {}

        for html_path in tqdm(html_path_list, desc="Parse HTML (Results)"):
            try:
                with open(html_path, "rb") as f:
                    html_content = f.read()

                # 文字コード変換 (EUC-JP -> UTF-8 は read_html 側で処理させるため、ここではデコードして StringIO に渡す)
                try:
                    html_text = html_content.decode("euc-jp", errors="replace")
                except Exception:
                    continue

                df = pd.read_html(StringIO(html_text))[0]
                df = df.rename(columns=lambda x: x.replace(" ", ""))

                soup = BeautifulSoup(html_text, "html.parser")
                data_intro = soup.find("div", attrs={"class": "data_intro"})
                
                # --- レース名と副題の取得 (race_class 抽出用) ---
                title = data_intro.find("h1").text.strip() if data_intro.find("h1") else ""
                subtitle_p = data_intro.find("p", attrs={"class": "smalltxt"})
                subtitle = subtitle_p.text.strip() if subtitle_p else ""
                full_title = f"{title} {subtitle}"
                
                # 正規表現で race_class を抽出 (RACE_CLASS_DICT のキーを使用)
                regex_race_class = "|".join([re.escape(k) for k in RACE_CLASS_DICT.keys()])
                match = re.search(rf"({regex_race_class})", full_title)
                if match:
                    df["race_class"] = RACE_CLASS_DICT[match.group(1)]
                else:
                    df["race_class"] = np.nan
                
                texts = (
                    data_intro.find_all("p")[0].text + data_intro.find_all("p")[1].text
                )
                info = re.findall(r"\w+", texts)

                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text] * len(df)
                    if "障" in text:
                        df["race_type"] = ["障害"] * len(df)
                    if "m" in text:
                        df["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(df)
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if "年" in text:
                        df["date"] = [text] * len(df)

                # horse_id の抽出
                horse_id_list_local: list[str] = []
                horse_a_list = soup.find(
                    "table", attrs={"summary": "レース結果"}
                ).find_all("a", attrs={"href": re.compile("^/horse")})
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list_local.append(horse_id[0])

                # jockey_id の抽出
                jockey_id_list: list[str] = []
                jockey_a_list = soup.find(
                    "table", attrs={"summary": "レース結果"}
                ).find_all("a", attrs={"href": re.compile("^/jockey")})
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])

                df["horse_id"] = horse_id_list_local
                df["jockey_id"] = jockey_id_list

                # ファイル名から ID を抽出
                match = re.search(r"(\d{12})\.bin", str(html_path))
                if match:
                    race_id = match.group(1)
                else:
                    race_id = Path(html_path).stem

                df.index = [race_id] * len(df)
                race_results[race_id] = df

            except Exception as e:
                continue

        if not race_results:
            return pd.DataFrame()

        return pd.concat(race_results.values())

    @staticmethod
    def scrape(race_id_list: list[str]) -> pd.DataFrame:
        """
        互換性のためのラッパー。HTML ダウンロード -> 解析 を一括で行う。
        """
        html_files = Results.download_html(race_id_list)
        return Results.scrape_from_html(html_files)


# ===================================================================
# 馬の過去成績
# ===================================================================
class HorseResults:
    """馬ごとの過去成績をスクレイピングする（Ajax API ベース）。"""

    AJAX_URL = "https://db.netkeiba.com/horse/ajax_horse_results.html"
    BATCH_SAVE_INTERVAL = 500  # 何頭ごとに中間保存するか

    @staticmethod
    def scrape_via_ajax(
        horse_id_list: list[str],
        save_path: Path | None = None,
        skip_wait: bool = False,
    ) -> pd.DataFrame:
        """
        netkeiba の Ajax API を使って馬の過去成績を取得する。
        500頭ごとに中間保存し、途中停止しても再開可能。

        Parameters
        ----------
        horse_id_list : list[str]
            取得対象の horse_id リスト
        save_path : Path, optional
            中間保存先パス。指定された場合、既存データを読み込んで
            未取得分のみ差分取得する。
        """
        # 既存データの読み込み（差分取得用）
        existing_df: pd.DataFrame | None = None
        if save_path and save_path.exists():
            with open(save_path, "rb") as f:
                existing_df = pickle.load(f)
            existing_ids = set(existing_df.index.unique())
            print(f"  既存データ: {len(existing_ids)} 頭取得済み")
            # 未取得分のみフィルタ
            horse_id_list = [hid for hid in horse_id_list if hid not in existing_ids]
            print(f"  未取得分: {len(horse_id_list)} 頭")
            if not horse_id_list:
                return existing_df

        # 新規取得分のみ dict に格納（既存データは DataFrame のまま保持）
        new_results: dict[str, pd.DataFrame] = {}

        def _build_combined_df() -> pd.DataFrame:
            """既存データ + 新規取得分を結合する。"""
            parts = []
            if existing_df is not None:
                parts.append(existing_df)
            if new_results:
                parts.append(pd.concat(new_results.values()))
            return pd.concat(parts) if parts else pd.DataFrame()

        fetched_count = 0
        for horse_id in tqdm(horse_id_list, desc="Fetch HorseResults (Ajax)"):
            try:
                url = f"{HorseResults.AJAX_URL}?input=UTF-8&output=json&id={horse_id}"
                res = safe_get(url, limiter=ajax_rate_limiter)

                data = res.json()

                if data.get("status") != "OK":
                    continue

                html_content = data["data"]
                if not html_content or "データがありません" in html_content:
                    continue

                soup = BeautifulSoup(html_content, "html.parser")
                table = soup.find("table")
                if table is None:
                    continue

                # ヘッダー行から <th> を取得
                header_row = table.find("tr")
                if header_row is None:
                    continue
                ths = header_row.find_all("th")
                col_names = [th.get_text(strip=True).replace(" ", "").replace("\u3000", "") for th in ths]

                # データ行をパース
                rows = []
                for tr in table.find_all("tr")[1:]:
                    tds = tr.find_all("td")
                    if len(tds) != len(col_names):
                        continue
                    row = [td.get_text(strip=True) for td in tds]
                    rows.append(row)

                if not rows:
                    continue

                df = pd.DataFrame(rows, columns=col_names)
                df.index = [horse_id] * len(df)
                new_results[horse_id] = df
                fetched_count += 1

                # バッチ保存
                if save_path and fetched_count % HorseResults.BATCH_SAVE_INTERVAL == 0:
                    tmp_df = _build_combined_df()
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "wb") as f:
                        pickle.dump(tmp_df, f)
                    total_count = len(existing_df.index.unique()) + len(new_results) if existing_df is not None else len(new_results)
                    print(f"\n  💾 中間保存: {total_count} 頭 ({save_path})")

                if not skip_wait:
                    ajax_rate_limiter.wait()
            except KeyboardInterrupt:
                # Ctrl+C でも途中保存
                if save_path and (new_results or existing_df is not None):
                    tmp_df = _build_combined_df()
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "wb") as f:
                        pickle.dump(tmp_df, f)
                    total_count = len(existing_df.index.unique()) + len(new_results) if existing_df is not None else len(new_results)
                    print(f"\n  💾 中断保存: {total_count} 頭 ({save_path})")
                raise
            except Exception as e:
                print(f"[HorseResults] Ajax Error at {horse_id}: {e}")
                continue

        return _build_combined_df()

    @staticmethod
    def scrape(horse_id_list: list[str], skip_wait: bool = False) -> pd.DataFrame:
        """Ajax API でスクレイピングする。"""
        return HorseResults.scrape_via_ajax(horse_id_list, skip_wait=skip_wait)


# ===================================================================
# 血統データ
# ===================================================================
class Peds:
    """馬の血統（5 世代）をスクレイピングする。"""

    @staticmethod
    def download_html(
        horse_id_list: list[str],
        save_dir: str = str(local_paths.HTML_PED_DIR),
        skip: bool = True,
        skip_wait: bool = False,
    ) -> list[str]:
        """
        horse/ped ページの HTML をスクレイピングして保存する。
        """
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        html_path_list = []

        for horse_id in tqdm(horse_id_list, desc="Download HTML (Peds)"):
            file_path = save_dir_path / f"{horse_id}.bin"
            html_path_list.append(str(file_path))

            if skip and file_path.exists():
                continue

            try:
                url = url_paths.PED_URL + horse_id
                res = safe_get(url)
                with open(file_path, "wb") as f:
                    f.write(res.content)
                if not skip_wait:
                    rate_limiter.wait()
            except Exception as e:
                print(f"[Peds] Download Error at {horse_id}: {e}")

        return html_path_list

    @staticmethod
    def scrape_from_html(html_path_list: list[str]) -> pd.DataFrame:
        """
        保存された HTML ファイルから血統データを解析する。
        """
        peds_dict: dict[str, pd.Series] = {}

        for html_path in tqdm(html_path_list, desc="Parse HTML (Peds)"):
            try:
                with open(html_path, "rb") as f:
                    html_content = f.read()

                try:
                    html_text = html_content.decode("euc-jp", errors="replace")
                except Exception:
                    continue

                if "データがありません" in html_text:
                    continue

                dfs = pd.read_html(StringIO(html_text))
                if not dfs:
                    continue
                df = dfs[0]

                generations: dict[int, pd.Series] = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df = df.drop(columns=[i])
                    df = df.drop_duplicates()

                # パスからID抽出
                match = re.search(r"(\d{10})\.bin", str(html_path))
                if match:
                    horse_id = match.group(1)
                else:
                    horse_id = Path(html_path).stem

                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
                peds_dict[horse_id] = ped.reset_index(drop=True)

            except Exception as e:
                continue

        if not peds_dict:
            return pd.DataFrame()

        return pd.concat(peds_dict.values(), axis=1).T.add_prefix("peds_")

    @staticmethod
    def scrape(horse_id_list: list[str], skip_wait: bool = False) -> pd.DataFrame:
        html_files = Peds.download_html(horse_id_list, skip_wait=skip_wait)
        return Peds.scrape_from_html(html_files)


# ===================================================================
# 払い戻しデータ
# ===================================================================
class Return:
    """払い戻し（配当）テーブルをスクレイピングする。"""

    @staticmethod
    def download_html(
        race_id_list: list[str],
        save_dir: str = str(local_paths.HTML_RACE_DIR),
        skip: bool = True,
    ) -> list[str]:
        """
        Return クラスは Results クラスと同じレースページの HTML を使用するため、
        Results.download_html を利用する。
        """
        return Results.download_html(race_id_list, save_dir, skip)

    @staticmethod
    def scrape_from_html(html_path_list: list[str]) -> pd.DataFrame:
        """
        保存された HTML ファイルから払い戻しテーブルを解析する。
        """
        return_tables: dict[str, pd.DataFrame] = {}

        for html_path in tqdm(html_path_list, desc="Parse HTML (Return)"):
            try:
                with open(html_path, "rb") as f:
                    content = f.read()

                content = content.replace(b"<br />", b"br")

                try:
                    html_text = content.decode("euc-jp", errors="replace")
                except Exception:
                    continue

                if "データがありません" in html_text:
                    continue

                dfs = pd.read_html(StringIO(html_text))
                if len(dfs) < 3:
                    continue

                df = pd.concat([dfs[1], dfs[2]])

                # パスからID抽出
                match = re.search(r"(\d{12})\.bin", str(html_path))
                if match:
                    race_id = match.group(1)
                else:
                    race_id = Path(html_path).stem

                df.index = [race_id] * len(df)
                return_tables[race_id] = df

            except Exception as e:
                continue

        if not return_tables:
            return pd.DataFrame()

        return pd.concat(return_tables.values())

    @staticmethod
    def scrape(race_id_list: list[str]) -> pd.DataFrame:
        html_files = Return.download_html(race_id_list)
        return Return.scrape_from_html(html_files)


# ===================================================================
# ユーティリティ
# ===================================================================
def update_data(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """既存データに新規データをマージ（重複インデックスは新しい方を優先）。"""
    filtered_old = old[~old.index.isin(new.index)]
    return pd.concat([filtered_old, new])


def scrape_kaisai_date(from_: str, to_: str) -> list[str]:
    """
    netkeiba のカレンダーページから開催日一覧を取得する。

    Parameters
    ----------
    from_ : str
        開始年月（例: "2019-01"）
    to_ : str
        終了年月（例: "2019-12"）

    Returns
    -------
    list[str]
        開催日の 8 桁文字列リスト（例: ["20190105", "20190106", ...]）
    """
    kaisai_date_list: list[str] = []
    for date in tqdm(pd.date_range(from_, to_, freq="MS"), desc="KaisaiDate"):
        year = date.year
        month = date.month
        url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"

        html = safe_get(url)
        html.encoding = "EUC-JP"

        rate_limiter.wait()

        soup = BeautifulSoup(html.text, "html.parser")
        a_list = soup.find("table", class_="Calendar_Table").find_all("a")
        for a in a_list:
            kaisai_date = re.findall(r"kaisai_date=(\d{8})", a["href"])[0]
            kaisai_date_list.append(kaisai_date)

    return kaisai_date_list


def scrape_race_id_list(kaisai_date_list: list[str]) -> list[str]:
    """
    開催日一覧から各開催日のレース一覧ページにアクセスし、
    レース ID のリストを取得する。

    Parameters
    ----------
    kaisai_date_list : list[str]
        scrape_kaisai_date で得た 8 桁日付文字列のリスト

    Returns
    -------
    list[str]
        レース ID のリスト（例: ["202501010101", "202501010102", ...]）
    """
    race_id_list: list[str] = []
    for kaisai_date in tqdm(kaisai_date_list, desc="RaceID"):
        url = (
            f"https://race.netkeiba.com/top/race_list_sub.html"
            f"?kaisai_date={kaisai_date}"
        )
        res = safe_get(url)
        res.encoding = "EUC-JP"

        rate_limiter.wait()

        soup = BeautifulSoup(res.text, "html.parser")
        a_list = soup.find_all("a", href=re.compile(r"race_id=\d{12}"))
        for a in a_list:
            race_id = re.findall(r"race_id=(\d{12})", a["href"])[0]
            race_id_list.append(race_id)

    # 重複排除して順序維持
    seen = set()
    unique_list = []
    for rid in race_id_list:
        if rid not in seen:
            seen.add(rid)
            unique_list.append(rid)

    return unique_list
