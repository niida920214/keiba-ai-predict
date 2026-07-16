"""
modules/jra_scraper_scrapling.py – Scrapling を使用した JRA 公式オッズ取得
========================================================================
Scrapling の StealthyFetcher + page_action を活用し、
JRA 公式サイトから単勝オッズをステルス取得する。

API インターフェースは既存 jra_scraper.py と完全互換:
  fetch_jra_odds_scrapling(race_id, headless=True) -> dict[int, float]
"""

import re
from io import StringIO

import pandas as pd

# 場所コード → 場所名
PLACE_CODE_TO_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def parse_race_id(race_id: str) -> dict:
    """
    netkeiba の race_id (12桁) を分解する。
    例: 202606020103 → 年=2026, 場所=中山, 開催回=2, 日=1, レース番号=3
    """
    return {
        "year": race_id[0:4],
        "place_code": race_id[4:6],
        "place_name": PLACE_CODE_TO_NAME.get(race_id[4:6], ""),
        "kai": int(race_id[6:8]),
        "day": int(race_id[8:10]),
        "race_num": int(race_id[10:12]),
    }


def _extract_odds_from_html(html: str) -> dict[int, float]:
    """
    HTML文字列からpd.read_htmlでオッズテーブルを解析し、
    {馬番: 単勝オッズ} の辞書を返す。
    """
    odds_map: dict[int, float] = {}

    try:
        dfs = pd.read_html(StringIO(html), flavor="lxml")
    except Exception:
        return odds_map

    # ---- 方法1: カラム名ベースで「馬番」「単勝」列を探す ----
    for df in dfs:
        cols = [str(c) for c in df.columns]
        umaban_col = None
        tansho_col = None

        for c in cols:
            if "馬番" in c:
                umaban_col = c
            if "単勝" in c and "複勝" not in c:
                tansho_col = c

        if umaban_col and tansho_col:
            for _, row in df.iterrows():
                try:
                    umaban = int(row[umaban_col])
                    odds_val = str(row[tansho_col]).strip()
                    odds = float(odds_val)
                    if odds > 0:
                        odds_map[umaban] = odds
                except (ValueError, TypeError):
                    continue
            if odds_map:
                return odds_map

    # ---- 方法2: パターンベースのフォールバック ----
    for df in dfs:
        if len(df.columns) >= 4 and len(df) >= 5:
            for i, col in enumerate(df.columns):
                try:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(vals) >= 5:
                        if vals.iloc[0] == 1 and (vals.diff().iloc[1:] == 1).all():
                            umaban_col_idx = i
                            for j in range(i + 1, len(df.columns)):
                                odds_vals = pd.to_numeric(df.iloc[:, j], errors="coerce").dropna()
                                if len(odds_vals) >= 5 and (odds_vals > 1.0).any():
                                    for k in range(len(df)):
                                        try:
                                            u = int(df.iloc[k, umaban_col_idx])
                                            o = float(df.iloc[k, j])
                                            if u > 0 and o > 0:
                                                odds_map[u] = o
                                        except (ValueError, TypeError):
                                            continue
                                    if odds_map:
                                        return odds_map
                except Exception:
                    continue

    return odds_map


def fetch_jra_odds_scrapling(race_id: str, headless: bool = True) -> dict[int, float]:
    """
    Scrapling (StealthyFetcher) を使用して JRA 公式サイトから単勝オッズを取得する。

    StealthyFetcher の page_action でブラウザ操作を行い、
    オッズページに遷移してからテーブルを解析する。

    Parameters
    ----------
    race_id : str
        netkeiba のレース ID（12桁）
    headless : bool
        ヘッドレスモードで実行するか（デフォルト: True）

    Returns
    -------
    dict[int, float]
        {馬番: 単勝オッズ}。取得失敗時は空辞書。
    """
    from scrapling.fetchers import StealthyFetcher

    info = parse_race_id(race_id)
    if not info["place_name"]:
        print(f"[JRA-Scrapling] 不明な場所コード: {info['place_code']}")
        return {}

    # page_action 内でナビゲーション完了後のHTMLを保存するためのコンテナ
    captured_html = {"html": ""}

    def navigate_to_odds(page):
        """
        Playwright Page オブジェクトを使って JRA サイト内をナビゲーションし、
        目的のオッズページまで遷移する。
        """
        import time

        # ---- 1. オッズリンクをクリック ----
        try:
            odds_link = page.get_by_role("link", name="オッズ", exact=True)
            if odds_link.count() > 0:
                odds_link.click()
                page.wait_for_load_state("networkidle", timeout=15000)
            else:
                print("[JRA-Scrapling] 「オッズ」リンクが見つかりません")
                return
        except Exception as e:
            print(f"[JRA-Scrapling] オッズリンクのクリックに失敗: {e}")
            return

        # ---- 2. 場所選択: "N回PLACE M日" ----
        target_text = f"{info['kai']}回{info['place_name']}{info['day']}日"
        try:
            place_link = page.get_by_role("link", name=re.compile(re.escape(target_text)))
            if place_link.count() == 0:
                place_link = page.get_by_role("link", name=re.compile(info["place_name"]))
            if place_link.count() == 0:
                print(f"[JRA-Scrapling] 開催 '{target_text}' が見つかりません")
                return
            place_link.first.click()
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception as e:
            print(f"[JRA-Scrapling] 場所選択に失敗: {e}")
            return

        # ---- 3. レース選択 ----
        race_num = info["race_num"]
        race_alt = f"{race_num}レース"
        found = False

        # 方法A: img[alt="Nレース"]
        try:
            img_loc = page.locator(f'img[alt="{race_alt}"]')
            if img_loc.count() > 0:
                parent = img_loc.first.locator("xpath=ancestor::a")
                if parent.count() > 0:
                    parent.first.click()
                else:
                    img_loc.first.click()
                page.wait_for_load_state("networkidle", timeout=15000)
                found = True
        except Exception:
            pass

        # 方法B: img src にレース番号画像を含むリンク
        if not found:
            try:
                src_loc = page.locator(f'a:has(img[src*="btn_race_num{race_num}.png"])')
                if src_loc.count() > 0:
                    src_loc.first.click()
                    page.wait_for_load_state("networkidle", timeout=15000)
                    found = True
            except Exception:
                pass

        # 方法C: get_by_role
        if not found:
            try:
                link = page.get_by_role("link", name=race_alt)
                if link.count() > 0:
                    link.first.click()
                    page.wait_for_load_state("networkidle", timeout=15000)
                    found = True
            except Exception:
                pass

        if not found:
            print(f"[JRA-Scrapling] {race_alt} のリンクが見つかりません")
            return

        # ---- 4. 単勝・複勝タブ ----
        try:
            tansho = page.get_by_role("link", name=re.compile("単勝"))
            if tansho.count() > 0:
                tansho.first.click()
                page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # ---- 5. テーブル描画待ち → HTML取得 ----
        try:
            page.wait_for_selector("table", timeout=10000)
        except Exception:
            pass
        time.sleep(3.0)
        captured_html["html"] = page.content()

    # ---- Scrapling でフェッチ ----
    try:
        response = StealthyFetcher.fetch(
            url="https://www.jra.go.jp/keiba/",
            headless=headless,
            network_idle=True,
            timeout=30000,
            page_action=navigate_to_odds,
            google_search=False,  # JRA サイトなので Google 経由の偽装は不要
        )

        odds_map = {}

        # pd.read_html でのパース (ナビゲーションで保存したHTMLがあれば優先)
        if captured_html["html"]:
            odds_map = _extract_odds_from_html(captured_html["html"])

        # さらにフォールバック: response の html_content
        if not odds_map:
            odds_map = _extract_odds_from_html(response.html_content)

        return odds_map

    except Exception as e:
        print(f"[JRA-Scrapling] スクレイピングエラー: {e}")
        return {}


# ---------------------------------------------------------------------------
# 直接実行時のテスト
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    race_id = sys.argv[1] if len(sys.argv) > 1 else "202606020103"
    print(f"JRA オッズ取得テスト (Scrapling版): race_id={race_id}")
    info = parse_race_id(race_id)
    print(f"  場所: {info['place_name']}, {info['kai']}回{info['day']}日, {info['race_num']}R")

    odds = fetch_jra_odds_scrapling(race_id, headless=False)
    if odds:
        print(f"\n単勝オッズ ({len(odds)}頭):")
        for u in sorted(odds.keys()):
            print(f"  {u:>2}番: {odds[u]:.1f}")
    else:
        print("オッズを取得できませんでした。")
