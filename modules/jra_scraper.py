"""
modules/jra_scraper.py – JRA 公式サイトから Playwright で単勝オッズを取得
===========================================================================
参考: 競馬AI開発 #24 (https://keiba-ai.com/vol24-K6D6G4CTwM4)

JRA 公式サイトは URL が変わらないまま JS でページ遷移するため、
Playwright でブラウザ操作 → page.content() → pd.read_html() で取得する。
"""

import asyncio
import re
from io import StringIO

import pandas as pd
from playwright.async_api import async_playwright

# 場所コード → 場所名 の逆引き辞書
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


async def fetch_jra_odds_async(race_id: str, headless: bool = True) -> dict[int, float]:
    """
    JRA 公式サイトから Playwright + pd.read_html で単勝オッズを取得する。

    流れ:
        1. JRA競馬メニュー → オッズ → 場所選択 → レース選択
        2. html = await page.content()
        3. pd.read_html(html) でテーブルを丸ごと取得
        4. 馬番・単勝オッズを抽出

    Returns
    -------
    dict[int, float]
        {馬番: 単勝オッズ}。取得失敗時は空辞書。
    """
    info = parse_race_id(race_id)
    if not info["place_name"]:
        print(f"[JRA] 不明な場所コード: {info['place_code']}")
        return {}

    odds_map: dict[int, float] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # ---- 1. JRA 競馬メニュー ----
            await page.goto("https://www.jra.go.jp/keiba/", timeout=15000)

            # ---- 2. オッズ → クリック ----
            await page.get_by_role("link", name="オッズ", exact=True).click()
            await page.wait_for_load_state("networkidle")

            # ---- 3. 場所選択: "N回PLACE M日" ----
            target_text = f"{info['kai']}回{info['place_name']}{info['day']}日"
            place_link = page.get_by_role("link", name=re.compile(re.escape(target_text)))
            if await place_link.count() == 0:
                place_link = page.get_by_role("link", name=re.compile(info["place_name"]))
            if await place_link.count() == 0:
                print(f"[JRA] 開催 '{target_text}' が見つかりません")
                return {}
            await place_link.first.click()
            await page.wait_for_load_state("networkidle")

            # ---- 4. レース選択 ----
            #   img alt="Nレース" の親 <a> をクリック
            #   または img src に btn_race_numN.png を含むリンクをクリック
            race_num = info["race_num"]
            race_alt = f"{race_num}レース"
            found = False

            # 方法A: img[alt="Nレース"]
            img_loc = page.locator(f'img[alt="{race_alt}"]')
            if await img_loc.count() > 0:
                parent = img_loc.first.locator("xpath=ancestor::a")
                if await parent.count() > 0:
                    await parent.first.click()
                else:
                    await img_loc.first.click()
                await page.wait_for_load_state("networkidle")
                found = True

            # 方法B: img src にレース番号画像を含むリンク
            if not found:
                src_loc = page.locator(f'a:has(img[src*="btn_race_num{race_num}.png"])')
                if await src_loc.count() > 0:
                    await src_loc.first.click()
                    await page.wait_for_load_state("networkidle")
                    found = True

            # 方法C: get_by_role
            if not found:
                link = page.get_by_role("link", name=race_alt)
                if await link.count() > 0:
                    await link.first.click()
                    await page.wait_for_load_state("networkidle")
                    found = True

            if not found:
                print(f"[JRA] {race_alt} のリンクが見つかりません")
                return {}

            # ---- 5. 単勝・複勝タブが別の場合はクリック ----
            try:
                tansho = page.get_by_role("link", name=re.compile("単勝"))
                if await tansho.count() > 0:
                    await tansho.first.click()
                    await page.wait_for_load_state("networkidle")
            except Exception:
                pass

            # ---- 6. ページ全体のHTMLを取得 → pd.read_html でテーブル解析 ----
            await page.wait_for_timeout(1500)  # テーブル描画待ち
            html = await page.content()

            dfs = pd.read_html(StringIO(html), flavor="lxml")

            # 単勝オッズのテーブルを探す:
            # 「馬番」と「単勝」(or「オッズ」) カラムを含むテーブル
            for df in dfs:
                cols = [str(c) for c in df.columns]
                cols_flat = " ".join(cols)

                # 馬番列と単勝列を特定
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
                        break

            # カラム名で見つからない場合のフォールバック:
            # 数値列のパターンから推測（馬番=1~18の整数列, オッズ=小数列）
            if not odds_map:
                for df in dfs:
                    if len(df.columns) >= 4 and len(df) >= 5:
                        # 2列目以降で馬番っぽい列（1始まり連番）とオッズっぽい列を探す
                        for i, col in enumerate(df.columns):
                            try:
                                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                                if len(vals) >= 5:
                                    # 1始まりの連続整数 → 馬番候補
                                    if vals.iloc[0] == 1 and (vals.diff().iloc[1:] == 1).all():
                                        umaban_col_idx = i
                                        # 次の次あたりがオッズ列（小数）
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
                                                    break
                                if odds_map:
                                    break
                            except Exception:
                                continue
                    if odds_map:
                        break

        except Exception as e:
            print(f"[JRA] スクレイピングエラー: {e}")
        finally:
            await context.close()
            await browser.close()

    return odds_map


def fetch_jra_odds(race_id: str, headless: bool = True) -> dict[int, float]:
    """同期版ラッパー。"""
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(fetch_jra_odds_async(race_id, headless=headless))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    race_id = sys.argv[1] if len(sys.argv) > 1 else "202606020103"
    print(f"JRA オッズ取得テスト: race_id={race_id}")
    info = parse_race_id(race_id)
    print(f"  場所: {info['place_name']}, {info['kai']}回{info['day']}日, {info['race_num']}R")

    odds = fetch_jra_odds(race_id, headless=False)
    if odds:
        print(f"\n単勝オッズ ({len(odds)}頭):")
        for u in sorted(odds.keys()):
            print(f"  {u:>2}番: {odds[u]:.1f}")
    else:
        print("オッズを取得できませんでした。")
