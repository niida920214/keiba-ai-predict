"""
jra_playwright_async.py – 単勝オッズ統合取得モジュール
=====================================================
一次: JRA 公式サイト（Playwright）→ リアルタイムオッズ
二次: netkeiba JSON API（requests）→ フォールバック（1日5回更新）

predict.py から `fetch_win_odds_map(race_id)` を呼ぶだけでオッズを取得できる。
"""

import requests


def _fetch_netkeiba_odds(race_id: str) -> dict[int, float]:
    """
    netkeiba の JSON API から単勝オッズを取得する（フォールバック）。
    無料ユーザーは 1 日 5 回更新のため精度は低い。
    """
    url = "https://race.netkeiba.com/api/api_get_jra_odds.html"
    params = {"race_id": race_id, "type": "1"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        res = requests.get(url, params=params, headers=headers, timeout=10)
        if res.status_code != 200:
            return {}

        data = res.json()
        if data.get("status") != "result":
            return {}

        tansho = data.get("data", {}).get("odds", {}).get("1", {})
        result: dict[int, float] = {}
        for umaban_str, values in tansho.items():
            try:
                result[int(umaban_str)] = float(values[0])
            except (ValueError, IndexError):
                continue
        return result

    except Exception:
        return {}


def fetch_win_odds_map(race_id: str) -> dict[int, float]:
    """
    単勝オッズを取得する統合関数。

    1. JRA 公式（Playwright）を試みる
    2. 失敗時は netkeiba API にフォールバック

    Parameters
    ----------
    race_id : str
        netkeiba のレース ID（12桁）

    Returns
    -------
    dict[int, float]
        {馬番: 単勝オッズ}。取得失敗時は空辞書。
    """
    # ---- 一次: JRA 公式 (Scrapling → Playwright フォールバック) ----
    jra_odds: dict[int, float] = {}

    # まず Scrapling 版を試行
    try:
        from modules.jra_scraper_scrapling import fetch_jra_odds_scrapling
        jra_odds = fetch_jra_odds_scrapling(race_id, headless=True)
        if jra_odds:
            print(f"  [JRA公式/Scrapling] リアルタイムオッズを取得しました（{len(jra_odds)}頭）")
        else:
            print("  [JRA公式/Scrapling] オッズを取得できませんでした")
    except ImportError:
        print("  [JRA公式] Scrapling未インストール、Playwright版にフォールバック")
    except Exception as e:
        print(f"  [JRA公式/Scrapling] エラー: {e}")

    # Scrapling で取得できなかった場合は従来の Playwright 版にフォールバック
    if not jra_odds:
        try:
            from modules.jra_scraper import fetch_jra_odds
            jra_odds = fetch_jra_odds(race_id, headless=True)
            if jra_odds:
                print(f"  [JRA公式/Playwright] リアルタイムオッズを取得しました（{len(jra_odds)}頭）")
            else:
                print("  [JRA公式/Playwright] オッズを取得できませんでした")
        except Exception as e:
            print(f"  [JRA公式/Playwright] エラー: {e}")

    # ---- 二次: netkeiba API（補完用） ----
    netkeiba_odds: dict[int, float] = {}
    try:
        netkeiba_odds = _fetch_netkeiba_odds(race_id)
        if netkeiba_odds:
            print(f"  [netkeiba] オッズを取得しました（{len(netkeiba_odds)}頭）")
    except Exception as e:
        print(f"  [netkeiba] エラー: {e}")

    # ---- マージ: JRA を優先し、不足分を netkeiba で補完 ----
    merged: dict[int, float] = {}
    if netkeiba_odds:
        merged.update(netkeiba_odds)  # まず netkeiba をベースに
    if jra_odds:
        merged.update(jra_odds)  # JRA で上書き（リアルタイム優先）

    return merged


# ---------------------------------------------------------------------------
# 直接実行時のテスト用
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    race_id = sys.argv[1] if len(sys.argv) > 1 else "202606020103"
    print(f"オッズ取得テスト: race_id={race_id}")
    odds_map = fetch_win_odds_map(race_id)
    if odds_map:
        print(f"\n単勝オッズ ({len(odds_map)}頭):")
        for u in sorted(odds_map.keys()):
            print(f"  {u:>2}番: {odds_map[u]:.1f}")
    else:
        print("オッズを取得できませんでした。")
