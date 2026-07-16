"""
main.py – 競馬データ スクレイピング & 前処理 一気通貫パイプライン
================================================================
1. scrape_kaisai_date で開催日一覧を取得
2. scrape_race_id_list でレース ID リストを自動生成
3. 差分スクレイピング（既取得分はスキップ）
4. 前処理パイプライン → data_c を pickle 保存
"""

import argparse
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from modules.constants import local_paths
# 前処理
from preprocessing import HorseResults, Peds, Results
# スクレイピング
from scraper import HorseResults as ScraperHorseResults
from scraper import Peds as ScraperPeds
from scraper import Results as ScraperResults
from scraper import Return as ScraperReturn
from scraper import scrape_kaisai_date, scrape_race_id_list, update_data

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
DATA_DIR = local_paths.DATA_DIR

# ★ 取得期間のデフォルト: 開始は固定、終了は現在の年月を自動使用
#    （--from-date / --to-date 引数で上書き可能）
DEFAULT_FROM_DATE = "2016-01"
DEFAULT_TO_DATE = datetime.now().strftime("%Y-%m")


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def save_pickle(obj, filepath: Path) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    if isinstance(obj, pd.DataFrame):
        print(f"  → 保存完了: {filepath}  (shape={obj.shape})")
    else:
        print(f"  → 保存完了: {filepath}")


def load_pickle(filepath: Path):
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="競馬データ スクレイピング & 前処理パイプライン")
    parser.add_argument("--clean", action="store_true",
                        help="壊れた horse_results を削除して Ajax API から再取得する")
    parser.add_argument("--clean-all", action="store_true",
                        help="全 raw データ・processed データを削除して完全再取得する")
    parser.add_argument("--from-date", default=DEFAULT_FROM_DATE,
                        help=f"取得開始年月 yyyy-mm（デフォルト: {DEFAULT_FROM_DATE}）")
    parser.add_argument("--to-date", default=DEFAULT_TO_DATE,
                        help="取得終了年月 yyyy-mm（デフォルト: 現在の年月）")
    args = parser.parse_args()

    FROM_DATE = args.from_date
    TO_DATE = args.to_date

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"データ保存先: {DATA_DIR}")
    print(f"取得期間: {FROM_DATE} 〜 {TO_DATE}\n")

    # --- クリーンモード ---
    if args.clean_all:
        print("⚠ --clean-all: 全 raw データ・processed データを削除します。")
        for d in [local_paths.RAW_RESULTS_DIR, local_paths.RAW_HORSE_RESULTS_DIR,
                  local_paths.RAW_PEDS_DIR, local_paths.RAW_RETURN_TABLES_DIR,
                  local_paths.PROCESSED_DIR, local_paths.HTML_RACE_DIR,
                  local_paths.HTML_HORSE_DIR, local_paths.HTML_PED_DIR]:
            if d.exists():
                shutil.rmtree(d)
                print(f"  削除: {d}")
        # processed の race_id キャッシュも削除
    elif args.clean:
        print("⚠ --clean: horse_results を削除して再取得します。")
        hr_path = local_paths.RAW_HORSE_RESULTS_DIR / "horse_results.pickle"
        if hr_path.exists():
            hr_path.unlink()
            print(f"  削除: {hr_path}")
        # processed データも再生成が必要
        for f in local_paths.PROCESSED_DIR.glob("data_c*"):
            f.unlink()
            print(f"  削除: {f}")
        for f in local_paths.PROCESSED_DIR.glob("results_obj*"):
            f.unlink()
            print(f"  削除: {f}")
        print()

    # ==================================================================
    # フェーズ 0: 開催日 → レース ID リストの自動生成
    # ==================================================================
    print("=" * 60)
    print(f"フェーズ 0: レース ID の自動生成 ({FROM_DATE} 〜 {TO_DATE})")
    print("=" * 60)

    race_id_path = local_paths.PROCESSED_DIR / "race_id_list.pickle"
    race_id_meta_path = local_paths.PROCESSED_DIR / "race_id_list_meta.pickle"

    # キャッシュの有効性チェック: FROM_DATE/TO_DATE が変更されていたら再取得
    cache_valid = False
    if race_id_path.exists():
        meta = load_pickle(race_id_meta_path)
        if meta and meta.get("from") == FROM_DATE and meta.get("to") == TO_DATE:
            cache_valid = True

    if cache_valid:
        race_id_list = load_pickle(race_id_path)
        print(f"  既存のレース ID を使用: {len(race_id_list)} 件（スキップ）")
    else:
        if race_id_path.exists():
            print("  [INFO] 期間設定が変更されたため、レース ID を再取得します。")
        print("\n[0-1] 開催日一覧を取得中...")
        kaisai_date_list = scrape_kaisai_date(FROM_DATE, TO_DATE)
        print(f"  取得した開催日: {len(kaisai_date_list)} 日")

        print("\n[0-2] レース ID 一覧を取得中...")
        race_id_list = scrape_race_id_list(kaisai_date_list)
        print(f"  取得したレース ID: {len(race_id_list)} 件")

        if not race_id_list:
            print(
                "  [WARN] レース ID を取得できませんでした。期間設定を確認してください。"
            )
            return

        save_pickle(race_id_list, race_id_path)
        save_pickle({"from": FROM_DATE, "to": TO_DATE}, race_id_meta_path)

    # ==================================================================
    # フェーズ 1: 差分スクレイピング
    # ==================================================================
    print("\n" + "=" * 60)
    print("フェーズ 1: 差分スクレイピング")
    print("=" * 60)

    # --- パス定義 ---
    results_path = local_paths.RAW_RESULTS_DIR / "race_results.pickle"
    horse_results_path = local_paths.RAW_HORSE_RESULTS_DIR / "horse_results.pickle"
    peds_path = local_paths.RAW_PEDS_DIR / "peds.pickle"
    return_path = local_paths.RAW_RETURN_TABLES_DIR / "return_tables.pickle"

    # --- 既存データの読み込み ---
    print("\n既存データを読み込み中...")

    print(f"  レース結果: {results_path.name} ...", end=" ", flush=True)
    old_results = load_pickle(results_path)
    print("OK" if old_results is not None else "なし")

    # horse_results は scrape_via_ajax 内部で差分ロードするため、ここではスキップ
    # （大量データの二重ロードを回避してメモリ効率を改善）
    print(f"  馬の過去成績: scrape_via_ajax 内で差分ロード（スキップ）")
    old_horse_results = None

    print(f"  血統データ: {peds_path.name} ...", end=" ", flush=True)
    old_peds = load_pickle(peds_path)
    print("OK" if old_peds is not None else "なし")

    print(f"  払い戻し: {return_path.name} ...", end=" ", flush=True)
    old_returns = load_pickle(return_path)
    print("OK" if old_returns is not None else "なし")

    # --- 1-1. レース結果（差分） ---
    if old_results is not None:
        existing_race_ids = set(old_results.index.unique())
        new_race_ids = [rid for rid in race_id_list if rid not in existing_race_ids]
    else:
        new_race_ids = race_id_list

    print(
        f"\n[1/4] レース結果（全 {len(race_id_list)} 件中、新規 {len(new_race_ids)} 件）"
    )

    if new_race_ids:
        new_results_df = ScraperResults.scrape(new_race_ids)
        if new_results_df is not None and not new_results_df.empty:
            race_results_df = (
                update_data(old_results, new_results_df)
                if old_results is not None
                else new_results_df
            )
            save_pickle(race_results_df, results_path)
        else:
            print("  [WARN] 新規レース結果を取得できませんでした。")
            race_results_df = old_results
    else:
        print("  [OK] すべて取得済み。スキップします。")
        race_results_df = old_results

    if race_results_df is None or race_results_df.empty:
        print("  [WARN] レース結果データがありません。処理を中断します。")
        return

    horse_id_list = race_results_df["horse_id"].unique().tolist()

    # --- 1-2. 馬の過去成績（Ajax API で統一取得） ---
    print(
        f"\n[2/4] 馬の過去成績（全 {len(horse_id_list)} 頭）"
    )

    # save_path を渡すことで中間保存 & 差分取得が有効になる
    horse_results_df = ScraperHorseResults.scrape_via_ajax(
        horse_id_list, save_path=horse_results_path
    )
    if not horse_results_df.empty:
        save_pickle(horse_results_df, horse_results_path)
    else:
        print("  [WARN] 過去成績を取得できませんでした。")
        if horse_results_path.exists():
            horse_results_df = load_pickle(horse_results_path)

    # --- 1-3. 血統データ（差分） ---
    if old_peds is not None:
        existing_ped_ids = set(old_peds.index.unique())
        new_ped_ids = [hid for hid in horse_id_list if hid not in existing_ped_ids]
    else:
        new_ped_ids = horse_id_list

    print(
        f"\n[3/4] 血統データ（全 {len(horse_id_list)} 頭中、新規 {len(new_ped_ids)} 頭）"
    )

    if new_ped_ids:
        new_peds_df = ScraperPeds.scrape(new_ped_ids)
        if not new_peds_df.empty:
            peds_df = (
                update_data(old_peds, new_peds_df)
                if old_peds is not None
                else new_peds_df
            )
            save_pickle(peds_df, peds_path)
        else:
            print("  [WARN] 新規の血統データを取得できませんでした。")
            peds_df = old_peds
    else:
        print("  [OK] すべて取得済み。スキップします。")
        peds_df = old_peds

    # --- 1-4. 払い戻しデータ（差分） ---
    if old_returns is not None:
        existing_return_ids = set(old_returns.index.unique())
        new_return_ids = [rid for rid in race_id_list if rid not in existing_return_ids]
    else:
        new_return_ids = race_id_list

    print(
        f"\n[4/4] 払い戻しデータ（全 {len(race_id_list)} 件中、新規 {len(new_return_ids)} 件）"
    )

    if new_return_ids:
        new_return_df = ScraperReturn.scrape(new_return_ids)
        if not new_return_df.empty:
            return_tables_df = (
                update_data(old_returns, new_return_df)
                if old_returns is not None
                else new_return_df
            )
            save_pickle(return_tables_df, return_path)
        else:
            print("  [WARN] 新規の払い戻しデータを取得できませんでした。")
    else:
        print("  [OK] すべて取得済み。スキップします。")

    # ==================================================================
    # フェーズ 2: 前処理パイプライン
    # ==================================================================
    print("\n" + "=" * 60)
    print("フェーズ 2: 前処理パイプライン")
    print("=" * 60)

    print("\n[Step 1] Results.preprocessing() ...")
    r = Results(race_results_df)
    r.preprocessing()
    print(f"  data_p shape: {r.data_p.shape}")
    # Phase 3: finishing_position, time_sec, intra-race features
    new_cols = ["finishing_position", "time_sec", "weight_rank", "age_diff", "impost_rank"]
    found = [c for c in new_cols if c in r.data_p.columns]
    print(f"  New features: {found}")

    print("\n[Step 2] HorseResults preprocessing ...")
    hr = HorseResults(horse_results_df)
    print(f"  horse_results shape: {hr.horse_results.shape}")

    print("\n[Step 3] Results.merge_horse_results() ...")
    r.merge_horse_results(hr)
    print(f"  data_h shape: {r.data_h.shape}")
    # Phase 3: pace features
    pace_cols = ["n_nige", "n_senkou", "pace_pressure", "legtype_pace_interaction"]
    found_pace = [c for c in pace_cols if c in r.data_h.columns]
    print(f"  Pace features: {found_pace}")

    print("\n[Step 4] Peds.encode() ...")
    p = Peds(peds_df)
    p.encode()
    print(f"  peds_e shape: {p.peds_e.shape}")

    print("\n[Step 5] Results.merge_peds() ...")
    r.merge_peds(p.peds_e)
    print(f"  data_pe shape: {r.data_pe.shape}")

    print("\n[Step 6] Results.process_categorical() ...")
    r.process_categorical()
    print(f"  data_c shape: {r.data_c.shape}")
    # Final verification
    all_new = [c for c in new_cols + pace_cols if c in r.data_c.columns]
    print(f"  All new features in data_c: {all_new}")

    print("\n[Save] Saving preprocessed data...")
    save_pickle(r.data_c, local_paths.PROCESSED_DIR / "data_c.pickle")

    # predict.py が必要とするのは LabelEncoder とカテゴリ一覧のみ。
    # Results オブジェクト全体 (数GB) の代わりに軽量メタデータを保存する。
    predict_meta = {
        "le_horse": r.le_horse,
        "le_jockey": r.le_jockey,
        "categories": {
            col: r.data_pe[col].unique()
            for col in ["weather", "race_type", "ground_state", "性"]
        },
    }
    save_pickle(predict_meta, local_paths.PROCESSED_DIR / "predict_meta.pickle")

    # ==================================================================
    # 完了サマリ
    # ==================================================================
    print("\n" + "=" * 60)
    print("全処理が完了しました！")
    print("=" * 60)
    print(f"\n保存先: {DATA_DIR}")
    for p_file in sorted(DATA_DIR.glob("*.pickle")):
        size_kb = p_file.stat().st_size / 1024
        print(f"  [FILE] {p_file.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
