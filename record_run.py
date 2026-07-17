"""
record_run.py -- パイプライン実行の永続記録
=============================================
GitHub Actions のパイプライン終盤で呼ばれ、実行記録1行を
Hugging Face 上の logs/run_history.jsonl に追記する。

記録内容: 開始時刻(JST)・実行者・実行した段階(①〜③)と成否・所要時間・
          データに含まれる最新開催日

環境変数:
    START_TS        : ジョブ開始時刻 (unix秒)
    OPERATOR        : 実行者名
    OUTCOME_UPDATE  : ① の結果 (success/failure/skipped)
    OUTCOME_TRAIN   : ② の結果
    OUTCOME_SIM     : ③ の結果
    DRY_RUN         : "1" なら記録内容を表示するだけでアップロードしない
"""

import json
import os
import pickle
import time
from datetime import datetime, timedelta, timezone

from modules.constants import local_paths

RUN_LOG_REMOTE = "logs/run_history.jsonl"


def data_through() -> str:
    """data_c に含まれる最新の開催日 (yyyy-mm-dd) を返す。"""
    path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not path.exists():
        return "-"
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
        return str(df["date"].max())[:10]
    except Exception as e:
        print(f"  [WARN] data_c から日付を取得できませんでした: {e}")
        return "-"


def build_record() -> dict:
    start_ts = float(os.environ.get("START_TS", time.time()))
    started_jst = datetime.fromtimestamp(start_ts, timezone.utc) + timedelta(hours=9)
    elapsed_min = int((time.time() - start_ts) // 60)

    outcomes = {
        "update": os.environ.get("OUTCOME_UPDATE", "skipped"),
        "train": os.environ.get("OUTCOME_TRAIN", "skipped"),
        "simulate": os.environ.get("OUTCOME_SIM", "skipped"),
    }
    executed = [v for v in outcomes.values() if v != "skipped"]
    overall = "failure" if "failure" in executed else ("success" if executed else "empty")

    return {
        "started_jst": started_jst.strftime("%Y-%m-%d %H:%M"),
        "operator": os.environ.get("OPERATOR", "") or "-",
        "steps": outcomes,
        "result": overall,
        "elapsed_min": elapsed_min,
        "data_through": data_through(),
    }


def append_to_cloud(record: dict) -> None:
    import cloud_storage
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    token, repo_id = cloud_storage.get_config()
    lines = []
    try:
        cached = hf_hub_download(
            repo_id=repo_id, filename=RUN_LOG_REMOTE,
            repo_type="dataset", token=token,
        )
        with open(cached, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except EntryNotFoundError:
        pass

    lines.append(json.dumps(record, ensure_ascii=False))

    tmp = local_paths.BASE_DIR / "run_history.jsonl.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cloud_storage._api().upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=RUN_LOG_REMOTE,
        repo_id=repo_id,
        repo_type="dataset",
    )
    tmp.unlink()
    print(f"  実行記録を追記しました（計 {len(lines)} 件）")


def main() -> None:
    record = build_record()
    print("実行記録:", json.dumps(record, ensure_ascii=False, indent=2))
    if os.environ.get("DRY_RUN") == "1":
        print("(DRY_RUN=1 のためアップロードはスキップ)")
        return
    append_to_cloud(record)


if __name__ == "__main__":
    main()
