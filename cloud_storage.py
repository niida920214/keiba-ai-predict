"""
cloud_storage.py -- Hugging Face Hub をデータ・モデル置き場として使うラッパー
==============================================================================
非公開の HF リポジトリに、学習データのバックアップとモデルを保存する。

必要な設定（.streamlit/secrets.toml または環境変数）:
    HF_TOKEN   = "hf_..."           # Write権限のトークン
    HF_REPO_ID = "user/keibaai-data" # 非公開リポジトリ名

ローカルCLIからは sync_data.py 経由で使う。
"""

import os
from pathlib import Path

from modules.constants import local_paths

# 予測に必要な最小セット（公開アプリが起動時に取得する）
PREDICT_FILES = {
    "model/lgbm_model.pickle": local_paths.MODEL_DIR / "lgbm_model.pickle",
    "model/lgbm_ranker.pickle": local_paths.MODEL_DIR / "lgbm_ranker.pickle",
    "model/optimal_stern_r.pickle": local_paths.MODEL_DIR / "optimal_stern_r.pickle",
    "model/optimal_alpha.pickle": local_paths.MODEL_DIR / "optimal_alpha.pickle",
    "data/processed/predict_meta.pickle": local_paths.PROCESSED_DIR / "predict_meta.pickle",
}

# ローカルのフルバックアップ対象（生データ＋前処理済みデータ＋全モデル）
DATA_FILES = {
    "data/raw/results/race_results.pickle": local_paths.RAW_RESULTS_DIR / "race_results.pickle",
    "data/raw/horse_results/horse_results.pickle": local_paths.RAW_HORSE_RESULTS_DIR / "horse_results.pickle",
    "data/raw/peds/peds.pickle": local_paths.RAW_PEDS_DIR / "peds.pickle",
    "data/raw/return_tables/return_tables.pickle": local_paths.RAW_RETURN_TABLES_DIR / "return_tables.pickle",
    "data/processed/data_c.pickle": local_paths.PROCESSED_DIR / "data_c.pickle",
    "data/processed/race_id_list.pickle": local_paths.PROCESSED_DIR / "race_id_list.pickle",
    "data/processed/race_id_list_meta.pickle": local_paths.PROCESSED_DIR / "race_id_list_meta.pickle",
    "model/lgbm_model_win.pickle": local_paths.MODEL_DIR / "lgbm_model_win.pickle",
    "model/simulation_data.pickle": local_paths.MODEL_DIR / "simulation_data.pickle",
    "model/best_params.pickle": local_paths.MODEL_DIR / "best_params.pickle",
    "model/best_params_ranker.pickle": local_paths.MODEL_DIR / "best_params_ranker.pickle",
    "model/best_params_win.pickle": local_paths.MODEL_DIR / "best_params_win.pickle",
    # Optunaのチューニング履歴。クラウドに保持することで、②モデル学習を
    # 実行するたびに前回の探索の続きから再開できる（試行回数が積み上がる）。
    "optuna_study.db": local_paths.BASE_DIR / "optuna_study.db",
}

# シミュレーション結果（管理者パネルでの表示用）※ simulate.py の実際の出力に一致させる
RESULTS_FILES = {
    "results/simulation_summary.csv": local_paths.RESULTS_DIR / "simulation_summary.csv",
    "results/all_strategies_return_rate.png": local_paths.RESULTS_DIR / "all_strategies_return_rate.png",
    "results/all_strategies_vs_nbets.png": local_paths.RESULTS_DIR / "all_strategies_vs_nbets.png",
    "results/harville_vs_winmodel.png": local_paths.RESULTS_DIR / "harville_vs_winmodel.png",
    "results/exotic_strategies_return_rate.png": local_paths.RESULTS_DIR / "exotic_strategies_return_rate.png",
}

ALL_FILES = {**PREDICT_FILES, **DATA_FILES, **RESULTS_FILES}


def _load_secrets_toml() -> dict:
    """.streamlit/secrets.toml を読む（Streamlit外のCLIから使うため）。"""
    path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_config() -> tuple[str, str]:
    """(token, repo_id) を返す。未設定の項目は空文字。

    優先順: 環境変数 > st.secrets (Streamlit実行中) > .streamlit/secrets.toml
    """
    token = os.environ.get("HF_TOKEN", "")
    repo_id = os.environ.get("HF_REPO_ID", "")

    if not (token and repo_id):
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                token = token or st.secrets.get("HF_TOKEN", "")
                repo_id = repo_id or st.secrets.get("HF_REPO_ID", "")
        except Exception:
            pass

    if not (token and repo_id):
        toml_secrets = _load_secrets_toml()
        token = token or toml_secrets.get("HF_TOKEN", "")
        repo_id = repo_id or toml_secrets.get("HF_REPO_ID", "")

    return token, repo_id


def is_configured() -> bool:
    token, repo_id = get_config()
    return bool(token and repo_id)


def _api():
    from huggingface_hub import HfApi
    token, _ = get_config()
    return HfApi(token=token)


def ensure_repo() -> str:
    """リポジトリが無ければ非公開で作成し、repo_id を返す。"""
    token, repo_id = get_config()
    if not (token and repo_id):
        raise RuntimeError("HF_TOKEN / HF_REPO_ID が設定されていません。")
    from huggingface_hub import create_repo
    create_repo(repo_id, token=token, repo_type="dataset", private=True, exist_ok=True)
    return repo_id


def upload_files(files: dict[str, Path], log=print) -> list[str]:
    """ローカルファイルを HF リポジトリへアップロードする。存在しないものはスキップ。"""
    repo_id = ensure_repo()
    api = _api()
    uploaded = []
    for remote_path, local_path in files.items():
        if not local_path.exists():
            log(f"  [SKIP] {local_path.name}（ローカルに存在しません）")
            continue
        size_mb = local_path.stat().st_size / 1024 / 1024
        log(f"  [UP] {remote_path}  ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        uploaded.append(remote_path)
    return uploaded


def download_files(files: dict[str, Path], log=print) -> list[str]:
    """HF リポジトリからローカルパスへダウンロードする。リポジトリに無いものはスキップ。"""
    token, repo_id = get_config()
    if not (token and repo_id):
        raise RuntimeError("HF_TOKEN / HF_REPO_ID が設定されていません。")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    downloaded = []
    for remote_path, local_path in files.items():
        try:
            cached = hf_hub_download(
                repo_id=repo_id, filename=remote_path,
                repo_type="dataset", token=token,
            )
        except EntryNotFoundError:
            log(f"  [SKIP] {remote_path}（クラウドに存在しません）")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(cached, local_path)
        size_mb = local_path.stat().st_size / 1024 / 1024
        log(f"  [DL] {remote_path}  ({size_mb:.1f} MB)")
        downloaded.append(remote_path)
    return downloaded


def list_remote_files() -> list[dict]:
    """リポジトリ上のファイル一覧（パス・サイズ・更新日時）を返す。"""
    token, repo_id = get_config()
    api = _api()
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    return [
        {"path": f.rfilename, "size_mb": round((f.size or 0) / 1024 / 1024, 2)}
        for f in info.siblings
    ]
