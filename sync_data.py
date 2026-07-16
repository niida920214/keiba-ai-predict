"""
sync_data.py -- ローカル⇔クラウドストレージ(Hugging Face Hub) 同期CLI
=======================================================================
管理者がローカルで使うツール。データ・モデルをクラウドにバックアップし、
別PCへの移行やローカル容量の節約（キャッシュ削除）を可能にする。

Usage:
    python sync_data.py upload            # 全データ＋モデルをクラウドへ保存
    python sync_data.py upload --predict  # 予測に必要な最小セットのみ
    python sync_data.py download          # クラウドから全データを復元
    python sync_data.py download --predict
    python sync_data.py list              # クラウド上のファイル一覧
"""

import argparse
import sys

import cloud_storage


def main() -> None:
    parser = argparse.ArgumentParser(description="ローカル⇔クラウド データ同期")
    parser.add_argument("action", choices=["upload", "download", "list"])
    parser.add_argument(
        "--predict", action="store_true",
        help="予測に必要な最小セット（モデル＋predict_meta）のみを対象にする",
    )
    args = parser.parse_args()

    if not cloud_storage.is_configured():
        print("エラー: クラウドストレージが未設定です。")
        print("  .streamlit/secrets.toml（または環境変数）に以下を設定してください:")
        print('    HF_TOKEN   = "hf_..."            # https://huggingface.co/settings/tokens (Write権限)')
        print('    HF_REPO_ID = "ユーザー名/keibaai-data"')
        sys.exit(1)

    files = cloud_storage.PREDICT_FILES if args.predict else cloud_storage.ALL_FILES

    if args.action == "upload":
        print(f"クラウドへアップロード中（{len(files)}ファイル対象）...")
        uploaded = cloud_storage.upload_files(files)
        print(f"\n完了: {len(uploaded)} ファイルをアップロードしました。")
    elif args.action == "download":
        print(f"クラウドからダウンロード中（{len(files)}ファイル対象）...")
        downloaded = cloud_storage.download_files(files)
        print(f"\n完了: {len(downloaded)} ファイルをダウンロードしました。")
    elif args.action == "list":
        print("クラウド上のファイル一覧:")
        for f in cloud_storage.list_remote_files():
            print(f"  {f['path']}  ({f['size_mb']} MB)")


if __name__ == "__main__":
    main()
