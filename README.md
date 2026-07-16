# 競馬AI (keibaAI)

LightGBM による競馬予測システム。データ収集からモデル学習・回収率シミュレーション・
レース予測までを含む本体プロジェクトです。

このリポジトリはそのまま Streamlit Community Cloud のデプロイソースになります
（エントリポイント: `streamlit_app.py`）。巨大な学習データ（`data/`）や出力
（`results/`）は `.gitignore` で除外されており、リポジトリには予測に必要な
学習済みモデル（`model/`）と軽量メタデータ（`data/processed/predict_meta.pickle`）
だけが含まれます。

## 構成

| ファイル | 役割 | 実行場所 |
|---|---|---|
| `streamlit_app.py` | レース予測UI（パスワード保護付き） | **公開デプロイ用** / ローカル可 |
| `app.py` | 管理者用コントロールパネル（データ更新・学習・シミュレーション・予測） | ローカル専用（`run_ui.bat`） |
| `main.py` | データ更新（差分スクレイピング＋前処理） | ローカル専用 |
| `train_model.py` | モデル学習（Optuna） | ローカル専用 |
| `simulate.py` | 回収率シミュレーション | ローカル専用 |
| `predict.py` | レース予測CLI（UIの中身でもある） | 両方 |

運用手順の詳細は `weekend_routine.md` を参照。

## アクセス制限（パスワード）

公開UIは共有パスワード方式のゲートで保護されています。

### ローカル実行

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# .streamlit/secrets.toml を編集して APP_PASSWORD を好きな値に変更する
streamlit run streamlit_app.py
```

`.streamlit/secrets.toml` は `.gitignore` 対象なのでコミットされません。

## Streamlit Community Cloud へのデプロイ

1. このリポジトリをGitHubにpushする
2. https://share.streamlit.io で "New app" → リポジトリを選択 → Main file path に `streamlit_app.py`
3. デプロイ後、**Settings → Secrets** に以下を貼り付けて保存

   ```toml
   APP_PASSWORD = "配りたいパスワード"
   ```

## モデル・データを更新したとき

ローカルで `main.py`（データ更新）や `train_model.py`（再学習）を実行すると、
`model/*.pickle` と `data/processed/predict_meta.pickle` が更新されます。
それらは Git 管理対象なので、**そのまま commit → push すれば公開版にも自動反映**されます。

```bash
git add model data/processed/predict_meta.pickle
git commit -m "Update model"
git push
```

## 注意事項

- 本ツールは個人の研究目的で作成した予測モデルです。的中や利益を保証するものではありません。
- レース予測のたびにnetkeiba等へアクセスするため、過度な連続実行は避けてください。
- クラウド環境ではPlaywrightブラウザが無いため、オッズはnetkeiba API（無料枠は1日5回更新）
  へのフォールバックになります。リアルタイムオッズで予測したい場合はローカルで実行してください。
