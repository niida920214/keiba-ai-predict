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

## アクセス制限（二段階パスワード）

公開UIは2種類のパスワードで保護されています。

| ロール | パスワード | できること |
|---|---|---|
| 一般ユーザー | `APP_PASSWORD` | レース予測のみ |
| 管理者 | `ADMIN_PASSWORD` | 予測＋管理者パネル（モデル更新・クラウド同期） |

ログイン画面は共通で、入力されたパスワードによってロールが自動判定されます。

### ローカル実行

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# .streamlit/secrets.toml を編集して APP_PASSWORD / ADMIN_PASSWORD を設定する
streamlit run streamlit_app.py
```

`.streamlit/secrets.toml` は `.gitignore` 対象なのでコミットされません。

## Streamlit Community Cloud へのデプロイ

1. このリポジトリをGitHubにpushする
2. https://share.streamlit.io で "New app" → リポジトリを選択 → Main file path に `streamlit_app.py`
3. デプロイ後、**Settings → Secrets** に以下を貼り付けて保存

   ```toml
   APP_PASSWORD = "一般ユーザー用パスワード"
   ADMIN_PASSWORD = "管理者専用パスワード"

   # クラウドストレージを使う場合（推奨）
   HF_TOKEN = "hf_..."
   HF_REPO_ID = "ユーザー名/keibaai-data"
   ```

## クラウドストレージ（Hugging Face Hub）

学習データとモデルの正本をクラウド（無料・非公開）に保管できます。
ローカルPCの容量圧迫や故障への備えになり、公開アプリは起動時に
クラウドから最新モデルを自動取得します。

### 初回セットアップ

1. https://huggingface.co で無料アカウントを作成
2. https://huggingface.co/settings/tokens で **Write権限** のトークンを作成
3. ローカルの `.streamlit/secrets.toml` と Streamlit Cloud の Secrets の両方に
   `HF_TOKEN` と `HF_REPO_ID`（例: `あなたのHFユーザー名/keibaai-data`）を追加
4. ローカルで初回アップロード:

   ```bash
   python sync_data.py upload    # データ＋モデル一式をクラウドへ（初回は数分）
   python sync_data.py list      # 保存されたか確認
   ```

リポジトリは初回アップロード時に**非公開**で自動作成されます。

### 運用

- `python sync_data.py upload` … ローカルの最新データ・モデルをクラウドへ保存
- `python sync_data.py download` … クラウドから復元（PC移行・故障復旧時）
- 公開アプリの管理者パネルからも、モデルのアップロード／取り込みが可能

## モデル・データを更新したとき

反映方法は3通りあります（どれでも可）:

- **管理者パネルから（推奨）**: 「🔁 パイプライン実行」でGitHub Actions上で
  データ更新〜学習〜シミュレーション〜クラウド保存まで全自動実行
- **ローカル＋クラウド経由**: `python main.py` 等を実行後 `python sync_data.py upload`
  → 公開アプリの再起動時に自動反映
  （すぐ反映したい場合は管理者パネルの「クラウドの最新モデルを取り込む」）
- **Git経由**: `git add model data/processed/predict_meta.pickle && git commit && git push`

## パイプラインのリモート実行（GitHub Actions）

管理者パネルの「🔁 パイプライン実行」から、データ更新（期間指定可）・モデル学習
（Optuna試行回数指定可）・シミュレーションを、GitHub Actionsのクラウド実行環境
（メモリ16GB・1ジョブ最長約6時間）で実行できます。データの受け渡しは
Hugging Face Hub 経由で全自動です。

### 初回セットアップ

1. **GitHubリポジトリのSecrets設定**（Actions実行時にHFへアクセスするため）
   - リポジトリページ → Settings → Secrets and variables → **Actions** → New repository secret
   - `HF_TOKEN` と `HF_REPO_ID` の2つを登録（値は `.streamlit/secrets.toml` と同じ）
2. **GitHubのFine-grainedトークン作成**（アプリからActionsを起動するため）
   - https://github.com/settings/personal-access-tokens → Generate new token
   - Repository access: 対象リポジトリのみ選択
   - Permissions → Repository permissions → **Actions: Read and write**
3. **Streamlit CloudのSecretsに追加**
   ```toml
   GH_TOKEN = "github_pat_..."
   GH_REPO  = "ユーザー名/keiba-ai-predict"
   ```

### 注意

- 学習（試行回数200）は数時間かかります。6時間の上限を超えそうな場合は
  試行回数を減らしてください。
- GitHubの無料枠は**プライベートリポジトリで月2,000分**です（パブリックは無制限）。
- GitHubのデータセンターIPからのスクレイピングがnetkeiba側にブロックされる
  可能性があります。初回実行で失敗する場合はローカル実行に切り替えてください。

## 注意事項

- 本ツールは個人の研究目的で作成した予測モデルです。的中や利益を保証するものではありません。
- レース予測のたびにnetkeiba等へアクセスするため、過度な連続実行は避けてください。
- クラウド環境ではPlaywrightブラウザが無いため、オッズはnetkeiba API（無料枠は1日5回更新）
  へのフォールバックになります。リアルタイムオッズで予測したい場合はローカルで実行してください。
