"""
streamlit_app.py -- 競馬AI レース予測（公開版）
==================================================
学習済みモデルを使ってレースIDから予測を行う公開用UI。

アクセス制御は二段階:
    - APP_PASSWORD   : 一般ユーザー（レース予測のみ）
    - ADMIN_PASSWORD : 管理者（予測＋管理者パネル: モデル更新・クラウド同期）

重い処理（データ更新・モデル再学習）はローカル運用専用。
"""

import hmac
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

import cloud_storage
from modules.constants import local_paths
from position_probability import PositionProbabilityEstimator
from predict import (
    build_combo_table,
    build_result_table,
    fetch_and_preprocess,
    fetch_market_odds,
    get_horse_names,
    get_umaban,
    integrate_benter,
    load_alpha,
    load_pickle_if_exists,
    predict_fundamental,
    predict_top3,
)

st.set_page_config(page_title="競馬AI レース予測", page_icon="🏇", layout="wide")


# ---------------------------------------------------------------------------
# 起動時クラウド同期（設定されていれば最新モデルを取得）
# ---------------------------------------------------------------------------
@st.cache_resource
def startup_cloud_sync() -> str:
    """プロセス起動時に一度だけ、クラウドから最新の予測用ファイルを取得する。"""
    if not cloud_storage.is_configured():
        return "クラウドストレージ未設定（リポジトリ同梱のモデルを使用）"
    try:
        downloaded = cloud_storage.download_files(
            cloud_storage.PREDICT_FILES, log=lambda _: None
        )
        return f"クラウドから {len(downloaded)} ファイルを取得済み"
    except Exception as e:
        return f"クラウド同期に失敗（同梱モデルで継続）: {e}"


# ---------------------------------------------------------------------------
# 二段階パスワード認証
# ---------------------------------------------------------------------------
def check_password() -> str | None:
    """認証済みなら 'admin' か 'user' を返す。未認証ならログイン画面を出して None。"""

    role = st.session_state.get("role")
    if role in ("admin", "user"):
        return role

    def password_entered():
        entered = st.session_state.get("password_input", "")
        admin_pw = st.secrets.get("ADMIN_PASSWORD", "")
        user_pw = st.secrets.get("APP_PASSWORD", "")
        if admin_pw and hmac.compare_digest(entered, admin_pw):
            st.session_state["role"] = "admin"
        elif user_pw and hmac.compare_digest(entered, user_pw):
            st.session_state["role"] = "user"
        else:
            st.session_state["auth_failed"] = True
        st.session_state["password_input"] = ""

    st.title("🏇 競馬AI レース予測")
    st.text_input(
        "パスワード", type="password", key="password_input", on_change=password_entered
    )
    if st.session_state.get("auth_failed"):
        st.error("パスワードが違います。")
    return None


# ---------------------------------------------------------------------------
# 予測ページ
# ---------------------------------------------------------------------------
MIN_INTERVAL_SEC = 15  # 連打による外部サイトへの過剰アクセスを防ぐための簡易クールダウン


def render_predict() -> None:
    st.caption(
        "netkeibaの出馬表URL末尾にある12桁のレースIDを入力してください。"
        "例: `race_id=202605010811` -> `202605010811`"
    )
    st.info(
        "本ツールは個人の研究目的で作成した予測モデルです。的中や利益を保証するものではありません。",
        icon="ℹ️",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        race_id = st.text_input("レースID（12桁）", max_chars=12, placeholder="202605010811")
    with col2:
        race_date = st.date_input("開催日", value=datetime.now())

    valid_race_id = bool(race_id) and race_id.isdigit() and len(race_id) == 12
    if race_id and not valid_race_id:
        st.error("レースIDは半角数字12桁で入力してください。")

    model_exists = (local_paths.MODEL_DIR / "lgbm_model.pickle").exists() or \
        (local_paths.MODEL_DIR / "lgbm_ranker.pickle").exists()
    if not model_exists:
        st.error("学習済みモデルが見つかりません。管理者にお問い合わせください。")

    last_call = st.session_state.get("last_call_ts", 0.0)
    cooldown_remaining = MIN_INTERVAL_SEC - (time.time() - last_call)

    run = st.button(
        "▶ 予測する",
        type="primary",
        disabled=not (valid_race_id and model_exists) or cooldown_remaining > 0,
    )
    if cooldown_remaining > 0:
        st.caption(f"連続実行防止のため、あと{cooldown_remaining:.0f}秒お待ちください。")

    if run:
        st.session_state["last_call_ts"] = time.time()
        date_str = race_date.strftime("%Y/%m/%d")
        with st.spinner("出馬表・過去成績・オッズを取得して予測しています…（20〜40秒程度）"):
            try:
                st_table = fetch_and_preprocess(race_id, date_str)

                classifier = load_pickle_if_exists(local_paths.MODEL_DIR / "lgbm_model.pickle")
                ranker_model = load_pickle_if_exists(local_paths.MODEL_DIR / "lgbm_ranker.pickle")

                odds_map = fetch_market_odds(race_id)

                drop_cols = ["date", "target", "finishing_position", "time_sec"]
                X = st_table.data_c.drop(
                    [c for c in drop_cols if c in st_table.data_c.columns], axis=1
                )
                umaban = get_umaban(st_table, len(X))

                p_fundamental = predict_fundamental(X, ranker_model, classifier)
                p_market = None
                if odds_map:
                    inv = np.array([
                        1.0 / odds_map[u] if odds_map.get(u, 0) and odds_map[u] > 0 else 0.0
                        for u in umaban
                    ])
                    total = inv.sum()
                    if total > 0:
                        p_market = inv / total

                p_win_final = integrate_benter(p_fundamental, p_market, load_alpha())

                if p_win_final is None:
                    st.error("予測できませんでした（モデル・市場オッズのいずれも取得できません）。")
                else:
                    p_top3_model = predict_top3(X, classifier)
                    stern_r = PositionProbabilityEstimator.load_stern_r()
                    estimator = PositionProbabilityEstimator(p_win_final, umaban=umaban, stern_r=stern_r)
                    p_place = np.array([estimator.p_place(int(u)) for u in umaban])

                    horse_names = get_horse_names(st_table, len(p_win_final))
                    df_res = build_result_table(
                        race_id, date_str, umaban, horse_names,
                        p_win_final, p_place, p_top3_model, p_fundamental, p_market, odds_map,
                    )
                    df_combo = build_combo_table(estimator, race_id, date_str)
                    st.session_state["last_result"] = (race_id, df_res, df_combo)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

    if "last_result" in st.session_state:
        shown_race_id, df_res, df_combo = st.session_state["last_result"]
        st.markdown("---")
        st.subheader(f"予測結果（レースID: {shown_race_id}）")

        display_cols = [
            "予測順位", "馬番", "馬名", "P(Win)", "P(Place)",
            "単勝オッズ", "人気", "EV", "Kelly",
        ]
        display_cols = [c for c in display_cols if c in df_res.columns]
        st.dataframe(df_res[display_cols], width="stretch", hide_index=True)

        if "EV" in df_res.columns and "Kelly" in df_res.columns:
            bets = df_res[(df_res["EV"] > 1.0) & (df_res["Kelly"] > 0)]
            if not bets.empty:
                st.markdown("**推奨買い目（単勝 EV > 1.0 かつ Kelly > 0）**")
                st.dataframe(bets[display_cols], width="stretch", hide_index=True)

        st.markdown("**馬連 / ワイド / 三連複 の上位候補**")
        st.dataframe(df_combo, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# GitHub Actions 連携（パイプラインのリモート実行）
# ---------------------------------------------------------------------------
GH_WORKFLOW_FILE = "pipeline.yml"


def gh_config() -> tuple[str, str]:
    """(token, "owner/repo") を返す。未設定の項目は空文字。"""
    token = st.secrets.get("GH_TOKEN", "")
    repo = st.secrets.get("GH_REPO", "")
    return token, repo


def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def gh_dispatch_pipeline(inputs: dict) -> tuple[bool, str]:
    """workflow_dispatch でパイプラインを起動する。"""
    token, repo = gh_config()
    url = (f"https://api.github.com/repos/{repo}/actions/workflows/"
           f"{GH_WORKFLOW_FILE}/dispatches")
    res = requests.post(
        url, headers=gh_headers(token),
        json={"ref": "main", "inputs": inputs}, timeout=30,
    )
    if res.status_code == 204:
        return True, "起動しました"
    return False, f"HTTP {res.status_code}: {res.text[:300]}"


GH_STATUS_JP = {
    "queued": "待機中", "in_progress": "実行中", "completed": "完了",
    "waiting": "待機中", "requested": "待機中", "pending": "待機中",
}
GH_CONCLUSION_JP = {
    "success": "✅ 成功", "failure": "❌ 失敗", "cancelled": "中断",
    "timed_out": "タイムアウト", "skipped": "スキップ",
}


def _iso_to_jst(iso: str) -> datetime:
    """GitHub APIのUTC時刻文字列を日本時間のdatetimeに変換する。"""
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=9)


def gh_recent_runs(limit: int = 5) -> list[dict]:
    """パイプラインの直近の実行履歴を返す（時刻は日本時間）。"""
    token, repo = gh_config()
    url = (f"https://api.github.com/repos/{repo}/actions/workflows/"
           f"{GH_WORKFLOW_FILE}/runs")
    res = requests.get(url, headers=gh_headers(token),
                       params={"per_page": limit}, timeout=30)
    res.raise_for_status()
    runs = []
    for r in res.json().get("workflow_runs", []):
        started = _iso_to_jst(r["created_at"])
        if r["status"] == "completed":
            elapsed = _iso_to_jst(r["updated_at"]) - started
        else:
            elapsed = datetime.utcnow() + timedelta(hours=9) - started
        runs.append({
            "id": r["id"],
            "status": r["status"],
            "開始時刻(日本時間)": started.strftime("%m/%d %H:%M"),
            "状態": GH_STATUS_JP.get(r["status"], r["status"]),
            "結果": GH_CONCLUSION_JP.get(r["conclusion"], r["conclusion"] or "—"),
            "所要時間": f"{int(elapsed.total_seconds() // 60)}分",
            "URL": r["html_url"],
        })
    return runs


def gh_run_steps(run_id: int) -> list[dict]:
    """実行中ジョブのステップごとの進捗を返す。"""
    token, repo = gh_config()
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    res = requests.get(url, headers=gh_headers(token), timeout=30)
    res.raise_for_status()
    jobs = res.json().get("jobs", [])
    if not jobs:
        return []
    steps = []
    for s in jobs[0].get("steps", []):
        if s["status"] == "completed":
            icon = {"success": "✅", "failure": "❌", "skipped": "⏭️"}.get(
                s["conclusion"], "✅")
        elif s["status"] == "in_progress":
            icon = "🔄"
        else:
            icon = "⏳"
        steps.append({"icon": icon, "name": s["name"], "status": s["status"]})
    return steps


# ---------------------------------------------------------------------------
# 管理者パネル
# ---------------------------------------------------------------------------
# アップロードを受け付けるファイル名 → 保存先
UPLOADABLE_FILES = {
    "lgbm_model.pickle": local_paths.MODEL_DIR / "lgbm_model.pickle",
    "lgbm_ranker.pickle": local_paths.MODEL_DIR / "lgbm_ranker.pickle",
    "lgbm_model_win.pickle": local_paths.MODEL_DIR / "lgbm_model_win.pickle",
    "optimal_stern_r.pickle": local_paths.MODEL_DIR / "optimal_stern_r.pickle",
    "optimal_alpha.pickle": local_paths.MODEL_DIR / "optimal_alpha.pickle",
    "predict_meta.pickle": local_paths.PROCESSED_DIR / "predict_meta.pickle",
}


def render_admin() -> None:
    st.subheader("🛠 管理者パネル")

    # --- 現在有効なモデルの状態 ---
    st.markdown("**現在有効なモデル・データ**")
    rows = []
    for name, path in UPLOADABLE_FILES.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            rows.append({"ファイル": name, "更新日時": mtime,
                         "サイズ(MB)": round(path.stat().st_size / 1024 / 1024, 2)})
        else:
            rows.append({"ファイル": name, "更新日時": "（なし）", "サイズ(MB)": None})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("---")

    # --- クラウドストレージ ---
    st.markdown("**クラウドストレージ（Hugging Face Hub）**")
    if cloud_storage.is_configured():
        _, repo_id = cloud_storage.get_config()
        st.success(f"接続設定済み: `{repo_id}`（非公開）")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬇ クラウドの最新モデルを取り込む"):
                try:
                    with st.spinner("ダウンロード中..."):
                        downloaded = cloud_storage.download_files(
                            cloud_storage.PREDICT_FILES, log=lambda _: None
                        )
                    st.success(f"{len(downloaded)} ファイルを取り込みました。以後の予測に反映されます。")
                except Exception as e:
                    st.error(f"取り込みに失敗しました: {e}")
        with col2:
            if st.button("📄 クラウド上のファイル一覧を表示"):
                try:
                    st.dataframe(pd.DataFrame(cloud_storage.list_remote_files()),
                                 width="stretch", hide_index=True)
                except Exception as e:
                    st.error(f"一覧の取得に失敗しました: {e}")
    else:
        st.warning(
            "クラウドストレージが未設定です。Streamlit Cloud の Settings → Secrets に "
            "`HF_TOKEN`（Hugging FaceのWriteトークン）と `HF_REPO_ID` を追加すると、"
            "モデルのクラウド保存・自動配信が有効になります。"
        )

    st.markdown("---")

    # --- モデルのアップロード ---
    st.markdown("**新しいモデル・データのアップロード**")
    st.caption(
        "ローカルで学習した `model/*.pickle` や `predict_meta.pickle` をここから反映できます。"
        "対応ファイル名: " + ", ".join(f"`{n}`" for n in UPLOADABLE_FILES)
    )
    uploads = st.file_uploader(
        "ファイルを選択（複数可）", accept_multiple_files=True, key="admin_uploads"
    )
    if uploads and st.button("⬆ アップロードして反映", type="primary"):
        applied, rejected = [], []
        for up in uploads:
            dest = UPLOADABLE_FILES.get(up.name)
            if dest is None:
                rejected.append(up.name)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(up.getbuffer())
            applied.append(up.name)

        if applied:
            st.success(f"反映しました: {', '.join(applied)}")
            if cloud_storage.is_configured():
                try:
                    with st.spinner("クラウドストレージへ保存中..."):
                        cloud_storage.upload_files(
                            {f"model/{n}" if "meta" not in n
                             else f"data/processed/{n}": UPLOADABLE_FILES[n]
                             for n in applied},
                            log=lambda _: None,
                        )
                    st.success("クラウドにも保存しました（次回起動時以降も維持されます）。")
                except Exception as e:
                    st.error(
                        f"クラウド保存に失敗: {e}\n"
                        "※ このままだとアプリ再起動時に元のモデルへ戻ります。"
                    )
            else:
                st.warning(
                    "クラウド未設定のため、この反映はアプリ再起動までの一時的なものです。"
                    "恒久反映には git push か、クラウドストレージの設定が必要です。"
                )
        if rejected:
            st.error(f"対応外のファイル名のためスキップ: {', '.join(rejected)}")

    st.markdown("---")

    # --- パイプラインのリモート実行 (GitHub Actions) ---
    st.markdown("**🔁 パイプライン実行（GitHub Actions）**")
    gh_token, gh_repo = gh_config()
    if not (gh_token and gh_repo):
        st.warning(
            "GitHub Actions 連携が未設定です。Secrets に `GH_TOKEN`"
            "（GitHubのFine-grainedトークン: 対象リポジトリの Actions Read/Write 権限）と "
            "`GH_REPO`（例: `niida920214/keiba-ai-predict`）を追加すると、"
            "データ更新・学習・シミュレーションをここから実行できます。"
        )
    else:
        st.caption(
            f"実行環境: GitHub Actions（`{gh_repo}`、メモリ16GB・最長約6時間）。"
            "データはクラウドストレージから取得し、完了後に自動でクラウドへ保存されます。"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            do_update = st.checkbox("① データ更新 (main.py)", value=True)
        with col2:
            do_train = st.checkbox("② モデル学習 (train_model.py)", value=False)
        with col3:
            do_simulate = st.checkbox("③ シミュレーション (simulate.py)", value=False)

        with st.expander("詳細オプション"):
            col1, col2, col3 = st.columns(3)
            with col1:
                from_date = st.text_input("取得開始年月 (yyyy-mm)", value="",
                                          placeholder="空欄=2016-01")
            with col2:
                to_date = st.text_input("取得終了年月 (yyyy-mm)", value="",
                                        placeholder="空欄=今月")
            with col3:
                trials = st.number_input("Optuna試行回数", min_value=10,
                                         max_value=500, value=200, step=10)
            if do_train:
                st.caption(
                    "⚠ 学習は試行回数200で数時間かかります。GitHub Actionsの上限"
                    "（1ジョブ約6時間）を超えそうな場合は試行回数を減らしてください。"
                )

        if st.button("🚀 パイプラインを起動", type="primary",
                     disabled=not (do_update or do_train or do_simulate)):
            ok, msg = gh_dispatch_pipeline({
                "run_update": do_update,
                "run_train": do_train,
                "run_simulate": do_simulate,
                "from_date": from_date.strip(),
                "to_date": to_date.strip(),
                "trials": str(int(trials)),
            })
            if ok:
                st.success(
                    "パイプラインを起動しました。進行状況は下の実行履歴"
                    "（反映まで数秒かかります）で確認できます。"
                    "完了後は「クラウドの最新モデルを取り込む」で反映してください。"
                )
            else:
                st.error(f"起動に失敗しました: {msg}")

        if st.button("🔄 実行履歴・進捗を更新"):
            pass  # ボタン押下でrerunされ、下の履歴が再取得される

        try:
            runs = gh_recent_runs()
            if runs:
                # 実行中のrunがあればステップ単位の進捗を表示
                active = next((r for r in runs if r["status"] != "completed"), None)
                if active:
                    st.markdown(
                        f"**現在の進捗**（開始: {active['開始時刻(日本時間)']}、"
                        f"経過: {active['所要時間']}）"
                    )
                    try:
                        steps = gh_run_steps(active["id"])
                        for s in steps:
                            st.markdown(f"{s['icon']} {s['name']}")
                        if not steps:
                            st.caption("ジョブ起動待ちです（数十秒後に更新してください）。")
                    except Exception:
                        st.caption("進捗の取得に失敗しました（更新で再試行できます）。")
                    st.caption("※ この画面は自動更新されません。「🔄 実行履歴・進捗を更新」で最新化してください。")

                df_runs = pd.DataFrame(runs).drop(columns=["id", "status"])
                st.dataframe(
                    df_runs, width="stretch", hide_index=True,
                    column_config={"URL": st.column_config.LinkColumn("ログ")},
                )
            else:
                st.caption("実行履歴はまだありません。")
        except Exception as e:
            st.error(f"実行履歴の取得に失敗しました: {e}")

    st.markdown("---")

    # --- シミュレーション結果の表示 ---
    st.markdown("**📊 シミュレーション結果**")
    if cloud_storage.is_configured():
        if st.button("クラウドから最新の結果を取得して表示"):
            try:
                with st.spinner("取得中..."):
                    cloud_storage.download_files(
                        cloud_storage.RESULTS_FILES, log=lambda _: None
                    )
                st.session_state["show_sim_results"] = True
            except Exception as e:
                st.error(f"取得に失敗しました: {e}")
        if st.session_state.get("show_sim_results"):
            summary_path = local_paths.RESULTS_DIR / "simulation_summary.csv"
            if summary_path.exists() and summary_path.stat().st_size > 0:
                try:
                    st.dataframe(pd.read_csv(summary_path), width="stretch")
                except Exception:
                    pass
            cols = st.columns(2)
            shown = 0
            for remote, path in cloud_storage.RESULTS_FILES.items():
                if path.suffix == ".png" and path.exists():
                    with cols[shown % 2]:
                        st.image(str(path), caption=path.name)
                    shown += 1
            if shown == 0:
                st.caption("表示できる結果画像がまだありません（シミュレーション未実行）。")
    else:
        st.caption("クラウドストレージ設定後に利用できます。")

    st.markdown("---")
    st.caption(
        "パイプラインはローカルでも実行できます: `python main.py` → "
        "`python train_model.py` → `python simulate.py` → `python sync_data.py upload`"
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
if "APP_PASSWORD" not in st.secrets or not st.secrets["APP_PASSWORD"]:
    st.title("🏇 競馬AI レース予測")
    st.error(
        "管理者がパスワードを設定していません。"
        "`.streamlit/secrets.toml`（ローカル）または Streamlit Cloud の Secrets 設定で "
        "`APP_PASSWORD` を設定してください。"
    )
    st.stop()

role = check_password()
if role is None:
    st.stop()

sync_status = startup_cloud_sync()

st.title("🏇 競馬AI レース予測")

if role == "admin":
    st.caption(f"👑 管理者としてログイン中｜{sync_status}")
    tab_predict, tab_admin = st.tabs(["🎯 レース予測", "🛠 管理者パネル"])
    with tab_predict:
        render_predict()
    with tab_admin:
        render_admin()
else:
    render_predict()
