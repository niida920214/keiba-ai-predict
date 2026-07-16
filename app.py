"""
app.py -- 競馬AI コントロールパネル (Streamlit)
==================================================
main.py / train_model.py / simulate.py / predict.py をボタンひとつで実行できる
ローカルWeb UI。各処理はサブプロセスとして起動し、ログをリアルタイム表示する。

起動方法:
    .venv/Scripts/streamlit run app.py
"""

import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.constants import local_paths

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable  # このアプリを動かしているvenvのpython.exe


# ---------------------------------------------------------------------------
# バックグラウンドジョブ管理
# ---------------------------------------------------------------------------
class Job:
    """サブプロセスを非同期実行し、ログを蓄積するラッパー。"""

    def __init__(self, name: str):
        self.name = name
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self.q: "queue.Queue[str]" = queue.Queue()
        self.status = "idle"  # idle / running / done / error / stopped
        self.returncode: int | None = None
        self._stop_requested = False

    def start(self, cmd: list[str]):
        self.lines = []
        self.status = "running"
        self.returncode = None
        self._stop_requested = False
        self._final_rerun_done = False
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.q.put(line)
        self.proc.wait()
        if self._stop_requested:
            self.status = "stopped"
        elif self.proc.returncode == 0:
            self.status = "done"
        else:
            self.status = "error"
        self.returncode = self.proc.returncode

    def drain(self):
        while True:
            try:
                self.lines.append(self.q.get_nowait())
            except queue.Empty:
                break

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self._stop_requested = True
            self.proc.terminate()

    def is_running(self) -> bool:
        return self.status == "running"


def get_job(key: str) -> Job:
    jobs = st.session_state.setdefault("jobs", {})
    if key not in jobs:
        jobs[key] = Job(key)
    return jobs[key]


@st.fragment(run_every=1.0)
def render_log(job: Job, height: int = 320):
    job.drain()
    text = "".join(job.lines[-1000:]) or "(ログはまだありません)"
    st.code(text, language=None, height=height)
    if job.status == "running":
        st.info("実行中…")
    elif job.status == "done":
        st.success(f"完了しました (exit code {job.returncode})")
    elif job.status == "error":
        st.error(f"エラーで終了しました (exit code {job.returncode})")
    elif job.status == "stopped":
        st.warning("ユーザーにより中断されました")

    # ジョブ終了直後に一度だけ全体を再描画し、結果テーブル等を表示させる
    if job.status in ("done", "error", "stopped") and not getattr(job, "_final_rerun_done", True):
        job._final_rerun_done = True
        st.rerun(scope="app")


def start_stop_buttons(job: Job, cmd: list[str], start_label: str, disabled: bool = False):
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(start_label, type="primary", disabled=job.is_running() or disabled,
                     key=f"start_{job.name}"):
            job.start(cmd)
            st.rerun()
    with col2:
        if st.button("中断", disabled=not job.is_running(), key=f"stop_{job.name}"):
            job.stop()
            st.rerun()


def file_info(path: Path) -> str:
    if not path.exists():
        return "未生成"
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    size_kb = path.stat().st_size / 1024
    return f"{mtime}  ({size_kb:.0f} KB)"


# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(page_title="競馬AI コントロールパネル", page_icon="🏇", layout="wide")
st.title("🏇 競馬AI コントロールパネル")

PAGES = ["🏠 ダッシュボード", "🔄 データ更新", "🧠 モデル学習", "📊 シミュレーション", "🎯 レース予測"]
page = st.sidebar.radio("メニュー", PAGES)

st.sidebar.markdown("---")
st.sidebar.caption(f"Python: {PYTHON}")
st.sidebar.caption("運用手順の詳細は weekend_routine.md を参照")


# ---------------------------------------------------------------------------
# 🏠 ダッシュボード
# ---------------------------------------------------------------------------
if page == "🏠 ダッシュボード":
    st.subheader("現在のデータ・モデル状況")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**学習用データ**")
        st.write("data_c.pickle:", file_info(local_paths.PROCESSED_DIR / "data_c.pickle"))
        st.write("race_id_list.pickle:", file_info(local_paths.PROCESSED_DIR / "race_id_list.pickle"))
        st.write("race_results.pickle:", file_info(local_paths.RAW_RESULTS_DIR / "race_results.pickle"))

    with col2:
        st.markdown("**モデル**")
        st.write("lgbm_model.pickle:", file_info(local_paths.MODEL_DIR / "lgbm_model.pickle"))
        st.write("lgbm_ranker.pickle:", file_info(local_paths.MODEL_DIR / "lgbm_ranker.pickle"))
        st.write("lgbm_model_win.pickle:", file_info(local_paths.MODEL_DIR / "lgbm_model_win.pickle"))

    with col3:
        st.markdown("**シミュレーション結果**")
        st.write("simulation_summary.csv:", file_info(local_paths.RESULTS_DIR / "simulation_summary.csv"))
        pred_dir = local_paths.RESULTS_DIR / "predictions"
        n_preds = len(list(pred_dir.glob("prediction_*.csv"))) if pred_dir.exists() else 0
        st.write("保存済み予測ファイル数:", n_preds)

    st.markdown("---")
    st.markdown(
        "**推奨される運用の流れ**\n"
        "1. 🔄 データ更新 -- 週末ごとに最新のレース結果を取り込む\n"
        "2. 🧠 モデル学習 -- 月1回程度、最新データでモデルを再学習\n"
        "3. 📊 シミュレーション -- 学習後に回収率を確認（任意）\n"
        "4. 🎯 レース予測 -- 次のレースのレースIDを指定して予測"
    )


# ---------------------------------------------------------------------------
# 🔄 データ更新 (main.py)
# ---------------------------------------------------------------------------
elif page == "🔄 データ更新":
    st.subheader("データ更新（スクレイピング & 前処理）")
    st.caption("main.py を実行します。既に取得済みのレースはスキップされる差分更新です。")

    mode = st.radio(
        "実行モード",
        ["通常（差分更新）", "--clean（過去成績データを再取得）", "--clean-all（全データを削除して完全再取得）"],
        index=0,
    )

    confirm = True
    if mode != "通常（差分更新）":
        st.warning("このモードは既存データを削除します。元に戻せません。")
        confirm = st.checkbox("削除して再取得することを理解しました")

    cmd = [PYTHON, "main.py"]
    if mode.startswith("--clean-all"):
        cmd.append("--clean-all")
    elif mode.startswith("--clean"):
        cmd.append("--clean")

    job = get_job("main")
    start_stop_buttons(job, cmd, "▶ データ更新を実行", disabled=not confirm)
    render_log(job)


# ---------------------------------------------------------------------------
# 🧠 モデル学習 (train_model.py)
# ---------------------------------------------------------------------------
elif page == "🧠 モデル学習":
    st.subheader("モデル再学習")
    st.caption(
        "train_model.py を実行します。Optunaによるハイパーパラメータ探索を含むため、"
        "環境によっては数十分〜数時間かかります。"
    )

    data_path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not data_path.exists():
        st.error("data_c.pickle がありません。先に「データ更新」を実行してください。")
    else:
        st.info(f"学習に使用するデータ: {file_info(data_path)}")

    job = get_job("train")
    start_stop_buttons(job, [PYTHON, "train_model.py"], "▶ モデル学習を実行",
                        disabled=not data_path.exists())
    render_log(job, height=420)


# ---------------------------------------------------------------------------
# 📊 シミュレーション (simulate.py)
# ---------------------------------------------------------------------------
elif page == "📊 シミュレーション":
    st.subheader("回収率シミュレーション")
    st.caption("simulate.py を実行します。学習済みモデルの回収率・シャープレシオ等を評価します。")

    model_path = local_paths.MODEL_DIR / "lgbm_model.pickle"
    if not model_path.exists():
        st.error("学習済みモデルがありません。先に「モデル学習」を実行してください。")

    job = get_job("simulate")
    start_stop_buttons(job, [PYTHON, "simulate.py"], "▶ シミュレーションを実行",
                        disabled=not model_path.exists())
    render_log(job)

    summary_path = local_paths.RESULTS_DIR / "simulation_summary.csv"
    if not job.is_running() and summary_path.exists():
        st.markdown("---")
        st.subheader("結果")
        st.caption(f"生成日時: {file_info(summary_path)}")
        try:
            df_summary = pd.read_csv(summary_path)
            if not df_summary.empty:
                st.dataframe(df_summary)
        except pd.errors.EmptyDataError:
            st.info("simulation_summary.csv は空です。")

        image_names = [
            "tansho_return_rate.png",
            "return_rate_vs_nbets.png",
            "sharpe_ratio.png",
            "umaren_box_return_rate.png",
            "ev_tansho_return_rate.png",
            "ev_return_rate_vs_nbets.png",
            "harville_beta_optimization.png",
            "kelly_vs_flat_return.png",
        ]
        cols = st.columns(2)
        for i, name in enumerate(image_names):
            img_path = local_paths.RESULTS_DIR / name
            if img_path.exists():
                with cols[i % 2]:
                    st.image(str(img_path), caption=name)


# ---------------------------------------------------------------------------
# 🎯 レース予測 (predict.py)
# ---------------------------------------------------------------------------
elif page == "🎯 レース予測":
    st.subheader("レース予測")
    st.caption(
        "netkeibaの出馬表URL末尾にある12桁のレースIDを入力してください。"
        "例: race_id=202605010811 -> 202605010811"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        race_id = st.text_input("レースID（12桁）", max_chars=12, placeholder="202605010811")
    with col2:
        race_date = st.date_input("開催日", value=datetime.now())

    valid_race_id = bool(race_id) and race_id.isdigit() and len(race_id) == 12
    if race_id and not valid_race_id:
        st.error("レースIDは半角数字12桁で入力してください。")

    ranker_or_model_exists = (local_paths.MODEL_DIR / "lgbm_model.pickle").exists() or \
        (local_paths.MODEL_DIR / "lgbm_ranker.pickle").exists()
    if not ranker_or_model_exists:
        st.error("学習済みモデルがありません。先に「モデル学習」を実行してください。")

    date_str = race_date.strftime("%Y/%m/%d")
    cmd = [PYTHON, "predict.py", race_id or "", "--date", date_str]

    job = get_job("predict")
    start_stop_buttons(job, cmd, "▶ 予測を実行",
                        disabled=not (valid_race_id and ranker_or_model_exists))
    render_log(job, height=420)

    if valid_race_id and not job.is_running():
        res_path = local_paths.RESULTS_DIR / "predictions" / f"prediction_{race_id}.csv"
        combo_path = local_paths.RESULTS_DIR / "predictions" / f"prediction_{race_id}_combos.csv"

        if res_path.exists():
            st.markdown("---")
            st.subheader("予測結果")
            st.caption(f"生成日時: {file_info(res_path)}")
            df_res = pd.read_csv(res_path, encoding="utf-8-sig")
            st.markdown("**各馬の予測**")
            st.dataframe(df_res, width="stretch")

            if "EV" in df_res.columns and "Kelly" in df_res.columns:
                bets = df_res[(df_res["EV"] > 1.0) & (df_res["Kelly"] > 0)]
                if not bets.empty:
                    st.markdown("**推奨買い目（単勝 EV > 1.0 かつ Kelly > 0）**")
                    st.dataframe(bets, width="stretch")

        if combo_path.exists():
            df_combo = pd.read_csv(combo_path, encoding="utf-8-sig")
            st.markdown("**馬連 / ワイド / 三連複 の上位候補**")
            st.dataframe(df_combo, width="stretch")
