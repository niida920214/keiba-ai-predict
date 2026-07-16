"""
streamlit_app.py -- 競馬AI レース予測（公開版）
==================================================
学習済みモデルを使ってレースIDから予測を行う、予測機能だけを切り出した公開用UI。
データ更新・モデル再学習など重い処理は含まない（それらはローカル運用専用）。
"""

import hmac
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

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
# 簡易パスワード認証（限られた人だけに公開するためのゲート）
# ---------------------------------------------------------------------------
def check_password() -> bool:
    """st.secrets['APP_PASSWORD'] と一致するパスワードが入力されるまで先に進ませない。"""

    if st.session_state.get("authenticated"):
        return True

    def password_entered():
        entered = st.session_state.get("password_input", "")
        correct = st.secrets.get("APP_PASSWORD", "")
        if correct and hmac.compare_digest(entered, correct):
            st.session_state["authenticated"] = True
            st.session_state["password_input"] = ""
        else:
            st.session_state["authenticated"] = False

    st.title("🏇 競馬AI レース予測")
    st.text_input(
        "パスワード", type="password", key="password_input", on_change=password_entered
    )
    if st.session_state.get("authenticated") is False:
        st.error("パスワードが違います。")
    return False


if "APP_PASSWORD" not in st.secrets or not st.secrets["APP_PASSWORD"]:
    st.title("🏇 競馬AI レース予測")
    st.error(
        "管理者がパスワードを設定していません。"
        "`.streamlit/secrets.toml`（ローカル）または Streamlit Cloud の Secrets 設定で "
        "`APP_PASSWORD` を設定してください。"
    )
    st.stop()

if not check_password():
    st.stop()

st.title("🏇 競馬AI レース予測")
st.caption(
    "netkeibaの出馬表URL末尾にある12桁のレースIDを入力してください。"
    "例: `race_id=202605010811` -> `202605010811`"
)
st.info(
    "本ツールは個人の研究目的で作成した予測モデルです。的中や利益を保証するものではありません。",
    icon="ℹ️",
)

MIN_INTERVAL_SEC = 15  # 連打による外部サイトへの過剰アクセスを防ぐための簡易クールダウン

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
