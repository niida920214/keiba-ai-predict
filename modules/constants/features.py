"""
features.py – 特徴量定義の一元管理
===================================
訓練・予測・シミュレーションで共通して使用する
特徴量の除外リスト等を定義する。
"""

# 訓練時に除外するカラム（目的変数・メタデータ・Benter式で排除するオッズ特徴量）
EXCLUDE_COLS = [
    "rank",
    "rank_win",
    "date",
    "単勝",
    "log_odds",
    "odds_rank",
    "finishing_position",
    "time_sec",
]

# 予測時に除外するカラム
PREDICT_DROP_COLS = [
    "date",
    "target",
    "finishing_position",
    "time_sec",
]
