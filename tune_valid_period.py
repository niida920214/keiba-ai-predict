"""
tune_valid_period.py – ウォークフォワード検証による最適valid期間の探索 (#43対応)
================================================================================
直近1年間(2023年を想定)をQ1〜Q4の4つのFoldに分割し、
それぞれのテスト期間に対して「1ヶ月, 2ヶ月, 3ヶ月, 6ヶ月」のvalid期間を設定し、
さらにその直前5年間をtrain期間としてLightGBMモデルを学習・評価する。

評価指標:
1. 平均LogLoss / AUC
2. 変動係数 (CV: Coefficient of Variation)
3. 1SEルール (One Standard Error Rule) に基づく最適期間の決定
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

from modules.constants import local_paths
from modules.constants.features import EXCLUDE_COLS


def get_walk_forward_splits(df: pd.DataFrame, target_year: int = 2023, train_years: int = 5):
    """
    指定年から4つの四半期(Q1~Q4)のテスト期間を作成し、
    それに合わせた各valid期間(1,2,3,6ヶ月)と固定train期間(5年)の分割インデックスを返す。
    """
    # データを日付順にソート（必須）
    df = df.sort_values("date")
    
    # テスト期間 (四半期ごと)
    quarters = [
        (f"{target_year}-01-01", f"{target_year}-03-31"),
        (f"{target_year}-04-01", f"{target_year}-06-30"),
        (f"{target_year}-07-01", f"{target_year}-09-30"),
        (f"{target_year}-10-01", f"{target_year}-12-31"),
    ]
    
    valid_months_list = [1, 2, 3, 6]
    splits = []
    
    for q_idx, (test_start_str, test_end_str) in enumerate(quarters):
        test_start = pd.to_datetime(test_start_str)
        test_end = pd.to_datetime(test_end_str)
        
        # テストデータのインデックス
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)
        test_idx = df[test_mask].index
        
        # もしこの四半期のデータが存在しなければスキップ
        if len(test_idx) == 0:
            print(f"[WARN] Fold {q_idx+1} ({test_start_str} - {test_end_str}) にデータが存在しません。スキップします。")
            continue
            
        fold_splits = {}
        for v_months in valid_months_list:
            # valid期間の終了日はテスト開始日の前日
            valid_end = test_start - pd.Timedelta(days=1)
            # valid期間の開始日は valid_end から v_months ヶ月前
            # 厳密なカレンダー計算ではなく、ざっくり 30.4 * v_months 日とする
            valid_start = valid_end - pd.Timedelta(days=int(30.4 * v_months))
            
            # train期間の終了日は valid開始日の前日
            train_end = valid_start - pd.Timedelta(days=1)
            # train期間の開始日は train_end から train_years 年前
            train_start = train_end - pd.Timedelta(days=int(365.25 * train_years))
            
            train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
            valid_mask = (df["date"] >= valid_start) & (df["date"] <= valid_end)
            
            fold_splits[v_months] = {
                "train": df[train_mask].index,
                "valid": df[valid_mask].index,
                "test": test_idx
            }
        
        splits.append({
            "fold_name": f"Q{q_idx+1} ({test_start_str}~)",
            "test_start": test_start_str,
            "test_end": test_end_str,
            "splits": fold_splits
        })
        
    return splits



def prepare_xy(df: pd.DataFrame, indices):
    subset = df.loc[indices]
    cols_to_drop = [c for c in EXCLUDE_COLS if c in subset.columns]
    X = subset.drop(columns=cols_to_drop)
    y = subset["rank"]
    return X, y


def main():
    print("="*60)
    print(" ウォークフォワード検証: 最適Valid期間の探索")
    print("="*60)
    
    # 1. データ読み込み
    data_c_path = local_paths.PROCESSED_DIR / "data_c.pickle"
    if not data_c_path.exists():
        print(f"[ERROR] {data_c_path} が見つかりません。")
        return
        
    print("[1] データを読み込み中...")
    with open(data_c_path, "rb") as f:
        df = pickle.load(f)
        
    # 日付型への変換がまだなら行う
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
        
    # 2. ウォークフォワード分割の作成
    # 学習対象の年として、データ内に存在する最新の完全な年（例：2024または2025）を使うのが一般的。
    # ここでは、最新データから1年前の年を基準とする（2025年なら2024年をテスト対象に）。
    latest_year = df["date"].dt.year.max()
    target_year = latest_year - 1 if latest_year > 2019 else 2023
    
    print(f"[2] データ分割中 (Target Test Year: {target_year})...")
    wf_splits = get_walk_forward_splits(df, target_year=target_year)
    
    if not wf_splits:
        print("[ERROR] 指定した年のデータが存在しないため、分割できませんでした。")
        return

    # モデリングの固定パラメータ (計算時間短縮のため探索はせず固定)
    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 1000,
    }

    # 各valid期間の設定ごとにスコアを記録
    results_by_valid_period = {1: [], 2: [], 3: [], 6: []}

    print("\n[3] 各Fold・Valid期間での学習と評価を開始します...")
    for fold_info in wf_splits:
        fold_name = fold_info["fold_name"]
        print(f"\n--- Fold: {fold_name} ---")
        
        for v_months, splits in fold_info["splits"].items():
            X_train, y_train = prepare_xy(df, splits["train"])
            X_valid, y_valid = prepare_xy(df, splits["valid"])
            X_test, y_test = prepare_xy(df, splits["test"])
            
            # データ数が極端に少ない場合はスキップ
            if len(X_train) < 100 or len(X_valid) < 10 or len(X_test) < 10:
                print(f"  [{v_months}ヶ月] Data too small. skipping.")
                continue

            model = LGBMClassifier(**base_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            test_logloss = log_loss(y_test, y_pred_proba)
            
            results_by_valid_period[v_months].append(test_logloss)
            print(f"  Valid {v_months}ヶ月 -> Test LogLoss: {test_logloss:.4f} (Best Iter: {model.best_iteration_})")

    # 4. 指標の集計と1SEルールの適用
    print("\n" + "="*60)
    print(" [結果まとめ] 最適Valid期間の評価")
    print("="*60)
    
    summary = []
    for v_months, scores in results_by_valid_period.items():
        if not scores:
            summary.append((v_months, np.nan, np.nan, np.nan))
            continue
            
        mean_loss = np.mean(scores)
        std_loss = np.std(scores, ddof=1) # 標本標準偏差
        cv = std_loss / mean_loss if mean_loss > 0 else np.nan # 変動係数
        se = std_loss / np.sqrt(len(scores)) # 標準誤差
        
        summary.append({
            "valid_months": v_months,
            "mean_logloss": mean_loss,
            "std": std_loss,
            "cv": cv,
            "se": se,
            "1se_upper_bound": mean_loss + se
        })
        
    summary_df = pd.DataFrame(summary).dropna()
    if summary_df.empty:
        print("評価データが不足しています。")
        return
        
    print(summary_df.to_markdown(index=False))
    
    # 最良スコアを持つ設定を特定
    best_setting = summary_df.loc[summary_df["mean_logloss"].idxmin()]
    best_months = best_setting["valid_months"]
    best_loss = best_setting["mean_logloss"]
    se_threshold = best_setting["1se_upper_bound"]
    
    print(f"\n>> 純粋な最高精度: Valid期間 {best_months} ヶ月 (LogLoss: {best_loss:.4f})")
    print(f">> 1SEルールの許容上限 (Best + SE): {se_threshold:.4f}")
    
    # 1SEルールの適用: SE上限以下の中で、最もvalid期間が短い(＝trainが長い)ものを選ぶ
    candidates = summary_df[summary_df["mean_logloss"] <= se_threshold]
    optimal_setting = candidates.loc[candidates["valid_months"].idxmin()]
    optimal_months = optimal_setting["valid_months"]
    
    print("\n★★★ 最終結論 ★★★")
    print(f"1SEルールを適用した最適設定: 【 Valid期間 {optimal_months} ヶ月 】")
    print("理由: 最高精度モデルの1標準誤差の範囲内に収まりつつ、モデルへの学習データを最大限確保(過学習制御とデータ量の最適バランス)できるため。")
    print("============================================================")

if __name__ == "__main__":
    main()
