from pathlib import Path

# プロジェクトルートの絶対パス（__file__ ベースで確定させる）
# local_paths.py は modules/constants/ に配置 → 3 階層上がプロジェクトルート
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# dataディレクトリ
DATA_DIR = BASE_DIR / "data"

# HTMLディレクトリ
HTML_DIR = DATA_DIR / "html"
HTML_RACE_DIR = HTML_DIR / "race"
HTML_HORSE_DIR = HTML_DIR / "horse"
HTML_PED_DIR = HTML_DIR / "ped"

# Rawデータディレクトリ
RAW_DIR = DATA_DIR / "raw"
RAW_RESULTS_DIR = RAW_DIR / "results"
RAW_HORSE_RESULTS_DIR = RAW_DIR / "horse_results"
RAW_PEDS_DIR = RAW_DIR / "peds"
RAW_RETURN_TABLES_DIR = RAW_DIR / "return_tables"
RAW_RACE_INFO_DIR = RAW_DIR / "race_info"

# Processedデータディレクトリ
PROCESSED_DIR = DATA_DIR / "processed"

# Modelディレクトリ
MODEL_DIR = BASE_DIR / "model"

# Resultsディレクトリ
RESULTS_DIR = BASE_DIR / "results"
