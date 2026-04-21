from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PKL_DIR = DATA_DIR / "pkls"
CSV_DIR = DATA_DIR / "csv"
MODEL_DIR = DATA_DIR / "models"
ENCODER_DIR = DATA_DIR / "encoders"

for d in [DATA_DIR, UPLOAD_DIR, PKL_DIR, CSV_DIR, MODEL_DIR, ENCODER_DIR]:
    d.mkdir(parents=True, exist_ok=True)