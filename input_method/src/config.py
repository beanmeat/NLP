from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

ROW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
LOGS_DIR = ROOT_DIR / 'data' / 'logs'
MODELS_DIR = ROOT_DIR / 'data' / 'models'


SEQ_LEN = 5