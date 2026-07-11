from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

ROW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'


SEQ_LEN = 5
BATCH_SIZE = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
EPOCHS = 10