from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYER = 2
NUM_DECODER_LAYER = 2

SEQ_LEN = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20
