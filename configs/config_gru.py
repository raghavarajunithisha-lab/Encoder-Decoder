# config_gru.py
MODEL_NAME = "GRU"
USE_TDA = False   

# ------------------ Data ------------------
CSV_PATH = "data/preprocessed_counselchat_data_df.csv"
INPUT_COL = "questionText"
OUTPUT_COL = "answerText"
MAX_EXAMPLES = None

# ------------------ Model ------------------
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 1

# ------------------ Training ------------------
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
MAX_OUTPUT_LEN = 100
CLIP_GRAD = 1.0
SEED = 42

# ------------------ TDA OPTIONS (Only if USE_TDA=True) ------------------
FASTTEXT_DIM = 50
LANDSCAPE_RESOLUTION = 100
TDA_PCA_COMPONENTS = 200
