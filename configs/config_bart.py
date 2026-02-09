# configs/config_lstm.py
# Configuration for LSTM seq2seq model

MODEL_NAME = "LSTM"
USE_TDA = True  # Set True to enable TDA features

# data
CSV_PATH = "data/preprocessed_mental_health_chatbot.csv"
INPUT_COL = "human_input"
OUTPUT_COL = "assistant_output"
MAX_EXAMPLES = 100

# model
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.2
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 1

# TDA-specific
FASTTEXT_DIM = 50
LANDSCAPE_RESOLUTION = 100
TDA_PCA_COMPONENTS = 60

# training
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
MAX_OUTPUT_LEN = 100
SEED = 42
