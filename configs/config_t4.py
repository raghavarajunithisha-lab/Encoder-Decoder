# config_gru.py
# Configuration for GRU seq2seq model
MODEL_NAME = "T4"
USE_TDA = True

# data
CSV_PATH = "data/preprocessed_mental_health_chatbot.csv"
INPUT_COL = "human_input"
OUTPUT_COL = "assistant_output"
MAX_EXAMPLES = None
EARLY_STOPPING_PATIENCE = 3

# model
EMBED_SIZE = 256
DROPOUT = 0.1
NUM_HEADS = 8         
FF_DIM = 1024
NUM_LAYERS = 4
MAX_OUTPUT_LEN = 50

# training
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
MAX_OUTPUT_LEN = 100
CLIP_GRAD = 1.0
SEED = 42
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 1

# TDA-specific
FASTTEXT_DIM = 50
LANDSCAPE_RESOLUTION = 100
TDA_PCA_COMPONENTS = 250