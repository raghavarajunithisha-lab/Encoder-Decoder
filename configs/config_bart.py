MODEL_NAME = "BART"
USE_TDA = False

# data
CSV_PATH = "data/preprocessed_mental_health_chatbot.csv"
INPUT_COL = "human_input"
OUTPUT_COL = "assistant_output"
MAX_EXAMPLES = None

# training
BART_MODEL_NAME = "facebook/bart-base"
OUTPUT_DIR = "./outputs/bart"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128
TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
SEED = 42
LOGGING_STEPS = 100

# TDA options (only used when USE_TDA=True)
FASTTEXT_DIM = 50
LANDSCAPE_RESOLUTION = 50
TDA_PCA_COMPONENTS = 250