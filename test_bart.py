import sys
import pandas as pd
import numpy as np

# Mock config
class Config:
    CSV_PATH = "data/preprocessed_mental_health_chatbot.csv"
    INPUT_COL = "human_input"
    OUTPUT_COL = "assistant_output"
    MAX_EXAMPLES = 50
    SEED = 42
    USE_TDA = False
    MODEL_NAME = "BART"
    BART_MODEL_NAME = "facebook/bart-base"
    MAX_INPUT_LEN = 128
    MAX_TARGET_LEN = 128
    BATCH_SIZE = 4
    TRAIN_EPOCHS = 1
    LEARNING_RATE = 5e-5
    OUTPUT_DIR = "outputs/bart_test"

cfg = Config()

from train import prepare_and_train
try:
    prepare_and_train(cfg)
except Exception as e:
    import traceback
    traceback.print_exc()
