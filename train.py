# train.py
# Unified training entrypoint that chooses model-specific training functions.

import os
import random
import numpy as np
import torch

from utils.data_utils import load_csv, SimpleVocab
from torch.utils.data import DataLoader  
from utils.tda_utils import train_fasttext, diagrams_to_landscape_vectors, sentence_diagram_from_embeddings

# model modules
from models import bart_model, lstm_model, gru_model, t4_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_vocabs_from_texts(texts_src, texts_tgt, max_size, min_freq):
    src_vocab = SimpleVocab(max_size=max_size, min_freq=min_freq)
    tgt_vocab = SimpleVocab(max_size=max_size, min_freq=min_freq)
    for s in texts_src:
        for tok in str(s).strip().split():
            src_vocab.add_token(tok)
    for t in texts_tgt:
        for tok in str(t).strip().split():
            tgt_vocab.add_token(tok)
    src_vocab.build(); tgt_vocab.build()
    return src_vocab, tgt_vocab

def prepare_and_train(cfg):
    set_seed(getattr(cfg, "SEED", 42))
    df = load_csv(cfg.CSV_PATH, cfg.INPUT_COL, cfg.OUTPUT_COL, max_examples=getattr(cfg, "MAX_EXAMPLES", None))
    n = len(df)
    val_n = max(1, int(0.1 * n))
    train_df = df[:-val_n].reset_index(drop=True)
    val_df = df[-val_n:].reset_index(drop=True)
    print(f"Loaded {n} examples. Train {len(train_df)} Val {len(val_df)}")

    # Optionally compute TDA features (shared pipeline): train FastText and compute landscape PCA features
    tda_train = None; tda_val = None
    if getattr(cfg, "USE_TDA", False):
        print("Computing TDA features (this may be slow)...")
        tokenized = [s.strip().split() for s in train_df[cfg.INPUT_COL].astype(str).tolist() + val_df[cfg.INPUT_COL].astype(str).tolist()]
        ft = train_fasttext(tokenized, vec_size=getattr(cfg, "FASTTEXT_DIM", getattr(cfg, "FASTTEXT_VEC_SIZE", 100)))
        # compute per-sentence persistence diagrams of token embeddings (2D projection)
        all_embeddings = []
        counts = []
        for s in train_df[cfg.INPUT_COL].tolist() + val_df[cfg.INPUT_COL].tolist():
            tokens = s.strip().split()
            vecs = []
            for tok in tokens:
                try:
                    vecs.append(ft.wv[tok])
                except Exception:
                    vecs.append(np.zeros(ft.vector_size))
            if len(vecs) == 0:
                vecs = np.zeros((1, ft.vector_size))
            vecs = np.array(vecs)
            all_embeddings.append(vecs)
            counts.append(vecs.shape[0])
        # convert each point-cloud to persistence diag then to landscape in a simplified way
        diagrams = []
        for emb in all_embeddings:
            diag = sentence_diagram_from_embeddings(emb)
            diagrams.append(diag)
        tda_feats = diagrams_to_landscape_vectors(diagrams, resolution=getattr(cfg, "LANDSCAPE_RESOLUTION", getattr(cfg, "TDA_LANDSCAPE_RESOLUTION", 200)), pca_dim=getattr(cfg, "TDA_PCA_COMPONENTS", getattr(cfg, "TDA_PCA_DIM", 10)))
        tda_train = tda_feats[:len(train_df)]
        tda_val = tda_feats[len(train_df):]

    # Prepare model-specific data and call training
    model_name = getattr(cfg, "MODEL_NAME", None)
    if model_name == "BART":
        tokenizer, tok_train, tok_test, data_collator = bart_model.prepare_tokenized_datasets(df, cfg)
        model = bart_model.build_model(cfg)
        trainer = bart_model.get_trainer(model, tokenizer, tok_train, tok_test, data_collator, cfg)
        trainer.train()
        trainer.save_model(os.path.join(cfg.OUTPUT_DIR, "final_model"))
        tokenizer.save_pretrained(os.path.join(cfg.OUTPUT_DIR, "final_model"))
        print("BART training complete.")
    elif model_name == "LSTM":
        src_vocab, tgt_vocab = build_vocabs_from_texts(train_df[cfg.INPUT_COL].tolist(), train_df[cfg.OUTPUT_COL].tolist(), cfg.MAX_VOCAB_SIZE, cfg.MIN_FREQ)
        from models.lstm_model import ChatDataset, collate_fn, train_lstm
        train_ds = ChatDataset(train_df[cfg.INPUT_COL].tolist(), train_df[cfg.OUTPUT_COL].tolist(), src_vocab, tgt_vocab, tda_train)
        val_ds = ChatDataset(val_df[cfg.INPUT_COL].tolist(), val_df[cfg.OUTPUT_COL].tolist(), src_vocab, tgt_vocab, tda_val)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        best = train_lstm(cfg, train_loader, val_loader, src_vocab, tgt_vocab, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("LSTM finished. Best BLEU:", best)
    elif model_name == "GRU":
        src_vocab, tgt_vocab = build_vocabs_from_texts(train_df[cfg.INPUT_COL].tolist(), train_df[cfg.OUTPUT_COL].tolist(), cfg.MAX_VOCAB_SIZE, cfg.MIN_FREQ)
        from models.gru_model import ChatDatasetPlain, collate_fn_plain, train_gru
        train_ds = ChatDatasetPlain(train_df[cfg.INPUT_COL].tolist(), train_df[cfg.OUTPUT_COL].tolist(), src_vocab, tgt_vocab)
        val_ds = ChatDatasetPlain(val_df[cfg.INPUT_COL].tolist(), val_df[cfg.OUTPUT_COL].tolist(), src_vocab, tgt_vocab)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_plain)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_plain)
        best = train_gru(cfg, train_loader, val_loader, src_vocab, tgt_vocab, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("GRU finished. Best BLEU:", best)
    elif model_name == "T4":
        src_vocab = None; tgt_vocab = None
        # build token lists
        src_token_lists = [s.strip().split() for s in train_df[cfg.INPUT_COL].tolist()]
        tgt_token_lists = [s.strip().split() for s in train_df[cfg.OUTPUT_COL].tolist()]
        from models.t4_model import Vocab, ChatDataset, collate_fn, train_t4
        src_vocab = Vocab(src_token_lists, min_freq=cfg.MIN_FREQ, max_size=cfg.MAX_VOCAB_SIZE)
        tgt_vocab = Vocab(tgt_token_lists, min_freq=cfg.MIN_FREQ, max_size=cfg.MAX_VOCAB_SIZE)
        train_ds = ChatDataset(train_df[cfg.INPUT_COL].tolist(), train_df[cfg.OUTPUT_COL].tolist(), src_vocab, tgt_vocab, tda_train)
        val_ds = ChatDataset(val_df[cfg.INPUT_COL].tolist(), val_df[cfg.OUTPUT_COL].tolist(), src_vocab, tgt_vocab, tda_val)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        best = train_t4(cfg, train_loader, val_loader, src_vocab, tgt_vocab, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("T4 finished. Best BLEU:", best)
    else:
        raise ValueError(f"Unknown model {model_name}")


