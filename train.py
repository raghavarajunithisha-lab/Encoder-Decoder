# train.py
# Unified training entrypoint that chooses model-specific training functions.

import os
import random
import numpy as np
import torch

from utils.data_utils import load_csv, SimpleVocab
from torch.utils.data import DataLoader
from utils.tda_utils import (
    train_fasttext,
    diagrams_to_landscape_vectors,
    sentence_diagram_from_embeddings,
    pad_or_truncate,
)

# model modules
from models import bart_model, lstm_model, gru_model, t4_model

from models.t4_model import Vocab, ChatDataset, collate_fn, train_t4
from torch.utils.data import DataLoader
import torch

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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

def _ensure_tda_matrix(tda_like, target_dim):
    """
    Ensure `tda_like` becomes a numeric 2D numpy array of shape (N, target_dim).
    Accepts:
      - numpy array (N, D) possibly with dtype=object or ragged rows
      - list of 1D arrays/lists
    Returns: np.ndarray dtype float32
    """
    if tda_like is None:
        return None

    # If caller mistakenly passed a (tda, pca, scaler) tuple, handle it:
    if isinstance(tda_like, tuple) or (isinstance(tda_like, list) and len(tda_like) == 3 and \
       (hasattr(tda_like[1], 'transform') or hasattr(tda_like[2], 'transform'))):
        # First element should be the numeric array
        tda_like = tda_like[0]

    # If it's already a numpy array and homogeneous numeric, try direct conversion
    if isinstance(tda_like, np.ndarray) and np.issubdtype(tda_like.dtype, np.number):
        # If second dim matches target_dim accept, otherwise pad/truncate each row
        if tda_like.ndim == 2 and tda_like.shape[1] == target_dim:
            return tda_like.astype(np.float32)
        # else fall through to row-wise handling

    # Row-wise handling: pad/truncate each vector to target_dim, then stack
    rows = []
    # If it's a numpy 1D of arrays (object), iterate
    if isinstance(tda_like, np.ndarray):
        tda_iter = list(tda_like)
    elif isinstance(tda_like, list):
        tda_iter = tda_like
    else:
        # Unexpected type — try to coerce
        tda_iter = list(tda_like)

    for v in tda_iter:
        try:
            rows.append(pad_or_truncate(v, target_dim))
        except Exception:
            # as a fallback, append zeros
            rows.append(np.zeros(target_dim, dtype=np.float32))

    if len(rows) == 0:
        return np.zeros((0, target_dim), dtype=np.float32)

    mat = np.vstack([np.asarray(r, dtype=np.float32) for r in rows])
    return mat.astype(np.float32)

def prepare_and_train(cfg):
    set_seed(getattr(cfg, "SEED", 42))
    # df = load_csv(cfg.CSV_PATH, cfg.INPUT_COL, cfg.OUTPUT_COL, max_examples=cfg.MAX_EXAMPLES)

    df = load_csv(cfg.CSV_PATH, cfg.INPUT_COL, cfg.OUTPUT_COL)

    n = len(df)
    val_n = max(1, int(0.1 * n))
    test_n = max(1, int(0.1 * n))

    train_df = df[:-(val_n + test_n)].reset_index(drop=True)
    val_df = df[-(val_n + test_n):-test_n].reset_index(drop=True)
    test_df = df[-test_n:].reset_index(drop=True)

    print(f"Loaded {n} examples. Train {len(train_df)} Val {len(val_df)} Test {len(test_df)}")

    # ---------- TDA FEATURES ----------
    tda_train = None
    tda_val = None
    tda_test = None

    if getattr(cfg, "USE_TDA", False):
        print("Computing TDA features (this may be slow)...")

        # Train FastText only on training sentences
        tokenized_train = [s.strip().split() for s in train_df[cfg.INPUT_COL].astype(str).tolist()]
        ft = train_fasttext(tokenized_train, vec_size=getattr(cfg, "FASTTEXT_DIM", getattr(cfg, "FASTTEXT_VEC_SIZE", 100)))

        # Build embeddings utility function
        def build_embeddings(data):
            embeddings = []
            for s in data:
                tokens = s.strip().split()
                vecs = []
                for tok in tokens:
                    try:
                        vecs.append(ft.wv[tok])
                    except Exception:
                        vecs.append(np.zeros(ft.vector_size))
                if len(vecs) == 0:
                    vecs = np.zeros((1, ft.vector_size))
                embeddings.append(np.array(vecs))
            return embeddings

        # Train diagrams and PCA on train set
        train_diagrams = [sentence_diagram_from_embeddings(emb) for emb in build_embeddings(train_df[cfg.INPUT_COL].tolist())]

        tda_train_raw, pca_model, scaler = diagrams_to_landscape_vectors(
            train_diagrams,
            resolution=getattr(cfg, "LANDSCAPE_RESOLUTION", getattr(cfg, "TDA_LANDSCAPE_RESOLUTION", 200)),
            pca_dim=getattr(cfg, "TDA_PCA_COMPONENTS", getattr(cfg, "TDA_PCA_DIM", 10))
        )

        # transform val and test using pca_model + scaler
        val_diagrams = [sentence_diagram_from_embeddings(emb) for emb in build_embeddings(val_df[cfg.INPUT_COL].tolist())]
        tda_val_raw, _, _ = diagrams_to_landscape_vectors(
            val_diagrams,
            resolution=getattr(cfg, "LANDSCAPE_RESOLUTION", getattr(cfg, "TDA_LANDSCAPE_RESOLUTION", 200)),
            pca_model=pca_model,
            scaler=scaler
        )

        test_diagrams = [sentence_diagram_from_embeddings(emb) for emb in build_embeddings(test_df[cfg.INPUT_COL].tolist())]
        tda_test_raw, _, _ = diagrams_to_landscape_vectors(
            test_diagrams,
            resolution=getattr(cfg, "LANDSCAPE_RESOLUTION", getattr(cfg, "TDA_LANDSCAPE_RESOLUTION", 200)),
            pca_model=pca_model,
            scaler=scaler
        )

        # Ensure numeric 2D arrays with consistent final dimension = TDA_PCA_COMPONENTS
        tda_dim = getattr(cfg, "TDA_PCA_COMPONENTS", getattr(cfg, "TDA_PCA_DIM", 10))
        tda_train = _ensure_tda_matrix(tda_train_raw, tda_dim)
        tda_val   = _ensure_tda_matrix(tda_val_raw, tda_dim)
        tda_test  = _ensure_tda_matrix(tda_test_raw, tda_dim)

    # ---------- MODEL TRAINING ----------
    model_name = getattr(cfg, "MODEL_NAME", None)

    if model_name == "BART":
        tokenizer, tok_train, data_collator = bart_model.prepare_tokenized_datasets(
            train_df, cfg, tda_array=tda_train if cfg.USE_TDA else None
        )

        _, tok_val, _ = bart_model.prepare_tokenized_datasets(
            val_df, cfg, tda_array=tda_val if cfg.USE_TDA else None
        )

        _, tok_test, _ = bart_model.prepare_tokenized_datasets(
            test_df, cfg, tda_array=tda_test if cfg.USE_TDA else None
        )

        model = bart_model.build_model(cfg)

        trainer = bart_model.get_trainer(
            model, tokenizer, tok_train, tok_val, tok_test, data_collator, cfg
        )

        trainer.train()

        print("\n FINAL TEST EVALUATION")
        trainer.evaluate(eval_dataset=tok_test, metric_key_prefix="test_eval")


    elif model_name == "LSTM":

        src_vocab, tgt_vocab = build_vocabs_from_texts(
            train_df[cfg.INPUT_COL].tolist(),
            train_df[cfg.OUTPUT_COL].tolist(),
            cfg.MAX_VOCAB_SIZE, cfg.MIN_FREQ
        )

        # pass TDA only if USE_TDA=True
        train_ds = ChatDataset(
            train_df[cfg.INPUT_COL].tolist(),
            train_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_train if cfg.USE_TDA else None
        )

        val_ds = ChatDataset(
            val_df[cfg.INPUT_COL].tolist(),
            val_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_val if cfg.USE_TDA else None
        )

        test_ds = ChatDataset(
            test_df[cfg.INPUT_COL].tolist(),
            test_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_test if cfg.USE_TDA else None
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        result = train_lstm(
            cfg,
            train_loader,
            val_loader,
            test_loader,
            src_vocab,
            tgt_vocab,
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        print("\n LSTM TRAINING LOGS\n")

        for row in result["history"]:
            print(f"\nEpoch {row['epoch']}:")
            print({
                "train_loss": row["train_loss"],
                "train_bleu": row["train_bleu"]
            })
            print({
                "val_loss": row["val_loss"],
                "val_bleu": row["val_bleu"]
            })

        print("\n FINAL LSTM RESULT")
        print({
            "test_loss": result["test_loss"],
            "test_bleu": result["test_bleu"]
        })


    elif model_name == "GRU":
        src_vocab, tgt_vocab = build_vocabs_from_texts(
            train_df[cfg.INPUT_COL].tolist(),
            train_df[cfg.OUTPUT_COL].tolist(),
            cfg.MAX_VOCAB_SIZE, cfg.MIN_FREQ
        )

        from models.gru_model import ChatDatasetPlain, collate_fn_plain, train_gru

        # Compute TDA features if requested (we already did above — tda_train/val/test available)
        # build datasets / loaders
        # PASS TDA arrays into the ChatDatasetPlain (only if cfg.USE_TDA is True)
        train_ds = ChatDatasetPlain(
            train_df[cfg.INPUT_COL].tolist(),
            train_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_train if cfg.USE_TDA else None
        )
        val_ds = ChatDatasetPlain(
            val_df[cfg.INPUT_COL].tolist(),
            val_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_val if cfg.USE_TDA else None
        )
        test_ds = ChatDatasetPlain(
            test_df[cfg.INPUT_COL].tolist(),
            test_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_test if cfg.USE_TDA else None
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_plain)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_plain)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_plain)

        # call train_gru with optional tda arrays (train_gru enables decoder TDA only if both cfg.USE_TDA and tda_train present)
        result = train_gru(
          cfg,
          train_loader,
          val_loader,
          test_loader,
          src_vocab,
          tgt_vocab,
          torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        print("\n FINAL TEST EVALUATION (GRU)")
        # print both test_loss and test_bleu as requested
        print({"test_loss": result.get("test_loss"), "test_bleu": result["test_bleu"]})

    elif model_name == "T4":

        # -------------------------
        # TOKEN LISTS (for vocab)
        # -------------------------
        src_token_lists = [
            s.strip().split()
            for s in train_df[cfg.INPUT_COL].tolist()
        ]

        tgt_token_lists = [
            s.strip().split()
            for s in train_df[cfg.OUTPUT_COL].tolist()
        ]

        # -------------------------
        # VOCABS
        # -------------------------
        src_vocab = Vocab(
            src_token_lists,
            min_freq=cfg.MIN_FREQ,
            max_size=cfg.MAX_VOCAB_SIZE
        )

        tgt_vocab = Vocab(
            tgt_token_lists,
            min_freq=cfg.MIN_FREQ,
            max_size=cfg.MAX_VOCAB_SIZE
        )

        # -------------------------
        # DATASETS
        # -------------------------
        train_ds = ChatDataset(
            train_df[cfg.INPUT_COL].tolist(),
            train_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_train if cfg.USE_TDA else None
        )

        val_ds = ChatDataset(
            val_df[cfg.INPUT_COL].tolist(),
            val_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_val if cfg.USE_TDA else None
        )

        test_ds = ChatDataset(
            test_df[cfg.INPUT_COL].tolist(),
            test_df[cfg.OUTPUT_COL].tolist(),
            src_vocab,
            tgt_vocab,
            tda=tda_test if cfg.USE_TDA else None
        )

        # -------------------------
        # DATALOADERS
        # -------------------------
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,   # decoding is easier & correct with batch=1
            shuffle=False,
            collate_fn=collate_fn
        )

        # -------------------------
        # TRAIN
        # -------------------------
        best = train_t4(
            cfg,
            train_loader,
            val_loader,
            test_loader,
            src_vocab,
            tgt_vocab,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # -------------------------
        # RESULTS
        # -------------------------
        print("\n FINAL T4 RESULT")
        print({
            "best_bleu": best["best_bleu"],
            "test_bleu": best["test_bleu"]
        })


    else:
        raise ValueError(f"Unknown model {model_name}")
