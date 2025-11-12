# utils/data_utils.py
# Data loading and simple tokenization + helpers shared across models.

import os
import pandas as pd
import numpy as np

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def load_csv(path, input_col, output_col, max_examples=None):
    df = pd.read_csv(path)
    if input_col not in df.columns or output_col not in df.columns:
        raise ValueError(f"Columns {input_col} or {output_col} not in {path}")
    df = df.dropna(subset=[input_col, output_col]).reset_index(drop=True)
    df[input_col] = df[input_col].astype(str).str.strip()
    df[output_col] = df[output_col].astype(str).str.strip()
    if max_examples:
        df = df.iloc[:max_examples].reset_index(drop=True)
    return df

class SimpleVocab:
    def __init__(self, max_size=20000, min_freq=1):
        self.token2idx = {}
        self.idx2token = []
        self.freqs = {}
        self.max_size = max_size
        self.min_freq = min_freq
        for tok in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self.add_token(tok, reserve=True)

    def add_token(self, token, reserve=False):
        if token in self.freqs:
            self.freqs[token] += 1
        else:
            self.freqs[token] = 1 if not reserve else 0

    def build(self):
        items = [(t, f) for t, f in self.freqs.items() if (f >= self.min_freq or t in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN))]
        items.sort(key=lambda x: (-x[1], x[0]))
        self.idx2token = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.token2idx = {tok: i for i, tok in enumerate(self.idx2token)}
        for tok, freq in items:
            if tok in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
                continue
            if len(self.idx2token) >= self.max_size:
                break
            self.token2idx[tok] = len(self.idx2token)
            self.idx2token.append(tok)

    def token_to_idx(self, token):
        return self.token2idx.get(token, self.token2idx[UNK_TOKEN])

    def idx_to_token(self, idx):
        if idx < 0 or idx >= len(self.idx2token):
            return UNK_TOKEN
        return self.idx2token[idx]

    def __len__(self):
        return len(self.idx2token)
