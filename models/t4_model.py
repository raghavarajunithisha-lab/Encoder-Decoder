# models/t4_model.py
# Transformer seq2seq (T4) implementation (PyTorch Transformer)

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def simple_tokenize(text):
    return str(text).strip().split() if isinstance(text, str) else []

class Vocab:
    def __init__(self, tokens_list=None, min_freq=1, max_size=None):
        token_freq = {}
        if tokens_list:
            for tokens in tokens_list:
                for t in tokens:
                    token_freq[t] = token_freq.get(t, 0) + 1
        sorted_tokens = sorted(token_freq.items(), key=lambda x: (-x[1], x[0]))
        words = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for word, freq in sorted_tokens:
            if freq >= min_freq and (max_size is None or len(words) < max_size):
                words.append(word)
        self.token2idx = {w: i for i, w in enumerate(words)}
        self.idx2token = {i: w for i, w in enumerate(words)}

    def token_to_idx(self, token):
        return self.token2idx.get(token, self.token2idx[UNK_TOKEN])

    def idx_to_token(self, idx):
        return self.idx2token.get(idx, UNK_TOKEN)

    def encode(self, tokens):
        return [self.token_to_idx(SOS_TOKEN)] + [self.token_to_idx(t) for t in tokens] + [self.token_to_idx(EOS_TOKEN)]

    def decode(self, indices):
        return [self.idx2token.get(i, UNK_TOKEN) for i in indices if i not in [self.token2idx[PAD_TOKEN], self.token2idx[SOS_TOKEN], self.token2idx[EOS_TOKEN]]]

class ChatDataset(Dataset):
    def __init__(self, inputs, outputs, src_vocab, tgt_vocab, tda_features=None):
        self.inputs = inputs
        self.outputs = outputs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tda_features = tda_features

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        src_ids = torch.tensor(self.src_vocab.encode(simple_tokenize(self.inputs[idx])), dtype=torch.long)
        tgt_ids = torch.tensor(self.tgt_vocab.encode(simple_tokenize(self.outputs[idx])), dtype=torch.long)
        if self.tda_features is not None:
            return src_ids, tgt_ids, torch.tensor(self.tda_features[idx], dtype=torch.float)
        return src_ids, tgt_ids

def collate_fn(batch):
    if len(batch[0]) == 2:
        src_seqs, tgt_seqs = zip(*batch)
        tda = None
    else:
        src_seqs, tgt_seqs, tda = zip(*batch)
        tda = torch.stack(tda)
    src_pad = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    src_lens = [s.size(0) for s in src_seqs]
    tgt_lens = [t.size(0) for t in tgt_seqs]
    return src_pad, src_lens, tgt_pad, tgt_lens, tda

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def create_padding_mask(batch, pad_idx=0):
    return (batch == pad_idx)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, num_layers=4, nhead=8, ff_dim=1024, dropout=0.1, tda_dim=None):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, embed_size, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, nhead, ff_dim, dropout, batch_first=True),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, nhead, ff_dim, dropout, batch_first=True),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.tda_proj = nn.Linear(tda_dim, embed_size) if tda_dim is not None else None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.float().masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, tda_vec=None):
        src_emb = self.dropout(self.pos_enc(self.src_embed(src)))
        tgt_emb = self.dropout(self.pos_enc(self.tgt_embed(tgt)))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        if self.tda_proj is not None and tda_vec is not None:
            tda_proj_vec = torch.tanh(self.tda_proj(tda_vec))
            if memory.size(1) > 0:
                memory[:, 0, :] = memory[:, 0, :] + tda_proj_vec.to(memory.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        return self.fc_out(output)

def train_t4(cfg, train_loader, val_loader, src_vocab, tgt_vocab, device):
    tda_dim = None
    if cfg.USE_TDA and hasattr(cfg, "TDA_PCA_DIM"):
        tda_dim = cfg.TDA_PCA_DIM
    model = TransformerSeq2Seq(len(src_vocab), len(tgt_vocab), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, nhead=cfg.NUM_HEADS, ff_dim=cfg.FF_DIM, dropout=cfg.DROPOUT, tda_dim=tda_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_bleu = -1.0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for src_batch, src_lens, tgt_batch, tgt_lens, tda_batch in tqdm(train_loader, desc="Train"):
            src_batch = src_batch.to(device); tgt_batch = tgt_batch.to(device)
            if tda_batch is not None: tda_batch = tda_batch.to(device)
            optimizer.zero_grad()
            src_mask = create_padding_mask(src_batch)
            tgt_mask = create_padding_mask(tgt_batch)
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            logits = model(src_batch, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask[:, :-1], tda_vec=tda_batch if cfg.USE_TDA else None)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * src_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # eval
        model.eval()
        all_preds, all_refs = [], []
        with torch.no_grad():
            for src_batch, src_lens, tgt_batch, tgt_lens, tda_batch in tqdm(val_loader, desc="Val"):
                src_batch = src_batch.to(device); tgt_batch = tgt_batch.to(device)
                if tda_batch is not None: tda_batch = tda_batch.to(device)
                src_mask = create_padding_mask(src_batch)
                tgt_mask = create_padding_mask(tgt_batch)
                tgt_input = tgt_batch[:, :-1]
                logits = model(src_batch, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask[:, :-1], tda_vec=tda_batch if cfg.USE_TDA else None)
                # greedy decode
                B = src_batch.size(0)
                generated = torch.full((B, 1), tgt_vocab.token_to_idx(SOS_TOKEN), dtype=torch.long, device=device)
                preds = [[] for _ in range(B)]
                for _ in range(cfg.MAX_OUTPUT_LEN):
                    tgt_mask_gen = create_padding_mask(generated)
                    logits_gen = model(src_batch, generated, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask_gen, tda_vec=None)
                    next_token = logits_gen[:, -1, :].argmax(-1)
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    for i in range(B):
                        preds[i].append(tgt_vocab.idx_to_token(int(next_token[i].cpu())))
                for i in range(B):
                    all_preds.append(" ".join([t for t in preds[i] if t not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]).strip())
                for i in range(B):
                    tgt_indices = tgt_batch[i].cpu().numpy().tolist()
                    toks = [tgt_vocab.idx_to_token(idx) for idx in tgt_indices]
                    if EOS_TOKEN in toks:
                        cut = toks.index(EOS_TOKEN)
                        toks = toks[1:cut] if toks[0] == SOS_TOKEN else toks[:cut]
                    else:
                        toks = toks[1:] if toks[0] == SOS_TOKEN else toks
                    toks = [t for t in toks if t not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)]
                    all_refs.append(" ".join(toks).strip())
        bleu = sacrebleu.corpus_bleu(all_preds, [[r] for r in all_refs]).score / 100.0
        print(f"Epoch {epoch} TrainLoss={avg_loss:.4f} BLEU={bleu:.4f}")
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save({"model": model.state_dict(), "src_vocab": src_vocab.token2idx, "tgt_vocab": tgt_vocab.token2idx}, "best_transformer_t4.pth")
    return best_bleu
