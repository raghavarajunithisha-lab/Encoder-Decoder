# models/t4_model.py

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import sacrebleu

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


# =========================
# TOKENIZE
# =========================

def simple_tokenize(text):
    return str(text).strip().split() if isinstance(text, str) else []


# =========================
# VOCAB
# =========================

class Vocab:
    def __init__(self, tokens_list=None, min_freq=1, max_size=None):
        freq = {}
        if tokens_list:
            for toks in tokens_list:
                for t in toks:
                    freq[t] = freq.get(t, 0) + 1

        words = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for w, f in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= min_freq and (max_size is None or len(words) < max_size):
                words.append(w)

        self.token2idx = {w: i for i, w in enumerate(words)}
        self.idx2token = {i: w for w, i in self.token2idx.items()}

    def __len__(self):
        return len(self.token2idx)

    def token_to_idx(self, tok):
        return self.token2idx.get(tok, self.token2idx[UNK_TOKEN])

    def idx_to_token(self, idx):
        return self.idx2token.get(idx, UNK_TOKEN)

    def encode(self, tokens):
        return (
            [self.token_to_idx(SOS_TOKEN)] +
            [self.token_to_idx(t) for t in tokens] +
            [self.token_to_idx(EOS_TOKEN)]
        )


# =========================
# DATASET
# =========================

class ChatDataset(Dataset):
    def __init__(self, inputs, outputs, src_vocab, tgt_vocab, tda=None):
        self.inputs = inputs
        self.outputs = outputs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tda = None if tda is None else torch.tensor(tda, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = torch.tensor(
            self.src_vocab.encode(simple_tokenize(self.inputs[idx])),
            dtype=torch.long
        )
        tgt = torch.tensor(
            self.tgt_vocab.encode(simple_tokenize(self.outputs[idx])),
            dtype=torch.long
        )

        if self.tda is not None:
            return src, tgt, self.tda[idx]
        return src, tgt


def collate_fn(batch):
    if len(batch[0]) == 3:
        src, tgt, tda = zip(*batch)
        tda = torch.stack(tda)
    else:
        src, tgt = zip(*batch)
        tda = None

    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt, tda


# =========================
# MODEL
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerSeq2Seq(nn.Module):
    def __init__(self, cfg, src_vocab, tgt_vocab):
        super().__init__()

        self.use_tda = cfg.USE_TDA

        self.src_emb = nn.Embedding(len(src_vocab), cfg.EMBED_SIZE, padding_idx=0)
        self.tgt_emb = nn.Embedding(len(tgt_vocab), cfg.EMBED_SIZE, padding_idx=0)
        self.pos = PositionalEncoding(cfg.EMBED_SIZE)
        self.drop = nn.Dropout(cfg.DROPOUT)

        enc_layer = nn.TransformerEncoderLayer(
            cfg.EMBED_SIZE, cfg.NUM_HEADS, cfg.FF_DIM, cfg.DROPOUT, batch_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            cfg.EMBED_SIZE, cfg.NUM_HEADS, cfg.FF_DIM, cfg.DROPOUT, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(enc_layer, cfg.NUM_LAYERS)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.NUM_LAYERS)

        self.fc = nn.Linear(cfg.EMBED_SIZE, len(tgt_vocab))

        self.tda_proj = nn.Linear(cfg.TDA_PCA_COMPONENTS, cfg.EMBED_SIZE)

    def forward(self, src, tgt_in, tda=None):
        src_pad = (src == 0)
        tgt_pad = (tgt_in == 0)

        tgt_len = tgt_in.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=src.device),
            diagonal=1
        ).bool()

        src_emb = self.drop(self.pos(self.src_emb(src)))
        tgt_emb = self.drop(self.pos(self.tgt_emb(tgt_in)))

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad)

        if self.use_tda and tda is not None:
            tda_vec = torch.tanh(self.tda_proj(tda)).unsqueeze(1)
            memory = memory + tda_vec

        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad
        )

        return self.fc(out)


# =========================
# TRAIN / VAL / TEST
# =========================

def train_t4(cfg, train_loader, val_loader, test_loader,
             src_vocab, tgt_vocab, device):

    model = TransformerSeq2Seq(cfg, src_vocab, tgt_vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_bleu = -1.0
    patience = cfg.EARLY_STOPPING_PATIENCE
    no_improve_epochs = 0

    history = []

    # -------- TRAIN + VAL --------
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        total_loss = 0
        train_preds, train_refs = [], []

        for src, tgt, tda in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)
            tda = None if tda is None else tda.to(device)

            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1], tda)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ---- TRAIN BLEU GENERATION ----
            pred_ids = logits.argmax(-1)
            for i in range(src.size(0)):
                pred_tokens = [
                    tgt_vocab.idx_to_token(int(x))
                    for x in pred_ids[i]
                    if tgt_vocab.idx_to_token(int(x)) != EOS_TOKEN
                ]
                ref_tokens = [
                    tgt_vocab.idx_to_token(int(x))
                    for x in tgt[i][1:]
                    if tgt_vocab.idx_to_token(int(x)) != EOS_TOKEN
                ]
                train_preds.append(" ".join(pred_tokens))
                train_refs.append(" ".join(ref_tokens))

        train_loss = total_loss / len(train_loader)
        train_bleu = sacrebleu.corpus_bleu(train_preds, [[r] for r in train_refs]).score / 100

        # -------- VALIDATION --------
        model.eval()
        val_preds, val_refs = [], []
        val_loss = 0

        with torch.no_grad():
            for src, tgt, tda in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tda = None if tda is None else tda.to(device)

                logits = model(src, tgt[:, :-1], tda)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
                val_loss += loss.item()

                pred_ids = logits.argmax(-1)
                for i in range(src.size(0)):
                    pred_tokens = [
                        tgt_vocab.idx_to_token(int(x))
                        for x in pred_ids[i]
                        if tgt_vocab.idx_to_token(int(x)) != EOS_TOKEN
                    ]
                    ref_tokens = [
                        tgt_vocab.idx_to_token(int(x))
                        for x in tgt[i][1:]
                        if tgt_vocab.idx_to_token(int(x)) != EOS_TOKEN
                    ]
                    val_preds.append(" ".join(pred_tokens))
                    val_refs.append(" ".join(ref_tokens))

        val_loss /= len(val_loader)
        val_bleu = sacrebleu.corpus_bleu(val_preds, [[r] for r in val_refs]).score / 100

        # ✅ STORE HISTORY (MATCHES YOUR EXPECTED FORMAT)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_bleu": train_bleu,
            "val_loss": val_loss,
            "val_bleu": val_bleu
        })

        print(f"\nEpoch {epoch+1}:")
        print({"train_loss": train_loss, "train_bleu": train_bleu})
        print({"val_loss": val_loss, "val_bleu": val_bleu})

        # -------- EARLY STOPPING --------
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_t4_transformer.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # -------- TEST --------
    model.load_state_dict(torch.load("best_t4_transformer.pth"))
    model.eval()

    preds, refs = [], []
    with torch.no_grad():
        for src, tgt, tda in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            tda = None if tda is None else tda.to(device)

            gen = torch.full((1, 1), tgt_vocab.token_to_idx(SOS_TOKEN), device=device)
            out = []

            for _ in range(cfg.MAX_OUTPUT_LEN):
                logits = model(src[:1], gen, tda[:1] if tda is not None else None)
                nxt = logits[:, -1].argmax(-1).item()
                if tgt_vocab.idx_to_token(nxt) == EOS_TOKEN:
                    break
                out.append(tgt_vocab.idx_to_token(nxt))
                gen = torch.cat([gen, torch.tensor([[nxt]], device=device)], 1)

            preds.append(" ".join(out))
            refs.append(" ".join(
                tgt_vocab.idx_to_token(int(x))
                for x in tgt[0][1:]
                if tgt_vocab.idx_to_token(int(x)) != EOS_TOKEN
            ))

    test_bleu = sacrebleu.corpus_bleu(preds, [[r] for r in refs]).score / 100

    return {
        "model": model,
        "history": history,
        "best_bleu": best_bleu,
        "test_bleu": test_bleu
    }
