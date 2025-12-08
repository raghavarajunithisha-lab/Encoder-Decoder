# models/gru_model.py

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sacrebleu

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# ------------------ TOKENIZER ------------------

def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    return text.strip().split()

# ------------------ VOCAB ------------------

class Vocab:
    def __init__(self, max_size=20000, min_freq=1):
        self.token2idx = {}
        self.idx2token = []
        self.freqs = {}
        self.max_size = max_size
        self.min_freq = min_freq
        for tok in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
            self.freqs[tok] = 0

    def add_token(self, token):
        self.freqs[token] = self.freqs.get(token, 0) + 1

    def build(self):
        items = [(t, f) for t, f in self.freqs.items()
                 if (f >= self.min_freq or t in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN))]
        items.sort(key=lambda x: (-x[1], x[0]))

        self.idx2token = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.token2idx = {tok: i for i, tok in enumerate(self.idx2token)}

        for tok, _ in items:
            if tok in self.token2idx:
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

# ------------------ DATASET ------------------

class ChatDatasetPlain(Dataset):
    def __init__(self, inputs, outputs, src_vocab, tgt_vocab, tda=None):
        self.inputs = inputs
        self.outputs = outputs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tda = None if tda is None else np.asarray(tda, dtype=np.float32)

        if self.tda is not None and len(self.tda) != len(self.inputs):
            self.tda = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = [SOS_TOKEN] + simple_tokenize(self.inputs[idx]) + [EOS_TOKEN]
        tgt = [SOS_TOKEN] + simple_tokenize(self.outputs[idx]) + [EOS_TOKEN]
        src_idx = torch.tensor([self.src_vocab.token_to_idx(t) for t in src], dtype=torch.long)
        tgt_idx = torch.tensor([self.tgt_vocab.token_to_idx(t) for t in tgt], dtype=torch.long)

        if self.tda is None:
            return src_idx, tgt_idx
        else:
            return src_idx, tgt_idx, torch.tensor(self.tda[idx], dtype=torch.float32)

def collate_fn_plain(batch):
    if len(batch[0]) == 3:
        src, tgt, tda = zip(*batch)
        tda = torch.stack(tda)
    else:
        src, tgt = zip(*batch)
        tda = None

    src_lens = torch.tensor([len(s) for s in src], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt], dtype=torch.long)

    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)

    return src_pad, src_lens, tgt_pad, tgt_lens, tda

# ------------------ MODEL ------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        h_cat = torch.cat((h[-2], h[-1]), dim=1)
        h_init = torch.tanh(self.fc(h_cat))
        return out, h_init.unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.fc = nn.Linear(enc_dim, dec_dim, bias=False)

    def forward(self, h, enc_out, mask):
        proj = self.fc(enc_out)
        scores = torch.bmm(proj, h.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_dim, dec_dim, use_tda=False, tda_dim=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size + enc_dim, dec_dim, batch_first=True)
        self.attn = Attention(enc_dim, dec_dim)
        self.fc = nn.Linear(dec_dim + enc_dim + embed_size, vocab_size)

        self.use_tda = use_tda and tda_dim > 0
        if self.use_tda:
            self.tda_proj = nn.Linear(tda_dim, enc_dim)

    def forward(self, inp, h, enc_out, mask, tda_context=None):
        emb = self.embedding(inp).unsqueeze(1)
        ctx = self.attn(h[-1], enc_out, mask)

        if self.use_tda and tda_context is not None:
            ctx = ctx + self.tda_proj(tda_context)

        x = torch.cat((emb, ctx.unsqueeze(1)), dim=2)
        out, h = self.gru(x, h)
        out = out.squeeze(1)
        out = self.fc(torch.cat((out, ctx, emb.squeeze(1)), dim=1))
        return out, h

def create_masks(x):
    return (x != 0)

# ------------------ TRAIN / VAL / TEST ------------------

def train_gru(cfg, train_loader, val_loader, test_loader,
              src_vocab, tgt_vocab, device,
              early_stopping_patience=3):

    enc_dim = cfg.HIDDEN_SIZE * 2
    decoder_use_tda = bool(getattr(cfg, "USE_TDA", False))

    # infer tda_dim from first training batch
    tda_dim = 0
    if decoder_use_tda:
        for batch in train_loader:
            if len(batch) == 5 and batch[-1] is not None:
                tda_dim = batch[-1].shape[1]
            break

    encoder = Encoder(len(src_vocab.token2idx), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE).to(device)
    decoder = Decoder(len(tgt_vocab.token2idx), cfg.EMBED_SIZE, enc_dim,
                      cfg.HIDDEN_SIZE, decoder_use_tda, tda_dim).to(device)

    opt_e = torch.optim.Adam(encoder.parameters(), lr=cfg.LEARNING_RATE)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=cfg.LEARNING_RATE)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    best_bleu = -1.0
    epochs_no_improve = 0

    def greedy_decode(enc_out, h, mask, max_len, tda_batch):
        batch_size = enc_out.size(0)
        x = torch.full((batch_size,), tgt_vocab.token_to_idx(SOS_TOKEN),
                       dtype=torch.long, device=device)
        preds = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            out, h = decoder(x, h, enc_out, mask, tda_batch)
            x = out.argmax(1)
            for i in range(batch_size):
                preds[i].append(x[i].item())

        results = []
        for p in preds:
            tokens = [tgt_vocab.idx_to_token(t) for t in p]
            if EOS_TOKEN in tokens:
                tokens = tokens[:tokens.index(EOS_TOKEN)]
            results.append(" ".join(tokens))
        return results

    def evaluate(loader):
        encoder.eval()
        decoder.eval()
        preds, refs = [], []
        total_loss, total_ex = 0.0, 0

        with torch.no_grad():
            for data in loader:
                if decoder_use_tda and len(data) == 5:
                    src, src_lens, tgt, tgt_lens, tda = data
                    tda = tda.to(device)
                else:
                    src, src_lens, tgt, tgt_lens, _ = (*data, None)[:5]
                    tda = None

                src, tgt = src.to(device), tgt.to(device)
                enc_out, h = encoder(src, src_lens.to(device))
                mask = create_masks(src)

                x = tgt[:, 0]
                for t in range(1, tgt.size(1)):
                    out, h = decoder(x, h, enc_out, mask, tda)
                    total_loss += crit(out, tgt[:, t]).item()
                    x = tgt[:, t]

                total_ex += src.size(0)

                enc_out2, h2 = encoder(src, src_lens.to(device))
                batch_preds = greedy_decode(enc_out2, h2, mask, cfg.MAX_OUTPUT_LEN, tda)

                for j in range(src.size(0)):
                    preds.append(batch_preds[j])
                    ref_tokens = [tgt_vocab.idx_to_token(idx.item()) for idx in tgt[j]][1:]
                    if EOS_TOKEN in ref_tokens:
                        ref_tokens = ref_tokens[:ref_tokens.index(EOS_TOKEN)]
                    refs.append(" ".join(ref_tokens))

        avg_loss = total_loss / total_ex if total_ex > 0 else 0.0
        bleu = sacrebleu.corpus_bleu(preds, [[r] for r in refs]).score / 100.0
        return avg_loss, bleu

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        encoder.train()
        decoder.train()

        for data in train_loader:
            if decoder_use_tda and len(data) == 5:
                src, src_lens, tgt, tgt_lens, tda = data
                tda = tda.to(device)
            else:
                src, src_lens, tgt, tgt_lens, _ = (*data, None)[:5]
                tda = None

            src, tgt = src.to(device), tgt.to(device)
            opt_e.zero_grad()
            opt_d.zero_grad()

            enc_out, h = encoder(src, src_lens.to(device))
            mask = create_masks(src)

            x = tgt[:, 0]
            loss = 0.0
            for t in range(1, tgt.size(1)):
                out, h = decoder(x, h, enc_out, mask, tda)
                loss += crit(out, tgt[:, t])
                x = tgt[:, t] if random.random() < cfg.TEACHER_FORCING_RATIO else out.argmax(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.CLIP_GRAD)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.CLIP_GRAD)
            opt_e.step()
            opt_d.step()

        train_loss, train_bleu = evaluate(train_loader)
        val_loss, val_bleu = evaluate(val_loader)

        print(f"\nEpoch {epoch}:")
        print({"train_loss": train_loss, "train_bleu": train_bleu})
        print({"val_loss": val_loss, "val_bleu": val_bleu})

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({"encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict()}, "best_gru.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load("best_gru.pth", map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    test_loss, test_bleu = evaluate(test_loader)

    return {
        "best_val_bleu": best_bleu,
        "test_bleu": test_bleu,
        "test_loss": test_loss,
        "encoder": encoder,
        "decoder": decoder,
    }
