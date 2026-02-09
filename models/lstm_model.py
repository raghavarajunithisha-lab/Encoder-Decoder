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

def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    return text.strip().split()

# ---------------- DATASET ----------------
class ChatDataset(Dataset):
    def __init__(self, inputs, outputs, src_vocab, tgt_vocab, tda=None):
        self.inputs = [str(x) for x in inputs]
        self.outputs = [str(x) for x in outputs]
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tda = tda

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src_tokens = [SOS_TOKEN] + simple_tokenize(self.inputs[idx]) + [EOS_TOKEN]
        tgt_tokens = [SOS_TOKEN] + simple_tokenize(self.outputs[idx]) + [EOS_TOKEN]
        src_idx = [self.src_vocab.token_to_idx(tok) for tok in src_tokens]
        tgt_idx = [self.tgt_vocab.token_to_idx(tok) for tok in tgt_tokens]
        if self.tda is not None:
            return torch.tensor(src_idx), torch.tensor(tgt_idx), torch.tensor(self.tda[idx], dtype=torch.float32)
        return torch.tensor(src_idx), torch.tensor(tgt_idx)

# ---------------- COLLATE ----------------
def collate_fn(batch):
    if len(batch[0]) == 3:
        src, tgt, tda = zip(*batch)
        tda = torch.stack(tda)
    else:
        src, tgt = zip(*batch)
        tda = None
    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    src_lens = torch.tensor([len(s) for s in src])
    return src_pad, src_lens, tgt_pad, tda

# ---------------- MODEL ----------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, tda_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.use_tda = tda_dim is not None
        if self.use_tda:
            self.tda_proj = nn.Linear(tda_dim, hidden_size)

    def forward(self, x, lens, tda=None):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        h0 = self.fc(h_cat)
        if self.use_tda and tda is not None:
            h0 = h0 + self.tda_proj(tda)
        h0 = h0.unsqueeze(0)
        return (h0, torch.zeros_like(h0))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        emb = self.embedding(x).unsqueeze(1)
        out, h = self.lstm(emb, h)
        return self.fc(out.squeeze(1)), h

# ---------------- EVALUATION HELPER ----------------
def run_evaluation(encoder, decoder, loader, tgt_vocab, device, loss_fn=None):
    encoder.eval()
    decoder.eval()
    all_preds, all_refs, total_loss = [], [], 0
    special = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}

    with torch.no_grad():
        for src, lens, tgt, tda in loader:
            src, tgt, lens = src.to(device), tgt.to(device), lens.to(device)
            tda = tda.to(device) if tda is not None else None
            h = encoder(src, lens, tda)
            inp = tgt[:, 0]
            
            batch_preds = [[] for _ in range(src.size(0))]
            for t in range(1, tgt.size(1)):
                out, h = decoder(inp, h)
                if loss_fn:
                    total_loss += loss_fn(out, tgt[:, t]).item()
                inp = out.argmax(1)
                for i, token_id in enumerate(inp):
                    batch_preds[i].append(tgt_vocab.idx_to_token(token_id.item()))

            for i in range(src.size(0)):
                ref_tokens = [tgt_vocab.idx_to_token(tid.item()) for tid in tgt[i][1:]]
                all_preds.append(" ".join([t for t in batch_preds[i] if t not in special]))
                all_refs.append(" ".join([t for t in ref_tokens if t not in special]))

    bleu = sacrebleu.corpus_bleu(all_preds, [[r] for r in all_refs]).score / 100
    avg_loss = total_loss / (len(loader) * tgt.size(1)) if loss_fn else 0
    return bleu, avg_loss

# ---------------- TRAIN ----------------
def train_lstm(cfg, train_loader, val_loader, test_loader, src_vocab, tgt_vocab, device):
    tda_dim = getattr(cfg, "TDA_PCA_COMPONENTS", None) if getattr(cfg, "USE_TDA", False) else None
    encoder = Encoder(len(src_vocab), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE, tda_dim).to(device)
    decoder = Decoder(len(tgt_vocab), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE).to(device)

    opt_e = torch.optim.Adam(encoder.parameters(), lr=cfg.LEARNING_RATE)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=cfg.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_bleu, counter, patience = 0.0, 0, 3
    history = []

    for epoch in range(cfg.NUM_EPOCHS):
        encoder.train()
        decoder.train()
        epoch_loss = 0

        for src, lens, tgt, tda in train_loader:
            src, tgt, lens = src.to(device), tgt.to(device), lens.to(device)
            tda = tda.to(device) if tda is not None else None
            opt_e.zero_grad(); opt_d.zero_grad()
            h = encoder(src, lens, tda)
            inp, batch_loss = tgt[:, 0], 0
            for t in range(1, tgt.size(1)):
                out, h = decoder(inp, h)
                batch_loss += loss_fn(out, tgt[:, t])
                inp = tgt[:, t] # Teacher Forcing
            batch_loss.backward()
            opt_e.step(); opt_d.step()
            epoch_loss += batch_loss.item()

        train_bleu, _ = run_evaluation(encoder, decoder, train_loader, tgt_vocab, device)
        val_bleu, val_loss = run_evaluation(encoder, decoder, val_loader, tgt_vocab, device, loss_fn)

        print(f"\nEpoch {epoch+1}:")
        print({'train_loss': epoch_loss / len(train_loader), 'train_bleu': train_bleu})
        print({'val_loss': val_loss, 'val_bleu': val_bleu})

        history.append({"epoch": epoch+1, "val_bleu": val_bleu, "train_loss": epoch_loss/len(train_loader)})

        if val_bleu > best_bleu:
            best_bleu, counter = val_bleu, 0
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, "best_lstm.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered."); break

    checkpoint = torch.load("best_lstm.pth")
    encoder.load_state_dict(checkpoint['encoder']); decoder.load_state_dict(checkpoint['decoder'])
    test_bleu, test_loss = run_evaluation(encoder, decoder, test_loader, tgt_vocab, device, loss_fn)

    return {"history": history, "best_bleu": best_bleu, "test_loss": test_loss, "test_bleu": test_bleu}