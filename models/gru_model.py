# models/gru_model.py
# GRU seq2seq model and training helpers (mirrors previous GRU code)

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
    if not isinstance(text, str):
        return []
    return text.strip().split()

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
        if token in self.freqs:
            self.freqs[token] += 1
        else:
            self.freqs[token] = 1

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

class ChatDatasetPlain(Dataset):
    def __init__(self, inputs, outputs, src_vocab, tgt_vocab):
        assert len(inputs) == len(outputs)
        self.inputs = inputs
        self.outputs = outputs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src_tokens = [SOS_TOKEN] + simple_tokenize(self.inputs[idx]) + [EOS_TOKEN]
        tgt_tokens = [SOS_TOKEN] + simple_tokenize(self.outputs[idx]) + [EOS_TOKEN]
        src_idx = [self.src_vocab.token_to_idx(tok) for tok in src_tokens]
        tgt_idx = [self.tgt_vocab.token_to_idx(tok) for tok in tgt_tokens]
        return torch.tensor(src_idx, dtype=torch.long), torch.tensor(tgt_idx, dtype=torch.long)

def collate_fn_plain(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    src_pad = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_pad, torch.tensor(src_lens), tgt_pad, torch.tensor(tgt_lens)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        h_dec_init = torch.tanh(self.fc_hidden(h_cat))
        return out, h_dec_init.unsqueeze(0)

class LuongAttention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.fc = nn.Linear(enc_hidden_size, dec_hidden_size, bias=False)

    def forward(self, dec_hidden, encoder_outputs, mask=None):
        proj = self.fc(encoder_outputs)
        dec_hidden_unsq = dec_hidden.unsqueeze(2)
        scores = torch.bmm(proj, dec_hidden_unsq).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers=1, dropout=0.1, use_tda_in_decoder=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size + enc_hidden_size, dec_hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = LuongAttention(enc_hidden_size, dec_hidden_size)
        self.fc_out = nn.Linear(dec_hidden_size + enc_hidden_size + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.use_tda_in_decoder = use_tda_in_decoder

    def forward(self, input_step, last_hidden, encoder_outputs, mask, tda_context=None):
        embedded = self.dropout(self.embedding(input_step).unsqueeze(1))
        dec_h = last_hidden[-1]
        context, attn_weights = self.attention(dec_h, encoder_outputs, mask)
        if (tda_context is not None) and self.use_tda_in_decoder:
            context = context + tda_context
        gru_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.gru(gru_input, last_hidden)
        output = output.squeeze(1)
        concat_input = torch.cat((output, context, embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(concat_input)
        return prediction, hidden, attn_weights

def create_masks(src_batch):
    return (src_batch != 0)

def train_gru(cfg, train_loader, val_loader, src_vocab, tgt_vocab, device):
    encoder = Encoder(len(src_vocab), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT).to(device)
    decoder = Decoder(len(tgt_vocab), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE * 2, cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT, use_tda_in_decoder=cfg.USE_TDA).to(device)
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=cfg.LEARNING_RATE)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_bleu = -1.0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        encoder.train(); decoder.train()
        total_loss = 0.0
        for src_batch, src_lens, tgt_batch, tgt_lens in tqdm(train_loader, desc="Train"):
            src_batch = src_batch.to(device); tgt_batch = tgt_batch.to(device); src_lens = src_lens.to(device)
            optimizer_enc.zero_grad(); optimizer_dec.zero_grad()
            encoder_outputs, encoder_hidden = encoder(src_batch, src_lens)
            decoder_hidden = encoder_hidden
            mask = create_masks(src_batch).to(device)
            input_step = tgt_batch[:, 0]
            loss = 0.0
            max_tgt_len = tgt_batch.size(1)
            for t in range(1, max_tgt_len):
                output_logits, decoder_hidden, _ = decoder(input_step, decoder_hidden, encoder_outputs, mask, tda_context=None)
                loss_step = criterion(output_logits, tgt_batch[:, t])
                loss = loss + loss_step
                teacher_force = random.random() < cfg.TEACHER_FORCING_RATIO
                top1 = output_logits.argmax(1)
                input_step = tgt_batch[:, t] if teacher_force else top1
            loss = loss / (max_tgt_len - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.CLIP_GRAD)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.CLIP_GRAD)
            optimizer_enc.step(); optimizer_dec.step()
            total_loss += loss.item() * src_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # evaluate
        encoder.eval(); decoder.eval()
        all_preds, all_refs = [], []
        with torch.no_grad():
            for src_batch, src_lens, tgt_batch, tgt_lens in tqdm(val_loader, desc="Val"):
                src_batch = src_batch.to(device); tgt_batch = tgt_batch.to(device); src_lens = src_lens.to(device)
                encoder_outputs, encoder_hidden = encoder(src_batch, src_lens)
                decoder_hidden = encoder_hidden
                input_step = torch.tensor([tgt_vocab.token_to_idx(SOS_TOKEN)] * src_batch.size(0), dtype=torch.long, device=device)
                preds_tokens = [[] for _ in range(src_batch.size(0))]
                for _ in range(cfg.MAX_OUTPUT_LEN):
                    output_logits, decoder_hidden, _ = decoder(input_step, decoder_hidden, encoder_outputs, create_masks(src_batch).to(device), tda_context=None)
                    top1 = output_logits.argmax(1)
                    input_step = top1
                    for i in range(src_batch.size(0)):
                        preds_tokens[i].append(tgt_vocab.idx_to_token(int(top1[i].cpu().numpy())))
                for i in range(src_batch.size(0)):
                    toks = preds_tokens[i]
                    if EOS_TOKEN in toks:
                        toks = toks[:toks.index(EOS_TOKEN)]
                    filtered = [t for t in toks if t not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
                    all_preds.append(" ".join(filtered).strip())
                for i in range(src_batch.size(0)):
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
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "src_vocab": src_vocab.token2idx,
                "tgt_vocab": tgt_vocab.token2idx,
            }, "best_seq2seq_gru.pth")
    return best_bleu
