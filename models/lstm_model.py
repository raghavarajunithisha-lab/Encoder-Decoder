# models/lstm_model.py
# Robust LSTM seq2seq model and train/eval helpers with safer batching and debugging.

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import sacrebleu

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# simple tokenizer (replace with nltk.word_tokenize if you want)
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
        # reserve special tokens
        for tok in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self.add_token(tok, reserve=True)

    def add_token(self, token, reserve=False):
        if token in self.freqs:
            self.freqs[token] += 1
        else:
            self.freqs[token] = 0 if reserve else 1

    def build(self):
        # only keep tokens meeting min freq (plus the special tokens)
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


class ChatDataset(Dataset):
    """Dataset returns (src_tensor, tgt_tensor) or (src, tgt, tda_vec)."""
    def __init__(self, inputs, outputs, src_vocab, tgt_vocab, tda_features=None):
        assert len(inputs) == len(outputs)
        self.inputs = ["" if x is None else str(x) for x in inputs]
        self.outputs = ["" if x is None else str(x) for x in outputs]
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tda_features = None if tda_features is None else tda_features.astype(np.float32)
        # ensure lengths match if TDA provided
        if self.tda_features is not None and len(self.tda_features) != len(self.inputs):
            raise ValueError("tda_features length must match inputs length")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # build token lists and map to indices
        src_tokens = [SOS_TOKEN] + simple_tokenize(self.inputs[idx]) + [EOS_TOKEN]
        tgt_tokens = [SOS_TOKEN] + simple_tokenize(self.outputs[idx]) + [EOS_TOKEN]
        # if either side is empty, return minimal tokens so code downstream does not break
        if len(src_tokens) == 0:
            src_tokens = [SOS_TOKEN, EOS_TOKEN]
        if len(tgt_tokens) == 0:
            tgt_tokens = [SOS_TOKEN, EOS_TOKEN]
        src_idx = [self.src_vocab.token_to_idx(tok) for tok in src_tokens]
        tgt_idx = [self.tgt_vocab.token_to_idx(tok) for tok in tgt_tokens]
        src_tensor = torch.tensor(src_idx, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_idx, dtype=torch.long)
        if self.tda_features is None:
            return src_tensor, tgt_tensor
        else:
            return src_tensor, tgt_tensor, torch.from_numpy(self.tda_features[idx])


def collate_fn(batch):
    """
    Handles batches of either pairs (src, tgt) or triples (src, tgt, tda).
    Pads sequences and returns tensors + length lists.
    """
    # ensure batch not empty
    if len(batch) == 0:
        raise ValueError("Received empty batch in collate_fn")

    # detect if TDA present by tuple length
    example = batch[0]
    has_tda = (len(example) == 3)

    if has_tda:
        src_seqs, tgt_seqs, tda_vals = zip(*batch)
    else:
        src_seqs, tgt_seqs = zip(*batch)
        tda_vals = None

    # convert to padded tensors (pad value is 0)
    src_pad = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    src_lens = torch.tensor([s.size(0) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([t.size(0) for t in tgt_seqs], dtype=torch.long)

    if has_tda:
        tda_tensor = torch.stack(tda_vals)
        return src_pad, src_lens, tgt_pad, tgt_lens, tda_tensor
    else:
        return src_pad, src_lens, tgt_pad, tgt_lens


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: (B, T)
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # get forward/backward last states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        c_forward = c_n[-2, :, :]
        c_backward = c_n[-1, :, :]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        c_cat = torch.cat((c_forward, c_backward), dim=1)
        h_dec_init = torch.tanh(self.fc_hidden(h_cat))
        c_dec_init = torch.tanh(self.fc_cell(c_cat))
        # return encoder outputs and initial decoder hidden (num_layers, batch, hidden)
        return out, (h_dec_init.unsqueeze(0), c_dec_init.unsqueeze(0))


class LuongAttention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.fc = nn.Linear(enc_hidden_size, dec_hidden_size, bias=False)

    def forward(self, dec_hidden, encoder_outputs, mask=None):
        # dec_hidden: (batch, dec_hidden)
        proj = self.fc(encoder_outputs)  # (batch, src_len, dec_hidden)
        dec_hidden_unsq = dec_hidden.unsqueeze(2)  # (batch, dec_hidden, 1)
        scores = torch.bmm(proj, dec_hidden_unsq).squeeze(2)  # (batch, src_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_hidden)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size + enc_hidden_size, dec_hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = LuongAttention(enc_hidden_size, dec_hidden_size)
        self.fc_out = nn.Linear(dec_hidden_size + enc_hidden_size + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, last_hidden, encoder_outputs, mask):
        # input_step: (batch,) token ids
        embedded = self.dropout(self.embedding(input_step).unsqueeze(1))  # (batch, 1, embed)
        dec_h = last_hidden[0][-1]  # (batch, hidden)
        context, attn_weights = self.attention(dec_h, encoder_outputs, mask)
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.lstm(lstm_input, last_hidden)
        output = output.squeeze(1)
        concat_input = torch.cat((output, context, embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(concat_input)
        return prediction, hidden, attn_weights


def create_masks(src_batch, src_lengths, device):
    return (src_batch != 0).to(device)


def train_lstm(cfg, train_loader, val_loader, src_vocab, tgt_vocab, device):
    """
    Train loop for LSTM seq2seq. This function is defensive:
    - prints debug for first batches
    - skips empty batches
    - catches and reports exceptions per-batch (so you can see which data row caused trouble)
    """
    encoder = Encoder(len(src_vocab), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT).to(device)
    decoder = Decoder(len(tgt_vocab), cfg.EMBED_SIZE, cfg.HIDDEN_SIZE * 2, cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT).to(device)
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=cfg.LEARNING_RATE)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_bleu = -1.0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        encoder.train(); decoder.train()
        total_loss = 0.0

        # training batches
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            # defensive: ensure batch has expected form
            try:
                if batch is None:
                    print(f"Skipping None batch at idx {batch_idx}")
                    continue
                if len(batch) == 4:
                    src_batch, src_lens, tgt_batch, tgt_lens = batch
                    tda_batch = None
                elif len(batch) == 5:
                    src_batch, src_lens, tgt_batch, tgt_lens, tda_batch = batch
                else:
                    print(f"Unexpected batch format (len={len(batch)}) at idx {batch_idx}, skipping")
                    continue

                # skip any zero-length examples
                if src_batch.size(0) == 0 or tgt_batch.size(0) == 0:
                    print(f"Skipping empty batch at idx {batch_idx}")
                    continue

                # move to device
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                src_lens = src_lens.to(device)

                optimizer_enc.zero_grad(); optimizer_dec.zero_grad()

                # forward
                encoder_outputs, encoder_hidden = encoder(src_batch, src_lens)
                if tda_batch is not None:
                    # placeholder for optional TDA integration if provided
                    pass
                decoder_hidden = encoder_hidden
                mask = create_masks(src_batch, src_lens, device)

                # initialize decoder input (batch of <sos>)
                input_step = tgt_batch[:, 0]  # using teacher forcing targets or greedy input later

                loss = 0.0
                max_tgt_len = tgt_batch.size(1)
                # protect against degenerate target length
                if max_tgt_len <= 1:
                    # nothing to predict
                    continue

                for t in range(1, max_tgt_len):
                    output_logits, decoder_hidden, _ = decoder(input_step, decoder_hidden, encoder_outputs, mask)
                    loss_step = criterion(output_logits, tgt_batch[:, t])
                    loss = loss + loss_step
                    teacher_force = random.random() < cfg.TEACHER_FORCING_RATIO
                    top1 = output_logits.argmax(1)
                    input_step = tgt_batch[:, t] if teacher_force else top1

                # average loss over time steps
                loss = loss / float(max_tgt_len - 1)
                loss.backward()

                # gradient clipping and step
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer_enc.step(); optimizer_dec.step()
                total_loss += loss.item() * src_batch.size(0)

                # debug: print first few batch shapes
                if batch_idx < 3:
                    print(f"[E{epoch}] batch {batch_idx} shapes: src {src_batch.shape}, tgt {tgt_batch.shape}")

            except Exception as e:
                # do not crash entire training; report the problem and continue
                print(f"Exception in training loop at batch {batch_idx}: {e}")
                continue

        avg_train_loss = total_loss / max(1, len(train_loader.dataset))

        # validation (greedy)
        encoder.eval(); decoder.eval()
        all_preds, all_refs = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val Epoch {epoch}")):
                try:
                    if len(batch) == 4:
                        src_batch, src_lens, tgt_batch, tgt_lens = batch
                    else:
                        src_batch, src_lens, tgt_batch, tgt_lens, _ = batch

                    if src_batch.size(0) == 0:
                        continue

                    src_batch = src_batch.to(device)
                    tgt_batch = tgt_batch.to(device)
                    src_lens = src_lens.to(device)

                    encoder_outputs, encoder_hidden = encoder(src_batch, src_lens)
                    decoder_hidden = encoder_hidden

                    # start decode with sos token
                    sos_idx = tgt_vocab.token_to_idx(SOS_TOKEN)
                    input_step = torch.tensor([sos_idx] * src_batch.size(0), dtype=torch.long, device=device)

                    preds_tokens = [[] for _ in range(src_batch.size(0))]
                    for _ in range(cfg.MAX_OUTPUT_LEN):
                        output_logits, decoder_hidden, _ = decoder(input_step, decoder_hidden, encoder_outputs, create_masks(src_batch, src_lens, device))
                        top1 = output_logits.argmax(1)
                        input_step = top1
                        for i in range(src_batch.size(0)):
                            preds_tokens[i].append(tgt_vocab.idx_to_token(int(top1[i].cpu().numpy())))

                    # convert preds to strings
                    for i in range(src_batch.size(0)):
                        toks = preds_tokens[i]
                        if EOS_TOKEN in toks:
                            toks = toks[:toks.index(EOS_TOKEN)]
                        filtered = [t for t in toks if t not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
                        all_preds.append(" ".join(filtered).strip())

                    # convert refs to strings
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

                except Exception as e:
                    print(f"Exception in validation at batch {batch_idx}: {e}")
                    continue

        bleu = sacrebleu.corpus_bleu(all_preds, [[r] for r in all_refs]).score / 100.0 if len(all_preds) > 0 else 0.0
        print(f"Epoch {epoch} TrainLoss={avg_train_loss:.4f} BLEU={bleu:.4f}")

        # save best
        if bleu > best_bleu:
            best_bleu = bleu
            try:
                torch.save({
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "src_vocab": src_vocab.token2idx,
                    "tgt_vocab": tgt_vocab.token2idx
                }, "best_seq2seq_lstm.pth")
                print(f"Saved best model (BLEU={bleu:.4f})")
            except Exception as e:
                print("Could not save model:", e)

    return best_bleu
