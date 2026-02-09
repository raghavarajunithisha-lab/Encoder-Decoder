# models/bart_model.py
# BART fine-tuning wrapper using HuggingFace Seq2SeqTrainer

import os
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load as load_metric
from transformers import TrainerCallback

from transformers import EarlyStoppingCallback



def prepare_tokenized_datasets(df, cfg, tda_array=None):
    tokenizer = BartTokenizerFast.from_pretrained(cfg.BART_MODEL_NAME)
    ds = Dataset.from_pandas(df)

    def preprocess_function(examples):
        # Tokenize Inputs
        model_inputs = tokenizer(
            examples[cfg.INPUT_COL],
            max_length=cfg.MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )
        
        # FIXED: Tokenize Labels without as_target_tokenizer
        labels = tokenizer(
            text_target=examples[cfg.OUTPUT_COL], # Use text_target argument
            max_length=cfg.MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )

        # Replace padding token IDs with -100 for loss masking
        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
            for labels_seq in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess_function, batched=True)

    if tda_array is not None:
        import numpy as _np
        tda_mat = _np.asarray(tda_array)
        if tda_mat.shape[0] != len(tokenized):
            desired = len(tokenized)
            if tda_mat.shape[0] < desired:
                pad = _np.zeros((desired - tda_mat.shape[0], tda_mat.shape[1]), dtype=np.float32)
                tda_mat = _np.vstack([tda_mat, pad])
            else:
                tda_mat = tda_mat[:desired]
        tda_list = [list(row.astype(float)) for row in tda_mat]
        tokenized = tokenized.add_column("tda", tda_list)

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    return tokenizer, tokenized, data_collator

def build_model(cfg):
    # Build the base BART model and — if USE_TDA — attach a learned projection for TDA
    base = BartForConditionalGeneration.from_pretrained(cfg.BART_MODEL_NAME)
    # If TDA is used, create a small projection module on the model to map tda_dim -> d_model
    if getattr(cfg, "USE_TDA", False):
        tda_dim = getattr(cfg, "TDA_PCA_COMPONENTS", getattr(cfg, "TDA_PCA_DIM", None))
        if tda_dim is None:
            raise ValueError("USE_TDA=True but no TDA_PCA_COMPONENTS/TDA_PCA_DIM provided in cfg")
        # attach projection module to the model
        base.tda_proj = nn.Linear(tda_dim, base.config.d_model)
        # flag for Trainer/forward
        base.use_tda = True
    else:
        base.use_tda = False
        base.tda_proj = None
    return base


class EpochEvalCallback(TrainerCallback):
    """ Prints metrics in the exact requested format """

    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.trainer_ref = None

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None) or self.trainer_ref
        if trainer is None:
            return
        
        epoch_num = int(state.epoch)
        
        # Perform Evaluation
        train_metrics = trainer.evaluate(eval_dataset=self.train_dataset, metric_key_prefix="train")
        val_metrics = trainer.evaluate(eval_dataset=self.val_dataset, metric_key_prefix="val")

      
        print(f"\nEpoch {epoch_num}:")
       


def get_trainer(model, tokenizer, train_dataset, eval_dataset, test_dataset, data_collator, cfg):
    metric = load_metric("bertscore")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        preds = np.where(preds != -100, preds, pad_id)
        labels = np.where(labels != -100, labels, pad_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        results = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        return {
            "bertscore_f1": float(np.mean(results["f1"]))
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        # FIXED: Must match save_strategy ("epoch") to use load_best_model_at_end
        eval_strategy="epoch",       
        # FIXED: Use "no" (not "none") to keep the console clean
        logging_strategy="no",       
        save_strategy="epoch",        
        load_best_model_at_end=True,
        # metric_for_best_model="eval_loss", # Optional: defaults to loss
        greater_is_better=False,   
        learning_rate=cfg.LEARNING_RATE,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        num_train_epochs=cfg.TRAIN_EPOCHS,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=150,
        report_to="none",          
        disable_tqdm=True,         
    )

    # If model.use_tda is True we need the Trainer to pass 'tda' from batch to model.forward.
    # Using a standard Seq2SeqTrainer will result in the batch dictionary containing 'tda' if the
    # HF Dataset has that column. The model.forward will receive it via kwargs.
    from transformers import TrainerCallback

    class EarlyStoppingCallbackCustom(TrainerCallback):
        def __init__(self, patience=5, delta=0.0):
            self.patience = patience
            self.delta = delta
            self.best_score = None
            self.counter = 0
            self.early_stop = False
            self.best_model_state = None

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            # HuggingFace provides eval_loss here
            val_loss = metrics.get("eval_loss")

            if val_loss is None:
                return control

            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.best_model_state = {k: v.cpu().clone() for k, v in kwargs["model"].state_dict().items()}

            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")

                if self.counter >= self.patience:
                    print("🛑 Early stopping triggered")
                    self.early_stop = True
                    control.should_training_stop = True

            else:
                self.best_score = score
                self.best_model_state = {k: v.cpu().clone() for k, v in kwargs["model"].state_dict().items()}
                self.counter = 0

            return control

        def load_best_model(self, model):
            if self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)

    early_stopping = EarlyStoppingCallbackCustom(patience=3,delta=0.0)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    # Add the epoch-eval callback so train+val get evaluated and printed each epoch
    epoch_cb = EpochEvalCallback(train_dataset=train_dataset, val_dataset=eval_dataset)
    # Add after trainer is created so we can set a direct back-reference
    trainer.add_callback(epoch_cb)
    epoch_cb.trainer_ref = trainer

    return trainer


# Monkey-patch the BART model forward to support optional `tda` argument.
# This keeps the rest of your trainer/code the same: trainer will pass batch
# dict items to model.forward; if 'tda' exists in the batch the model will use it.
# We only monkey-patch if not already patched.
_original_bart_forward = getattr(BartForConditionalGeneration, "forward", None)


def _patched_bart_forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, **kwargs):
    """
    Accepts optional 'tda' in kwargs (shape (B, tda_dim) list/tensor).
    If present and model has attribute tda_proj, project and add to encoder hidden states.
    Otherwise falls back to original forward behavior.
    """
    # Extract TDA if present
    tda = kwargs.pop("tda", None)

# Remove Trainer-internal keys that BART forward does not accept
    kwargs.pop("num_items_in_batch", None)


    # If the model was not configured to use TDA, just call original forward
    if (not getattr(self, "use_tda", False)) or (tda is None) or (getattr(self, "tda_proj", None) is None):
        # call original HF forward (preserves original signature)
        return _original_bart_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs,
        )

    # At this point: model.use_tda == True and tda provided.
    # Ensure tda is a torch tensor on model device
    if not isinstance(tda, torch.Tensor):
        tda = torch.tensor(tda, dtype=torch.float32, device=self.device if hasattr(self, "device") else next(self.parameters()).device)
    else:
        tda = tda.to(next(self.parameters()).device).float()

    # Run encoder to get encoder outputs
    encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    enc_hid = encoder_outputs.last_hidden_state  # (B, seq_len, d_model)

    # Project tda to d_model and add (broadcast across seq_len)
    proj = self.tda_proj(tda)  # (B, d_model)
    proj = proj.unsqueeze(1)  # (B, 1, d_model)
    enc_hid = enc_hid + proj  # broadcast add

    # Replace encoder_outputs.last_hidden_state with modified enc_hid.
    # encoder_outputs is a BaseModelOutput; it is ok to set the attribute.
    encoder_outputs.last_hidden_state = enc_hid

    # Call the original model forward but supply encoder_outputs so decoder uses it.
    return _original_bart_forward(
        self,
        input_ids=None,  # encoder already run, pass None for input_ids
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
        encoder_outputs=encoder_outputs,
        **kwargs,
    )


# Apply the patch once
if _original_bart_forward is not None and getattr(BartForConditionalGeneration, "forward", None) is not _patched_bart_forward:
    BartForConditionalGeneration.forward = _patched_bart_forward
