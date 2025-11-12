# models/bart_model.py
# BART fine-tuning wrapper using HuggingFace Seq2SeqTrainer

import os
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from evaluate import load as load_metric
from transformers.modeling_outputs import BaseModelOutput
from torch import nn

def prepare_tokenized_datasets(df, cfg):
    tokenizer = BartTokenizerFast.from_pretrained(cfg.BART_MODEL_NAME)
    ds = Dataset.from_pandas(df)
    split = ds.train_test_split(test_size=0.1, seed=cfg.SEED)
    train_ds, test_ds = split["train"], split["test"]

    def preprocess_function(examples):
        inputs = tokenizer(examples[cfg.INPUT_COL], max_length=cfg.MAX_INPUT_LEN, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[cfg.OUTPUT_COL], max_length=cfg.MAX_TARGET_LEN, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    return tokenizer, tokenized_train, tokenized_test, data_collator

def build_model(cfg):
    base = BartForConditionalGeneration.from_pretrained(cfg.BART_MODEL_NAME)
    return base

def get_trainer(model, tokenizer, train_dataset, eval_dataset, data_collator, cfg):
    metric = load_metric("bertscore")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        preds = np.array(preds)
        preds = np.where(preds != -100, preds, pad_id)
        labels = np.array(labels)
        labels = np.where(labels != -100, labels, pad_id)
        preds = preds.astype(np.int64)
        labels = labels.astype(np.int64)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        results = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        return {
            "bertscore_precision": float(np.mean(results["precision"])),
            "bertscore_recall": float(np.mean(results["recall"])),
            "bertscore_f1": float(np.mean(results["f1"])),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=cfg.LEARNING_RATE,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        num_train_epochs=cfg.TRAIN_EPOCHS,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=150,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bertscore_f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer
