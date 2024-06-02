# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/06/02 18:32:53
@Project -> : nlp-qm$
==================================================================================
"""
# preprocess_data.py

from datasets import load_dataset
from transformers import BertTokenizerFast
import torch


def preprocess_function(examples, tokenizer):
  questions = [q.strip() for q in examples["question"]]
  inputs = tokenizer(
    questions,
    examples["context"],
    max_length=384,
    truncation="only_second",
    padding="max_length",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    return_tensors="pt"
  )

  # 添加 start_positions 和 end_positions
  start_positions = []
  end_positions = []

  for i in range(len(examples["answers"]["answer_start"])):
    start_pos = examples["answers"]["answer_start"][i]
    end_pos = start_pos + len(examples["answers"]["text"][i])

    start_positions.append(start_pos)
    end_positions.append(end_pos)

  inputs.update({
    "start_positions": torch.tensor(start_positions, dtype=torch.long),
    "end_positions": torch.tensor(end_positions, dtype=torch.long)
  })
  return inputs


def main():
  # Load SQuAD dataset
  dataset = load_dataset("squad")

  # Initialize tokenizer
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

  # Preprocess dataset
  encoded_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True,
                                remove_columns=dataset["train"].column_names)

  # Save preprocessed data
  encoded_dataset.save_to_disk("encoded_dataset")


if __name__ == "__main__":
  main()


