# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/06/02 18:34:24
@Project -> : nlp-qm$
==================================================================================
"""
from transformers import BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_from_disk

def main():
    # Load preprocessed data
    encoded_dataset = load_from_disk("encoded_dataset")

    # Load fine-tuned BERT model
    model = BertForQuestionAnswering.from_pretrained('./bert-finetuned-squad')

    # Evaluation arguments
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=encoded_dataset["validation"]
    )

    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()
