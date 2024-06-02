# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/06/02 18:33:55
@Project -> : nlp-qm$
==================================================================================
"""
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_from_disk
import torch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 将inputs中的start_positions和end_positions转为模型的输出格式
        start_positions = inputs.pop("start_positions")
        end_positions = inputs.pop("end_positions")

        # 获取模型的输出
        outputs = model(**inputs)

        # 获取开始和结束的logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # 计算损失
        loss_fct = torch.nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        return (total_loss, outputs) if return_outputs else total_loss


def main():
    # Load preprocessed data
    encoded_dataset = load_from_disk("encoded_dataset")

    # Load pre-trained BERT model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Trainer setup
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"]
    )

    # Train model
    trainer.train()

    # Save trained model and tokenizer
    model.save_pretrained("./bert-finetuned-squad")
    tokenizer.save_pretrained("./bert-finetuned-squad")


if __name__ == "__main__":
    main()


