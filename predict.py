# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/06/02 18:34:48
@Project -> : nlp-qm$
==================================================================================
"""
from transformers import BertTokenizer, BertForQuestionAnswering

def main():
    # Load fine-tuned model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("./bert-finetuned-squad")
    model = BertForQuestionAnswering.from_pretrained("./bert-finetuned-squad")

    # Example question and context
    question = "What is BERT?"
    context = "BERT is a language representation model."

    # Encode input
    inputs = tokenizer(question, context, return_tensors='pt')

    # Model prediction
    outputs = model(**inputs)

    # Extract answer
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start_index:answer_end_index+1]))

    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
