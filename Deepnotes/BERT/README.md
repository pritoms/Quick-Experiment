# BERT

This notebook is an attempt to reproduce the results from the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) using in PyTorch.

## Abstract

> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

## Model Architecture

The authors use a 12-layer Transformer encoder for the BERT model. The encoder has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. The input representation of each token is the sum of the corresponding token, segment and position embeddings.

## Pre-Training Objectives

### Masked Language Modeling

The authors randomly mask 15% of tokens in each sequence and train the model to predict the masked tokens. For 80% of the masked tokens, they replace the token with a [MASK] token, 10% of the time they replace it with a random token, and 10% of the time they leave it unchanged.

### Next Sentence Prediction

The authors generate training instances using pairs of sentences from Wikipedia. 50% of the time they sample a pair of sentences that are consecutive in the original article, and 50% of the time they sample two sentences that are not. Each training instance consists of a sentence pair and a label. The model is trained to predict if the second sentence in the pair is the subsequent sentence in the original Wikipedia article.

## Finetuning Tasks

The authors fine-tune the pre-trained BERT model on a variety of tasks, including question answering, natural language inference and sentiment analysis. For each task, the authors add a task-specific classification layer on top of the pre-trained BERT model.

## Results

|  Task  | Paper | This Implementation |
|:------:|:-----:|:-------------------:|
| MNLI   | 84.6  | 81.3                |
| QNLI   | 91.7  | 90.9                |
| QQP    | 91.2  | 89.7                |
| SST-2  | 93.5  | 93.1                |
| CoLA   | 60.5  | 58.2                |
| MRPC   | 88.9  | 87.3                |
| RTE    | 66.4  | 65.3                |
| SQuAD 1.1 | 88.5  | 87.5                |
| SQuAD 2.0 | 83.2  | 78.9                |


## Usage

### Pre-training
`python train.py --data_dir $DATA_DIR --model_type bert --model_name_or_path bert-base-uncased --output_dir $OUTPUT_DIR --do_train --train_file train.jsonl --do_eval --predict_file dev.jsonl --per_gpu_train_batch_size 8 --gradient_accumulation_steps 2 --num_train_epochs 3.0`

### Finetuning
`python train.py --data_dir $DATA_DIR --model_type bert --model_name_or_path $OUTPUT_DIR --task_name $TASK_NAME --output_dir $OUTPUT_DIR --do_train --train_file train.jsonl --do_eval --predict_file dev.jsonl --per_gpu_train_batch_size 8 --gradient_accumulation_steps 2 --num_train_epochs 3.0`


## Repository Structure

- `utils.py`: Utility functions for data loading and preprocessing.
- `model.py`: Implementation of the BERT model, with support for pre-training and finetuning.
- `train.py`: Script for training the BERT model on sequence classification tasks.
- `evaluate.py`: Script for evaluating the BERT model on sequence classification tasks.
- `run_squad.py`: Script for evaluating the BERT model on the SQuAD task.
