# Classify text with BERT

## Learning Objectives

1. Learn how to load a pre-trained BERT model from TensorFlow Hub
2. Learn how to build your own model by combining with a classifier
3. Learn how to train a BERT model by fine-tuning
4. Learn how to save your trained model and use it
5. Learn how to evaluate a text classification model

This lab will show you how to fine-tune BERT to perform sentiment analysis on a dataset of plain-text IMDB movie reviews. In addition to training a model, you will learn how to preprocess text into an appropriate format.

## Before you start

Please ensure you have a GPU (1 x NVIDIA Tesla T4 should be enough) attached to your Notebook instance to ensure that the training doesn't take too long.

## About BERT

[BERT](https://arxiv.org/abs/1810.04805) and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models. The BERT family of models uses the Transformer encoder architecture to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.

BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.