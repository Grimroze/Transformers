Transformer from Scratch
This project implements the Transformer model architecture from scratch using TensorFlow, based on the seminal paper "Attention Is All You Need". It covers building all core components, including:

Scaled Dot Product Self-Attention

Multi-Head Self-Attention

Encoder and Decoder blocks with residual connections and layer normalization

Positional and token embeddings using Byte Pair Encoding (BPEmb) tokenizer

Masking mechanisms for padding and look-ahead in sequences

Full encoder-decoder Transformer model capable of sequence-to-sequence learning

This implementation is educational and modular, enabling understanding of inner workings through straightforward code and practical examples. It serves as a foundation for experimenting with modern NLP tasks including language modeling, translation, and text generation.

Features
Complete Transformer model from scratch in TensorFlow

Custom implementation of multi-head self-attention and feed-forward layers

Integration of BPEmb subword tokenizer for efficient vectorization

Examples demonstrating masked attention and training preparation

Designed for extensibility and experimentation

Getting Started
Install dependencies: TensorFlow, BPEmb, NumPy

Run notebook cells sequentially to build and test model blocks

Customize hyperparameters and tokenization for different datasets

Extend to your own sequence-to-sequence tasks or integrate with pretrained models

Resources
Original paper: Attention Is All You Need

BPEmb tokenizer: https://bpemb.h-its.org/

Tutorial series: NLP Demystified - Transformer module https://www.nlpdemystified.org/course/transformers