# 🎵 Music Genre Classification using DistilHuBERT and Hugging Face Transformers

This repository contains a complete implementation of a **music genre classification** pipeline using **encoder-only transformer models** from Hugging Face. The model is trained and evaluated on the [GTZAN dataset](https://huggingface.co/datasets/marsyas/gtzan), a popular benchmark for music genre recognition consisting of 30-second audio clips across 10 music genres.

> 🔍 **Objective**: Automatically classify a given audio clip into one of 10 genres (e.g., pop, metal, jazz) using a pre-trained transformer model fine-tuned for audio classification.

---

## 📚 Background and Motivation

Music genre classification is a classic problem in music information retrieval (MIR) and has numerous applications in recommendation systems, audio indexing, and playlist generation. Traditionally, this task was approached with handcrafted features and shallow classifiers. However, recent advances in **self-supervised learning for audio**, particularly with models like **Wav2Vec2**, **HuBERT**, and **DistilHuBERT**, have made it possible to learn meaningful audio representations from raw waveforms.

This project aims to demonstrate:
- How to **fine-tune transformer models** (specifically `ntu-spml/distilhubert`) on a custom audio classification task.
- How to process and normalize audio data using **AutoFeatureExtractor** from Hugging Face.
- How to train, evaluate, and interpret the results using the `Trainer` API and `evaluate` library.

---

## 🧠 Model and Tools

- **Model**: [`ntu-spml/distilhubert`](https://huggingface.co/ntu-spml/distilhubert)
  - A distilled version of HuBERT: lighter and faster, but with comparable performance.
- **Framework**: Hugging Face 🤗 Transformers
- **Dataset**: [GTZAN Music Genre Dataset](https://huggingface.co/datasets/marsyas/gtzan)
- **Training Interface**: Hugging Face `Trainer`
- **Metrics**: Accuracy using 🤗 Evaluate

---

## 🧰 Dependencies

Install the required libraries (we recommend Python 3.9+):

```bash
pip install git+https://github.com/huggingface/transformers
pip install datasets evaluate gradio torchaudio
