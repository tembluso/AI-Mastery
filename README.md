# 🧠 AI Mastery Journey

This repository documents my **20-week self-designed AI Mastery course**, built week by week with daily exercises and projects.  

The goal: **learn AI from scratch by building, experimenting, and documenting everything**.  
Each week has 6 days of learning (notebooks) + 1 final project (Day 7).  

---

## 📂 Repository Structure

```txt
AI-Mastery/
├─ Week1/
│  ├─ Day1.ipynb
│  ├─ Day2.ipynb
│  ├─ Day6.ipynb
│  └─ Day7-Project/
│     ├─ train_and_save.py
│     ├─ app.py
│     └─ README.md   (optional, explains this project)
├─ Week2/
│  └─ Day1.ipynb
└─ Week20/
```


- **Daily notebooks (Day1–Day6):** step-by-step exercises, notes, and experiments.  
- **Day 7 Project:** a small end-to-end project for the week, applying everything learned.  
- **Notes:** sometimes you’ll find `.txt` files with my reflections or explanations. These are part of the journey.  

---

## ✅ Completed Projects

- **Week 1 Project — Titanic Survival Predictor**  
  Logistic Regression model trained on the Titanic dataset, wrapped in a Streamlit web app.  
  - Preprocessing pipeline with `scikit-learn`  
  - Logistic Regression classifier  
  - Streamlit app for interactive predictions 

- **Week 2 Project — Model Playground — California Housing** 
  
  An interactive **Streamlit app** to compare machine learning models on the California Housing dataset.  
  - Train & evaluate models with adjustable hyperparameters.
  - See test metrics: **R²** and **RMSE**.
  - Inspect **feature importances** (trees) or **coefficients** (linear regression).
  - Side-by-side **model comparison**.
  - Make **custom predictions** with input values.
  - Visualize **learning curves** to diagnose underfitting/overfitting.

- **Week 3 Project — SMS Spam Classifier** 
  
  An interactive **Streamlit app** that classifies SMS messages as **Spam** or **Ham** (not spam).  
  - Built with **Naive Bayes + TF-IDF**
  - Enter your own SMS text or choose a sample message.
  - Adjustable **decision threshold** to trade off **precision vs recall**.
  - Displays **spam probability** and model prediction.
  - Shows **top spammy words** learned by the Naive Bayes model.

- **Week 4 Project — Movie Recommender 🎬** 
  
  A small, ship‑able item/user‑based collaborative filtering app on MovieLens 100k.
  - Item‑based or user‑based cosine similarity.
  - Filter by minimum number of ratings per movie.
  - Option to exclude already‑watched items.
  - Optional minimum release year filter.
  - Cached data loading and similarity computation for speed.


- **Week 5 Project — Digit Recognizer** 
  
  This is the mini-project app for drawing a digit and getting a prediction using a simple PyTorch MLP.
  - Drawable canvas (280×280) with adjustable stroke width and color.
  - Preprocessing to 28×28 grayscale (MNIST format) with inversion and normalization.
  - PyTorch MLP (256→128) with ReLU, trained with Adam.
  - Top-3 probability display and visualization of the processed input.

- **Week 6 Project — Fashion Recognizer** 
  
  This is the mini-project app where you can train a Fashion-MNIST classifier, watch live metrics,
  browse predictions, and inspect a confusion matrix.
  - Sidebar controls: optimizer (Adam / SGD+Momentum), learning rate, weight decay, dropout, hidden sizes, epochs, batch size.
  - Live training with progress and plots (loss + validation accuracy).
  - Prediction browser: grid of random test images with true/pred labels and **top-3 probabilities**; toggle to show only misclassifications.
  - Confusion matrix tab for full test set.

- **Week 7 Project — CIFAR‑10 Mini‑App** 
  
  Train a baseline CNN on CIFAR‑10, save the best checkpoint, and ship a small demo app (Streamlit/CLI) to classify uploaded images.
  - Loads a 3‑block CNN trained on CIFAR‑10 (32×32 color images, 10 classes).
  - Preprocessing: resize + normalize to match training pipeline.
  - **Top‑K predictions** with probabilities (adjustable in sidebar).
  - Upload support for PNG/JPG/JPEG/WEBP formats.
  - Probability bar chart for visual feedback.
  - CLI script for quick classification outside Streamlit.

- **Week 8 Project — Transfer Learning Demo** 
  
  Fine-tune a pretrained ResNet18 on a small image classification dataset and ship a demo app with explainability.
  - Loads a fine-tuned **ResNet18** checkpoint.
  - Upload support for `.png`, `.jpg`, `.jpeg`, and `.webp` images.
  - **Top-3 predictions** with probabilities.
  - **Grad-CAM heatmap** overlay to show where the model is focusing.
  - Example failure-case analysis, such as background-driven misclassification.

- **Week 9 Project — IMDB Sentiment Baseline** 
  
  Train a baseline LSTM text classifier on IMDB reviews and ship it with both CLI and Streamlit inference.
  - Loads and preprocesses the IMDB dataset.
  - Trains a 1-layer **LSTM** sentiment classifier.
  - Saves model weights, vocabulary, and config in a single `bundle.pt`.
  - CLI script for quick sentiment prediction.
  - Streamlit app with single-review and batch-analysis tabs.

- **Week 10 Project — IMDB Sentiment App** 
  
  Upgrade the IMDB sentiment model using a BiLSTM + Attention architecture and token-level interpretability.
  - Trains a **BiLSTM + Attention** model on an IMDB subset.
  - Predicts positive/negative sentiment with probabilities.
  - Highlights influential words using **Integrated Gradients**.
  - CLI script for batch predictions.
  - Streamlit app for interactive review analysis.

- **Week 11 Project — Attention Visualizer** 
  
  Build a didactic Streamlit tool to visualize how self-attention works across different heads and masks.
  - Input support for numbers or words.
  - Adjustable `d_model`, number of heads, and random seed.
  - Mask options: none, padding, and causal.
  - Per-head and averaged attention heatmaps.
  - Matrix inspection for Q, K, V, and output representations.

- **Week 12 Project — Mini Transformer Demo** 
  
  Train a small encoder-decoder Transformer on a reverse-sequence task and visualize decoder cross-attention.
  - Train or load a Transformer checkpoint inside the app.
  - Evaluate test accuracy and BLEU-1 score.
  - Custom sequence input with predicted vs reference output.
  - Cross-attention heatmaps showing the learned reverse mapping.
  - Default architecture: `d_model=64`, 2 layers, 4 heads.

- **Week 13 Project — Mini GPT vs BERT Playground** 
  
  Build a small playground comparing GPT-style text continuation with BERT-style masked word prediction.
  - Uses pretrained **GPT-2** for text continuation.
  - Uses pretrained **BERT** for `[MASK]` token prediction.
  - Streamlit app with adjustable sampling controls.
  - CLI script for quick GPT/BERT tests.
  - No training required; focuses on understanding model behavior.

- **Week 14 Project — Fine-Tuned LLM Demo** 
  
  Build a CPU-friendly app for fine-tuned sentiment classification and optional review-style text generation.
  - Fine-tunes **DistilBERT** for IMDB sentiment classification.
  - Optional **GPT-2** fine-tuning for review-style generation.
  - Simple PyTorch training loops without relying on `Trainer`.
  - Streamlit app for inference and tiny CPU bootstrap training.
  - CLI scripts for sentiment prediction and text generation.

- **Week 15 Project — Mini-RAG Notebook** 
  
  Build a small Retrieval-Augmented Generation pipeline for answering questions from uploaded documents.
  - Ingests PDFs, TXT, and Markdown files.
  - Chunks documents with configurable overlap.
  - Uses SentenceTransformers embeddings and **FAISS** retrieval.
  - Optional CrossEncoder reranking.
  - Streamlit UI with grounded answers and visible source passages.
 
- **Week 16 Project — Custom Q&A Bot** 
  
  Build a fully working Q&A app using TF-IDF retrieval and extractive answering, designed to run quickly on CPU with minimal dependencies.
  - Ingests documents from a local sample document folder.
  - Uses **TF-IDF retrieval** to find relevant text chunks.
  - Generates answers by stitching together the top retrieved chunks.
  - Streamlit frontend for asking questions interactively.
  - No external LLM or API key required.

- **Week 17 Project — GAN & VAE Art Generator** 
  
  Build a lightweight Streamlit app to train and generate MNIST digits using GANs and VAEs.
  - Generate image grids with either **GAN** or **VAE** models.
  - Latent interpolation between generated samples.
  - VAE reconstruction of real digit images.
  - Mini-training and full-training options from the UI.
  - Diversity and sharpness proxy metrics with downloadable generated grids.
 
- **Week 18 Project — LunarLander A2C Mini-App** 
  
  Train a reinforcement learning agent on Gymnasium’s LunarLander environment and replay captured episodes in a Streamlit app.
  - Implements an **Actor-Critic (A2C)** agent in PyTorch.
  - Sidebar controls for episodes, learning rate, hidden size, discount factor, critic coefficient, entropy term, and seed.
  - Live learning curve with episode rewards and moving average.
  - Best-of-N evaluation to capture the strongest landing attempt.
  - Frame slider to replay the episode timeline, including touchdown or crash.
  - Runs with Gymnasium, PyTorch, Matplotlib, Pygame, and Streamlit.

 - **Week 19 Project — Efficiency Showdown**

  Compare three versions of the same MINST neural network to explore CPU deployment efficiency techniques.
  - Benchmarks a baseline teacher model an **int8 quantized** model, and a smaller **distilled student** model.
  - Compares accuracy, model size, and inference latency.
  - streamlit dashboard with metrics and bar charts.
  - Upload your own digit image or select a random test sample.
  - **Top-3 predictions** shown for each model.
  - Uses PyTorch, dynamic quantization, knowledge distillation, and TorchScript export.

- **Week 20 Project — RAG Tweet Assistant** 
  
  Build a Streamlit app that turns personal notes into grounded social posts using a small RAG pipeline.
  - Upload or pre-index notes from PDF, TXT, and Markdown files.
  - Uses **OpenAI embeddings** with a local **FAISS** vector store.
  - Retrieves top-k chunks and generates post variants with OpenAI Chat.
  - UI controls for platform, number of variants, character limit, hashtags, and emojis.
  - Includes an evaluation script for testing the retrieval/generation pipeline.
  - Reads API keys from `.env` or environment variables to avoid committing secrets.

---

## 🎯 Goals of this repo
- Show my **learning journey** transparently (not just polished code).  
- Build a foundation in AI/ML step by step.  
- Keep a record of every exercise, experiment, and project.  

---

## ⚠️ Disclaimer
This repo is a **learning journal**.  
Code may be messy, overly commented, or experimental. That’s intentional as it reflects real learning progress.  
