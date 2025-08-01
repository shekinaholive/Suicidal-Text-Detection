# 🧠 Suicide Text Detection using NLP

This project aims to detect suicidal ideation from user-generated text using Natural Language Processing (NLP) techniques. Two models were developed and compared: a traditional machine learning model (XGBoost) and a transformer-based deep learning model (BERT).

## 📂 Project Overview
- **Objective:** Classify texts as suicidal or non-suicidal.
- **Dataset:** SuicideWatch (e.g., Reddit forum posts)
- **Tasks:** Data preprocessing, model training, evaluation, and comparison.

## 🧪 Models Used
- **XGBoost:** With TF-IDF features for a fast and interpretable baseline.
- **BERT:** Fine-tuned transformer model for deeper semantic understanding.

## 🧹 Preprocessing
- Lowercasing, punctuation & stopword removal
- Tokenization (custom for XGBoost, BERT tokenizer for transformer model)

## 📊 Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score (macro avg)
- Visuals: Confusion Matrix, Classification Reports

## ✅ Results
- BERT outperformed traditional models in identifying suicidal intent.
- Demonstrated effectiveness of contextual embeddings in mental health NLP.

## 📌 Disclaimer
This project is for academic and research purposes only. It does not replace professional mental health support.

