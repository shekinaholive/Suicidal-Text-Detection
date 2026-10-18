# Suicide Risk Text Classification — BERT vs XGBoost

An **academic NLP classification experiment** comparing a traditional **TF-IDF + XGBoost** baseline with a fine-tuned **BERT** classifier for identifying risk-related language in text.

## Approach

- **Baseline:** TF-IDF text features with XGBoost.
- **Transformer:** Fine-tuned BERT for binary text classification.
- **Interpretability:** The Gradio notebook uses **Captum LayerIntegratedGradients** to visualize token contributions.
- **Interface:** Gradio for interactive text inputs and explanations.

These are **two compared classifiers**, not a single BERT–XGBoost hybrid.

## Repository contents

| File | Description |
| --- | --- |
| [Gradio_NLP_Preprocess_V2.ipynb](Gradio_NLP_Preprocess_V2.ipynb) | Loads a saved BERT classifier and tokenizer, preprocesses text, predicts labels, calculates token attribution, and launches Gradio |
| [README.md](README.md) | Scope, setup requirements and limitations |

## Reported results

The associated project resume reports **98% BERT classification accuracy** on approximately **232,000 processed social-media records**. The files currently in this repository **do not contain a complete, independently reproducible training-and-evaluation pipeline**, so this is a *project-reported result*, not a benchmark revalidated by this README. For rigorous evaluation, include a held-out test split, class-specific precision/recall/F1, a confusion matrix and details of data leakage controls.

## Running the Gradio notebook

1. Prepare a compatible Python environment with **torch**, **transformers**, **captum** and **gradio**.
2. Supply the previously fine-tuned model and tokenizer as a local directory named **saved_model1/**, or change both **from_pretrained("saved_model1")** paths.
3. Open and run **Gradio_NLP_Preprocess_V2.ipynb** in Jupyter/Colab. The notebook loads the checkpoint, constructs token-attribution explanations, then launches Gradio.

**Note:** The checkpoint is **not included** here. The notebook alone cannot reproduce inference without it.

## Interpretation and limitations

- Token attribution shows which tokens influence a particular model output; it does **not** establish the writer's intent or provide a clinical explanation.
- The notebook includes a manually adjustable non-suicidal confidence threshold and notes a bias toward non-suicidal predictions. Treat it as an **experimental heuristic**, not a calibrated or validated safety threshold.
- Classification can fail on negation, slang, sarcasm, quoted text, cultural variations and unseen data distributions.
- Do not collect or publish identifiable or sensitive text without appropriate consent and protection.

## Safety and intended use

**Research and demonstration only.** A model label is **not** a diagnosis, risk assessment, crisis triage decision or substitute for human support. Do not deploy for automated decisions about individuals.

---
**Author:** [Shekinah Olive](https://github.com/shekinaholive)
