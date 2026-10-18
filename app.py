"""Research-only BERT text classifier and token-attribution Gradio demo.

Load an externally provided fine-tuned checkpoint from saved_model1/.
The class indices below follow the ORIGINAL uploaded notebook and MUST
be checked against the training labels for any replacement checkpoint.

No configurable decision threshold: output is model argmax with both raw
softmax scores. Do not interpret outputs as mental-health risk estimates.
"""
from __future__ import annotations

import html
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("NLP_CHECKPOINT_DIR", "saved_model1"))
# The notebook labels 0 as suicidal and 1 as non-suicidal.
# This mapping cannot be independently verified without training metadata.
CLASS_NAMES = ("Suicidal-language class", "Non-suicidal-language class")


@lru_cache(maxsize=1)
def load_components():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Fine-tuned BERT checkpoint missing at {MODEL_DIR}. "
            "Supply your original model and tokenizer files; no pretrained "
            "base checkpoint is a substitute for your fine-tuned model."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    if int(model.config.num_labels) != 2:
        raise ValueError("Expected exactly two output classes.")
    if not hasattr(model, "bert"):
        raise ValueError(
            "Expected the BERT architecture used by the original notebook; "
            "another architecture needs a different attribution layer."
        )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def render_attributions(tokenizer, scores, token_ids, target_class):
    max_abs = max((abs(float(s)) for s in scores), default=0.0)
    items = []
    for token_id, score in zip(token_ids, scores):
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if token in tokenizer.all_special_tokens:
            continue
        # HTML escape all user-derived display tokens.
        visible = html.escape(token.replace("##", ""))
        if not visible:
            continue
        strength = min(abs(float(score)) / (max_abs + 1e-8), 1.0)
        bg = (
            f"rgba(19, 136, 99, {0.12 + 0.55 * strength:.3f})"
            if score >= 0 else
            f"rgba(193, 69, 71, {0.12 + 0.55 * strength:.3f})"
        )
        items.append(
            f'<span style="display:inline-block;padding:2px 4px;'
            f'margin:2px;background:{bg};border-radius:3px">{visible}</span>'
        )
    title = html.escape(CLASS_NAMES[target_class])
    return (
        f"<p>Integrated Gradients contributions to <strong>{title}</strong>. "
        "Green increases this class's model score relative to the baseline; "
        "red decreases it. Colors do not establish human intent.</p>"
        + "".join(items)
    )


def classify(text):
    if not isinstance(text, str) or not text.strip():
        raise gr.Error("Please enter text for the research demonstration.")
    tokenizer, model, device = load_components()
    encoded = tokenizer(
        text.strip(), return_tensors="pt", truncation=True, max_length=128
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=1)[0]
    prediction = int(probabilities.argmax().item())
    label = CLASS_NAMES[prediction]
    scores = {
        CLASS_NAMES[i]: round(float(probabilities[i].item()), 5)
        for i in range(2)
    }

    def target_score(input_ids, attention_mask, target_idx):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.softmax(outputs.logits, dim=1)[:, target_idx]

    # Original notebook used LayerIntegratedGradients at BERT embeddings.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("A tokenizer pad_token_id is required for IG baselines.")
    baseline = torch.full_like(encoded["input_ids"], pad_id)
    lig = LayerIntegratedGradients(target_score, model.bert.embeddings)
    attributions = lig.attribute(
        inputs=encoded["input_ids"],
        baselines=baseline,
        additional_forward_args=(encoded["attention_mask"], prediction),
        n_steps=24,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
    token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    explanation = render_attributions(tokenizer, token_scores, token_ids, prediction)
    return (
        f"Predicted model class: {label}\n"
        f"Class softmax scores: {scores}\n"
        "Research-only classifier. These scores are not calibrated clinical probabilities.",
        explanation,
    )


if __name__ == "__main__":
    demo = gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=4, label="Input text", placeholder="Enter non-identifying sample text"),
        outputs=[
            gr.Textbox(label="Research classification"),
            gr.HTML(label="Model token explanation"),
        ],
        title="BERT Text Classification — Research Demo",
        description=(
            "Fine-tuned BERT and Captum Integrated Gradients. "
            "Checkpoint not distributed. The original notebook assumes "
            "class 0 = suicidal-language and class 1 = non-suicidal-language; "
            "verify this mapping from training metadata. "
            "Not suitable for clinical risk assessment, emergency screening, "
            "or decisions about individuals."
        ),
        flagging_mode="never",
    )
    demo.launch()
