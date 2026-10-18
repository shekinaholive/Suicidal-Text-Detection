# Clean BERT inference demo

\`app.py\` is an optional research-only refactor of the original \`Gradio_NLP_Preprocess_V2.ipynb\`. The notebook remains available as a historical experiment.

## Prerequisites

Install compatible versions of \`torch\`, \`transformers\`, \`captum\` and \`gradio\`.

Provide the **same fine-tuned BERT checkpoint and tokenizer** used by the original experiment in a local directory named \`saved_model1/\` or set \`NLP_CHECKPOINT_DIR\` to another local folder. The repository **does not include** the model weights, training data or complete training notebook.

\`\`\`bash
python app.py
\`\`\`

If the checkpoint is absent, the app refuses to produce predictions rather than quietly using an unfine-tuned model.

## Behavioral corrections from the notebook

1. **Prediction:** Uses the model's natural \`argmax\` class. The original notebook overrides this with a manually chosen non-suicidal threshold of **0.65** despite not providing a calibration study. This experimental threshold has been **removed from the refactored demo**, not from the historical notebook.
2. **Token explanations:** Calculates Integrated Gradients for the **predicted class**, keeping the highlighted contribution direction consistent with the displayed class. The original notebook computed attribution only for class 0 but varied visualization by a numerical \`confidence\` argument compared against the string \`"suicidal_prob"\`.
3. **Display:** Escapes rendered tokenizer text for HTML safety. Provides both raw softmax scores without describing them as clinical confidence or risk.
4. **Preprocessing:** Uses native BERT tokenization with a 128-token maximum and whitespace trimming. Avoids arbitrary stopword deletion that can alter negation.

## Known unknowns

**Label mapping needs verification.** The original notebook explicitly used **0 → Suicidal, 1 → Non-Suicidal**. This script preserves that assumption; confirmation requires checking the checkpoint's \`id2label\`, original training labels and evaluation outputs. If your trained model uses the reverse order, correct \`CLASS_NAMES\` before use. Model predictions are unavailable until the checkpoint is supplied.

The README's **98%** BERT accuracy is *project-reported*, not independently revalidated here. The comparison experiment (BERT vs XGBoost) is described in documentation but the current repository does not contain a reproducible XGBoost pipeline.

This classifier must **not** be used for diagnosis, triage, automatic moderation or decisions about real people. When someone may be in distress, rely on qualified human support, not the demo.
