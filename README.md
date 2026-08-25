# BENTO

**BEN**chmarking **TO**kens — compare language models on MMLU-style multiple-choice questions while measuring how many tokens each answer actually costs.

The repo wires up API models (Gemini, GPT, Claude) and open models (Llama, BERT), fine-tunes a few of them on auxiliary multiple-choice data, then records **input**, **output**, and **cache** token counts for each run.

## Why this exists

Accuracy alone hides the bill. Two models can get the same MMLU item right while using very different context, generation, and KV-cache sizes. BENTO is set up to:

1. Ask the same multiple-choice questions across providers.
2. Fine-tune smaller open models when few-shot prompting is not enough.
3. Log token usage (`t_in`, `t_out`, estimated cache) next to the prediction.

## Repository layout

```
bento/
├── README.md
├── models/
│   ├── models.ipynb                 # API + Hugging Face client setup
│   ├── all-model-mmlu-setup.ipynb   # Llama 3.1 8B LoRA fine-tune + token counts (CUDA / Unsloth)
│   └── mmlu-fine-tuned.ipynb        # BERT multiple-choice fine-tune on ARC-Easy
└── data/                            # MMLU splits (see below; add locally if missing)
    ├── README.txt
    ├── dev/                         # few-shot exemplars
    ├── val/
    ├── test/                        # evaluation questions
    └── auxiliary_train/             # extra MC datasets for fine-tuning
```

A macOS/Apple Silicon variant of the Llama notebook lives at `(macos)all-model-mmlu-setup.ipynb`. It drops Unsloth, uses PyTorch MPS, and fine-tunes with PEFT LoRA.

## Models

| Notebook | Models | Role |
| --- | --- | --- |
| `models/models.ipynb` | Gemini 2.5 Flash, GPT, Claude Sonnet, Llama 2 7B, BERT | Prompt / API setup |
| `models/all-model-mmlu-setup.ipynb` | `unsloth/Meta-Llama-3.1-8B-Instruct` (4-bit) | SFT on ARC-Easy, then inference + token stats |
| `models/mmlu-fine-tuned.ipynb` | `bert-base-uncased` | Multiple-choice classification on ARC-Easy |
| `(macos)all-model-mmlu-setup.ipynb` | `meta-llama/Llama-2-7b-hf` | Same Llama flow on Apple Silicon |

## Data

Questions follow the [MMLU](https://github.com/hendrycks/test) format: a stem, choices A–D, and a letter answer. Knowledge cutoff for the original test is **January 1, 2020**.

| Split | Use |
| --- | --- |
| `data/dev/` | Few-shot priming |
| `data/val/` | Hyperparameter / checkpoint selection |
| `data/test/` | Held-out evaluation |
| `data/auxiliary_train/` | Fine-tuning for models that cannot do few-shot well |

Auxiliary files come from other multiple-choice NLP sets (MCTest, RACE, ARC, OBQA). The Llama notebooks currently train on **ARC-Easy** (`allenai/ai2_arc` or `data/auxiliary_train/arc_easy.csv`). BERT reads that CSV directly.

`data/auxiliary_train/race.csv` is not in git — it is 147 MB, over GitHub's 100 MB file limit. Keep a local copy if you need RACE for fine-tuning.

## Token metrics

After a generation, `models/all-model-mmlu-setup.ipynb` reports:

- **`t_in`** — tokens in the user question / prompt
- **`t_out`** — tokens in the model completion
- **`t_cache`** — KV-cache size (estimated)
- **`CPR`** — cache / prompt ratio (placeholder in the notebook until filled in)

Use these to compare cost, not just accuracy, across API and local models.

## Setup

Python 3.10+ and Jupyter. Create a virtualenv, then install what your machine can actually run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate evaluate python-dotenv huggingface_hub pandas numpy scikit-learn
```

**NVIDIA GPU (Unsloth path):** also install `unsloth`, `trl`, and a CUDA build of PyTorch. Open `models/all-model-mmlu-setup.ipynb`.

**Apple Silicon:** skip Unsloth. Install `peft` and a Mac PyTorch build with MPS. Open `(macos)all-model-mmlu-setup.ipynb`.

**API clients** (from `models/models.ipynb`): `google-genai`, `openai`, `anthropic`.

Put secrets in a `.env` file in the repo root (already gitignored):

```
HUGGINGFACE_TOKEN=hf_...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Llama checkpoints on Hugging Face are gated. Request access to the model card, then log in with `HUGGINGFACE_TOKEN` before running those notebooks.

## How to run

1. Start Jupyter from the repo root so relative paths like `../data/...` resolve.
2. Run `models/models.ipynb` first to confirm API keys and Hugging Face downloads.
3. Fine-tune BERT with `models/mmlu-fine-tuned.ipynb`, or Llama with the CUDA / macOS notebooks.
4. Point the evaluation cells at `data/test/` (or a single subject CSV) when scoring MMLU.

BERT treats each item as four `(question, choice)` pairs and uses `AutoModelForMultipleChoice`. The Llama notebooks format ARC/MMLU items as chat messages and ask for a single letter `A`–`D`.

## Citation

If you use the MMLU splits, cite the original benchmark:

```bibtex
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
