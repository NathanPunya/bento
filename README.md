# BENTO

**BEN**chmarking **TO**kens — compare language models on MMLU-style multiple-choice questions while measuring how many tokens each answer costs.

## Requirements

- Python 3.10+
- Jupyter
- A Hugging Face account and token (Llama checkpoints are gated)
- **Windows / NVIDIA:** CUDA GPU for the Unsloth notebook
- **macOS / Apple Silicon:** MPS (no Unsloth)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install torch transformers datasets accelerate evaluate peft python-dotenv huggingface_hub pandas numpy scikit-learn
```

**Windows / CUDA:** also install `unsloth`, `trl`, and a CUDA build of PyTorch.

**API notebooks:** `python -m pip install google-genai openai anthropic`

Create a `.env` in the repo root:

```
HUGGINGFACE_TOKEN=hf_...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Request access on each gated model card, then run notebooks from the repo root so paths like `../data/...` resolve. In Jupyter, select the `.venv` kernel.

## Run

| Goal | Notebook |
| --- | --- |
| API clients (Gemini, GPT, Claude, Hugging Face) | `models/models.ipynb` |
| BERT multiple-choice fine-tune | `models/mmlu-fine-tuned.ipynb` |
| Llama fine-tune + token counts (CUDA) | `models/(windows)all-model-mmlu-setup.ipynb` |
| Llama fine-tune (Apple Silicon) | `(macos)all-model-mmlu-setup.ipynb` |

1. Start Jupyter in the repo root and pick the venv kernel.
2. Open the notebook for your platform from the table above.
3. Run all cells top to bottom.
4. For MMLU scoring, point evaluation cells at `data/test/` (or a single subject CSV).

BERT scores each item as four `(question, choice)` pairs with `AutoModelForMultipleChoice`. The Llama notebooks ask for a single letter `A`–`D`.

## Data

MMLU-format CSVs: question, choices A–D, letter answer. Original knowledge cutoff is January 1, 2020. See [hendrycks/test](https://github.com/hendrycks/test).

| Path | Use |
| --- | --- |
| `data/dev/` | Few-shot priming |
| `data/val/` | Checkpoint selection |
| `data/test/` | Evaluation |
| `data/auxiliary_train/` | Fine-tuning (ARC, MCTest, OBQA, …) |

Llama notebooks train on **ARC-Easy** (`allenai/ai2_arc` or `data/auxiliary_train/arc_easy.csv`). BERT reads that CSV directly. `race.csv` is omitted from git (over 100 MB).

After a Llama generation, the CUDA notebook reports `t_in`, `t_out`, and estimated cache tokens.

## Citation

```bibtex
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
