## Enhancing Small Language Models via Chain-of-Thought Distillation - Thesis Plan

This document is the working outline and content scaffold for the engineering thesis. It captures the full pipeline, known results, repo mappings, and clearly marked placeholders you will fill with figures, analysis outputs, and any missing specifics.


## Front matter

- Title (EN): Enhancing Small Language Models via Chain-of-Thought Distillation
- Title (PL): Udoskonalanie małych modeli językowych poprzez destylację łańcucha myśli
- Author: Kacper Rabczewski
- Supervisor: dr inż. Krzysztof Kluza
- Institution/Department: Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie Wydział Elektrotechniki, Automatyki, Informatyki i Inżynierii Biomedycznej Katedra Informatyki Stosowanej

## Language and style note

- Language: English (B2/C1 level typical of a 4th-year CS/MLE engineer).
- Tone: neutral, objective, and adequate for an engineering thesis; avoid expressive or biased language.
- Vocabulary: prefer common CS/ML terms; avoid rare or flowery words; define acronyms on first use and keep a glossary.
- Clarity: avoid long clauses; prefer “use” over “utilize,” “show” over “demonstrate,” etc.
- Evidence and claims: quantify where possible; avoid subjective adjectives without data.
- Naturalness: write like a human; allow moderate variance in sentence and word length; mix simple and compound sentences; avoid repetitive phrasing; keep tone neutral and clear.
- Voice: prefer active voice when it improves clarity; passive is acceptable in methods/results.

## Table of Contents (with recommended page allocation for a 30–40 page thesis)

- Front matter (not counted)
- Abstract (0.5 page)

1. Introduction (3–4 pages)
- Problem, RQs, scope, contributions, motivation

2. Background (5–6 pages)

4. Methodology (6–7 pages total)

6. Results (4–5 pages)

7. Discussion (2–3 pages)
- Interpretation, limitations, implications

8. Conclusion (1–2 pages)
- Summary and practical guidance

9. Ethics, safety, and licensing (0.5–1 page)

10. Reproducibility statement (0.5–1 page)

References (1–2 pages; often excluded from page limit)

Appendices (unlimited; excluded)
- 9. Supplements and confirmed details (map subsections 9.1–9.4 here)
- Additional figures/tables, config exports, repository map, primers


## Abstract

## 1. Introduction

## 2. Background

  - How meaning emerges through context: tokens are mapped to embeddings (average representations from pretraining). As they pass through transformer blocks, self-attention mixes information across positions so embeddings are “saturated” with sentence context. For example, in “Apple phones are good,” the token “Apple” absorbs signal from “phones,” biasing the representation toward the company sense rather than the fruit. After all blocks, the contextualized states drive the next-token prediction.

## 4. Methodology

### 4.1 Problem analysis and system requirements
- Functional goal: improve GSM8K reasoning accuracy of a <4B student via SCoTD; maintain strict output format.
- Constraints: single 24 GB GPU, API-based teacher generation, limited budget; reproducible runs (seed=42).
- Non-functional: training stability, VRAM headroom, data hygiene, cost tracking.

### 4.2 Proposed architecture and data flow
- Components: teacher generator (API) → cleaner → processed dataset → trainer (QLoRA) → evaluator.
- Diagrams: [TO FILL] UML component diagram; system/data-flow diagram for generation→training→eval; training loop schematic.
- Pipeline: Teacher CoT generation → cleaning → processed dataset → SCoTD/label‑only finetuning → evaluation.

### 4.3 Chosen technologies and rationale
- Models: Teacher DeepSeek R1 Distill; Student Qwen2.5‑3B.
- Finetuning: PEFT/LoRA over q/k/v/o + MLP; QLoRA 4‑bit nf4; bf16 compute; gradient checkpointing.
- Runtime: bitsandbytes paged_adamw_8bit; async Python generator via OpenRouter; strict parser for EM.

### 4.4 Teacher CoT dataset generation

- Provider: OpenRouter API (Python async generator with concurrency control; tuned from 100 to 25 to reduce errors/timeouts).
- Teacher model and decoding in `models.py`.
- Prompts: Stored under `prompts/cot/` and `prompts/label_only/`; loaded via `src/config.py`.
- Sampling scale: 30k raw samples generated (30 per question for the first 1k GSM8K train questions).
- Logged metadata per sample: token usage (prompt/completion/total), latency, finish_reason, request_id, correctness.

### 4.5 Cleaning and preprocessing

- Cleaning (see `notebooks/clean_dataset.ipynb`):
  - Keep only correct samples (where teacher numeric answer equals gold numeric answer).
  - Deduplicate by exact `teacher_answer_text`.
  - Remove extreme outliers by total tokens (exclude top 1%).
  - Resulting size: ~25k cleaned samples.
- Processing (see `notebooks/process_dataset.ipynb`): select columns
  - `{question, gold_answer_text, gold_answer_number, teacher_answer_text, teacher_answer_number}`
  - Save JSONL to `artifacts/data/processed/dataset.jsonl`.

### 4.6 Training setups

Common:
- Base model: `Qwen/Qwen2.5-3B` (decoder-only causal LM).
- Tokenizer: fast tokenizer; pad token set to EOS when missing; max seq length 2048; truncation policy keeps full answer and trims prompt from left if needed.
- Quantization: BitsAndBytes 4-bit (nf4), double quantization; bf16 compute.
- Gradient checkpointing enabled; use_cache=False during training.
- Optimizer: paged_adamw_8bit; seeds set to 42.
- Data split: by `question_id`, TRAIN_SPLIT=0.8; dedupe-by-question for label-only mode to avoid multiple labels per question.

SCoTD (train on teacher CoTs):
- LoRA: r=16, alpha=32, dropout=0.05; target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
- Training args (example from notebook): epochs=10, lr=2e-5, per_device_batch=8, grad_accum=2, warmup_ratio=0.03, save/eval every 200 steps, bf16 enabled.

Label-only (train to emit only the final numeric answer string):
- Same LoRA config.
- Training args (example from notebook): epochs=20, lr=1e-5, per_device_batch=16, grad_accum=1, warmup_ratio=0.1, weight_decay=0.1, max_grad_norm=1.0, save/eval every 50 steps.

### 4.7 Inference and evaluation

- Inference prompts: build from the same system/user templates as training; label-only vs CoT modes.
- Decoding: greedy (do_sample=False), max_new_tokens=1024.
- Accuracy metric: numeric exact-match using parser expecting a line like “Final Answer: <number>”. If the format deviates, it is counted as incorrect.
- Teacher evaluation: same generator config; test-set answers collected for accuracy and cost tracking.


## 5. Experimental setup

- Hardware: RunPod RTX 4090 (24GB VRAM), 64GB RAM; price $0.34/hr.
- Software:
  - Python 3.11, uv as package manager.
  - Core libraries: PyTorch, Transformers, PEFT, bitsandbytes, datasets, accelerate, tensorboard, pandas, seaborn, matplotlib. See `pyproject.toml`.
  - Environment variables: `TOKENIZERS_PARALLELISM=false`; seeds set to 42.
  - CUDA/Torch versions: CUDA 12.9, torch 2.8.0
- Data protocol:
  - Teacher CoTs generated for the first ~1,000 training questions of GSM8K.
  - Train/eval split by `question_id` (80/20) for student training.

## 6. Results

### 6.1 Headline accuracies (GSM8K test)

- Base (CoT prompting): 70.51%
- Base (label-only prompting): 10.99%
- Student SCoTD best checkpoint: 78.54% (checkpoint 8)
- Student label-only best checkpoint: 15.01% (checkpoint 4/6/8 similar)
- Teacher: 93.61% (API; cost ~$0.282 for test-set predictions)

### 6.2 Training dynamics

- SCoTD (example eval loss decreasing to ~0.213–0.215 across steps; see TensorBoard).
- Label-only: validation loss decreases until ~200 steps then degrades (overfitting/formatting mismatch).

[TO FILL: Insert training curves from TensorBoard as Figures]

### 6.3 Data and cost analysis

- Raw vs cleaned dataset sizes: 30000 → 24952 samples after correctness filter, dedup, and token outlier removal.

### 6.4 Qualitative analysis

- Examples of correct/incorrect teacher CoTs. [TO FILL: curated examples]
- Label-only formatting pitfalls: instances where the numeric answer is correct but formatting deviates and is counted as incorrect by the strict parser. [TO FILL: 3–5 examples]


## 7. Discussion

- Interpretation of results:
  - SCoTD vs base CoT: Distilling intermediate reasoning improves the small model (+8.03 points in accuracy in these runs).
  - Label-only underperformance largely reflects evaluation format sensitivity; not necessarily a failure to compute the number, but to adhere to strict output format under greedy decoding.
  - Limitations: Single dataset (GSM8K), single small model (Qwen2.5-3B), limited data scale (~25k cleaned), strict evaluation without tolerant parsing.
  - SCoTD student vs base CoT: discuss the observed +8.03 percentage-point gain and what it implies about distilling intermediate reasoning traces into a small model.
  - Label-only underperformance: document that strict formatting in evaluation likely undercounts true capability under greedy decoding; emphasize that the thesis intentionally uses the strict metric without post-hoc fixes.
- Threats to validity: dataset scope and potential contamination, single model family, strict parser sensitivity, hardware/seed dependence.


## 8. Conclusion

- Summary of findings: SCoTD gave a clear improvement over base CoT prompting on GSM8K for a 3B model with modest compute.
- Practical guidance: parameter-efficient finetuning (LoRA + 4-bit) is effective for reasoning enhancement with constrained resources.
- Closing: Explicit prompts, dataset hygiene, and evaluation design are as critical as model choice.


## 9. Supplements and confirmed details

### 9.1 Environment reproduction script

Use the following to recreate the environment and Jupyter kernel (uv-based):

```
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/RabaDaba1/chain-of-thought-distillation.git

cd /workspace/chain-of-thought-distillation/code

uv sync

uv run python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

git config --global user.email "krabczewski@gmail.com"
git config --global user.name "Kacper Rabczewski"
```

Note: Paths may vary; adapt the working directory if cloning elsewhere.

### 9.2 Teacher configuration

- Model: `deepseek/deepseek-r1-distill-qwen-32b`
- System prompt and user-template: see `artifacts/data/raw/dataset_meta.json` and `.../teacher_gsm8k_test_meta.json` (strict format with “Reasoning:” bullet steps and final line `Final Answer: <number>`).
- Decoding and runtime params: `temperature=1.0`, `top_p=1.0`, `max_concurrency=25`, `request_timeout=120`. [Greedy decoding for student/base tests; teacher generation followed the meta config.]
- Scope: First 1,000 questions from GSM8K train set; ~30 samples per question targeted; ~30k raw generations.

### 9.3 Dataset analysis snapshots (train-generation set)

- Per-question sample count: mean=30.0, std=0.0 (30 each over 1,000 questions).
- Unique questions: 1,000
- Unique teacher answers (text): 27,041
- Overall correctness (teacher number vs gold): 93.75%
- Token/latency quantiles (Q1/median/Q3/0.9/0.95/0.99):
  - completion_tokens: 173 / 225 / 304 / 424 / 532 / 862
  - prompt_tokens: 497 / 510 / 524 / 541 / 552.05 / 580.01
  - total_tokens: 679 / 739 / 827 / 949 / 1056 / 1392
  - latency_ms: 4348 / 5435 / 7060.25 / 9533.30 / 11714.15 / 18487.13

### 9.4 Cost snapshot (train-generation set)

- Total cost: $6.33240
- Average cost per query: $0.00021 (median $0.00020, std $0.000062)
- Pricing basis: placeholders in analysis; final per-1M input/output prices to be inserted in Appendix C.
- Pod rent $0.34/hr, with $20 invested

## References

Maintain the bibliography in `BIBLIOGRAPHY.bib` and cite the following canonical works (keys shown):

- Chain-of-Thought prompting: `wei2022chainofthought`
- Self-consistency for CoT: `wang2022selfconsistency`
- LoRA: `hu2021lora`
- QLoRA and 4-bit quantization: `dettmers2023qlora`
- GSM8K dataset: `cobbe2021gsm8k`
- Transformers library: `wolf2020transformers`
- Tokenization (SentencePiece/BPE): `kudo2018sentencepiece`, `sennrich2016bpe`
- bitsandbytes: `bitsandbytes`
- PEFT: `peft`
- Teacher model card/report (placeholder): `deepseekr1distill` 
- Student base model report (placeholder): `qwen25`


## 10. Ethics, safety, and licensing

- Dataset licensing and usage rights: GSM8K is MIT License.
- API usage and privacy: No sensitive data used; ensure compliance with provider ToS (OpenRouter).
- Chain-of-thought disclosure: CoTs generated for research and not redistributed beyond allowed terms (MIT License)
- Environmental and cost considerations: report token usage and spend; minimize wasteful runs.


## 11. Reproducibility statement

- Seeds: 42 for training and data splitting.
- Versioning: commit hash and `uv.lock` for dependencies; Python 3.11.
- Hardware notes: GPU class (RTX 4090 24GB) sufficient for LoRA + 4-bit.
- Artifacts: checkpoints under `artifacts/models/...`; predictions CSVs; meta JSONs for teacher calls.
- Exact prompts and configs archived in repo (see Appendices).


## 12. Acronyms and glossary [TO FILL]

- LLM - Large Language Model
- CoT - Chain of Thought
- SCoTD - Supervised Chain-of-Thought Distillation
- LoRA - Low-Rank Adaptation
- QLoRA - Quantized LoRA
- nf4 - NormalFloat4 (4-bit quantization format)
- PEFT - Parameter-Efficient Fine-Tuning
- BF16 - bfloat16 (reduced-precision floating point with wide exponent; stable training vs FP16)
- BPE - Byte Pair Encoding
- Qwen2.5 - Qwen/Qwen2.5 family of decoder-only LLMs
- DeepSeek R1 Distill - DeepSeek distilled reasoning model series (teacher)

## Appendices

### A. Prompts (verbatim)

- CoT system prompt: from `prompts/cot/system.txt`.
- CoT user prompt: from `prompts/cot/user.txt`.
- Label-only system prompt: from `prompts/label_only/system.txt`.
- Label-only user prompt: from `prompts/label_only/user.txt`.

Include sanitized examples as needed.

### B. Configuration summaries

- Training configs (SCoTD, label-only): learning rates, epochs, batch sizes, grad accumulation, warmup, weight decay, saving/eval intervals.
- LoRA config: r=16, alpha=32, dropout=0.05; targeted modules.
- Quantization: 4-bit nf4, double quant, bf16 compute.
- Seeds and environment variables.

[TO FILL: Exported JSON/YAML of final configs]

### C. Pricing basis and API cost tables
- Final per‑1M input/output prices and teacher API pricing details used for Sections 9.3–9.4.

### D. Reproducibility map to repository

- Data generation: `src/dataset_generator/lib/teacher_client.py` (API), `src/dataset_generator/io/jsonl.py`, `src/dataset_generator/io/meta.py`, `src/dataset_generator/helpers/metrics.py`, `src/dataset_generator/helpers/answers.py`.
- Cleaning: `notebooks/clean_dataset.ipynb`.
- Processing: `notebooks/process_dataset.ipynb`.
- Training: `notebooks/train_student_sctod.ipynb`, `notebooks/train_student_label_only.ipynb`.
- Benchmarking and exports: `notebooks/benchmark.ipynb`; `predictions.csv`, `best_models_predictions.csv`.
- Configuration and prompts: `src/config.py`, `prompts/`.

### E. Figures and tables index [TO FILL]

- Figure 1: Pipeline overview.
- Figure 2–3: Training loss/validation loss (SCoTD, label-only).
- Figure 4: Accuracy per checkpoint (both modes).
- Figure 5–7: Distributions (answer length, completion tokens, latency).
- Figure 8: Cost per query distribution.
- Table 1: Hyperparameters per mode.
- Table 2: Dataset stats before/after cleaning.
- Table 3: Accuracy summary (base vs student SCoTD vs label-only vs teacher).

### F. Technology primers (explanatory)

- QLoRA with 4-bit nf4:
  - Load base weights quantized to 4-bit nf4 (double quantization) with bf16 compute; train low-rank adapters (LoRA) on targeted modules. Preserves memory while retaining headroom to learn reasoning traces.
- Tokenizers (SentencePiece/BPE):
  - Subword tokenization balances vocabulary size with coverage; affects sequence length, truncation, and numeric formatting behaviors.
- bitsandbytes:
  - Provides 4/8-bit quantization kernels and paged optimizers (e.g., paged_adamw_8bit) reducing VRAM and host RAM usage.
- PEFT:
  - Library for parameter-efficient strategies (LoRA et al.); enables injecting adapters into q/k/v/o and MLP projections.
- DeepSeek R1 Distill (teacher):
  - `deepseek/deepseek-r1-distill-qwen-32b` used via OpenRouter; strong reasoning CoT generator under structured prompts.
- Qwen 2.5 (student base):
  - `Qwen/Qwen2.5-3B` decoder-only model; suitable for LoRA + 4-bit finetuning on a single 24GB GPU.
- BF16 and bf16 training:
  - bfloat16 compute offers wider dynamic range vs FP16, improving stability; used for forward/backward passes with quantized base weights.

## Optional repository polish (non-blocking) [TO FILL if desired]

- Add an `artifacts/configs/` folder with exported training/inference configs and prompt snapshots for archival.
- A short `README` in `notebooks/` describing the run order and expected outputs.
