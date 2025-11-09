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

## Table of Contents (30–40 page thesis)

- Front matter (not counted)
- Abstract (0.5 page)

1. Introduction (3–4 pages)

2. Background (5–6 pages)

4. Methodology (6–7 pages total)

6. Results (4–5 pages)

7. Discussion (2–3 pages)
- Interpretation, limitations, implications

8. Conclusion (1–2 pages)
- Summary and practical guidance

9. Ethics, safety, and licensing (0.5–1 page)

References (1–2 pages; often excluded from page limit)

## Abstract

## 1. Introduction

## 2. Background

## 4. Methodology

## 6. Results

### 6.1 Headline accuracies (GSM8K test)

- Base (CoT prompting): 70.51%
- Base (label-only prompting): 10.99%
- Student SCoTD best checkpoint: 78.54% (checkpoint 8)
- Student label-only best checkpoint: 15.01% (checkpoint 4)
- Teacher: 93.61% (API; cost ~$0.282 for test-set predictions)

student_label_only_checkpoint_200: 15.01%
student_sctod_checkpoint_1400: 78.54%
base_cot_prompting: 70.51%
base_label_only: 10.99%


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
- Scope: First 1,000 questions from GSM8K train set; ~30 samples per question targeted; 30k raw generations.

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

Maintain the bibliography in `bibliografia.bib`. Cite using the following keys (complete set from the current .bib):

- li2024symbolicchainofthoughtdistillationsmall
- wei2023chainofthoughtpromptingelicitsreasoning
- wang2023selfconsistencyimproveschainthought
- hu2021loralowrankadaptationlarge
- dettmers2023qloraefficientfinetuningquantized
- cobbe2021trainingverifierssolvemath
- wolf2020huggingfacestransformersstateoftheartnatural
- kudo2018sentencepiecesimplelanguageindependent
- sennrich2016neuralmachinetranslationrare
- bitsandbytes
- peft
- lhoest2021datasetscommunitylibrarynatural
- Ansel_PyTorch_2_Faster_2024
- accelerate
- openrouter
- runpod
- The_pandas_development_team_pandas-dev_pandas_Pandas
- Hunter_Matplotlib_A_2D_2007
- Waskom2021
- kluyver2016jupyter
- uv
- tqdm
- deepseekai2025deepseekr1incentivizingreasoningcapability
- qwen2025qwen25technicalreport
- vaswani2023attentionneed
- brown2020languagemodelsfewshotlearners
- kaplan2020scalinglawsneurallanguage
- hoffmann2022trainingcomputeoptimallargelanguage
- ouyang2022traininglanguagemodelsfollow
- hinton2015distilling
- zagoruyko2017paying
- furlanello2018bornagain
- xie2020noisystudent
- kim2016sequence
- gou2021survey
- houlsby2019parameter
- pfeiffer2020adapterfusion
- li2021prefixtuning
- lester2021power
- nvidia2024qlora

## Ethics, safety, and licensing

- Dataset licensing and usage rights: GSM8K is MIT License.
- API usage and privacy: No sensitive data used; ensure compliance with provider ToS (OpenRouter).
- Chain-of-thought disclosure: CoTs generated for research and not redistributed beyond allowed terms (MIT License)
- Environmental and cost considerations: report token usage and spend; minimize wasteful runs.

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

### A. Prompts

- CoT system prompt: from `prompts/cot/system.txt`.
- CoT user prompt: from `prompts/cot/user.txt`.
- Label-only system prompt: from `prompts/label_only/system.txt`.
- Label-only user prompt: from `prompts/label_only/user.txt`.

### B. Configuration summaries

- Training configs (SCoTD, label-only): learning rates, epochs, batch sizes, grad accumulation, warmup, weight decay, saving/eval intervals.
- LoRA config: r=16, alpha=32, dropout=0.05; targeted modules.
- Quantization: 4-bit nf4, double quant, bf16 compute.
- Seeds and environment variables.

### C. Pricing basis and API cost tables
- Final per‑1M input/output prices and teacher API pricing details used for Sections 9.3–9.4.

### D. Reproducibility map to repository

- Data generation: `src/dataset_generator/lib/teacher_client.py` (API), `src/dataset_generator/io/jsonl.py`, `src/dataset_generator/io/meta.py`, `src/dataset_generator/helpers/metrics.py`, `src/dataset_generator/helpers/answers.py`.
- Cleaning: `notebooks/clean_dataset.ipynb`.
- Processing: `notebooks/process_dataset.ipynb`.
- Training: `notebooks/train_student_sctod.ipynb`, `notebooks/train_student_label_only.ipynb`.
- Benchmarking and exports: `notebooks/benchmark.ipynb`; `predictions.csv`, `best_models_predictions.csv`.
- Configuration and prompts: `src/config.py`, `prompts/`.

### E. Figures and tables index
- Figure 1: Pipeline overview.
- Figure 2: Label-only training loss, learning rate, grad norm, eval loss curves.
- Figure 3: SCoTD training loss, learning rate, grad norm, eval loss curves.
- Figure 4: Accuracy per checkpoint (SCoTD vs label-only vs baselines).
- Figure 5: Distribution of reasoning step counts (correct vs incorrect teacher samples).
- Figure 6: Latency distribution (ms) by correctness.
- Figure 7: Cost per query distribution (USD) by correctness.
- Figure 8: Answer length (characters) distribution (correct vs incorrect).
- Figure 9: Completion token length distribution (correct vs incorrect).
- Table 1: Hyperparameters per mode.
- Table 2: Dataset stats before/after cleaning.
- Table 3: Accuracy summary (base vs student SCoTD vs label-only vs teacher).

## Optional repository polish (non-blocking) [TO FILL if desired]

- Add an `artifacts/configs/` folder with exported training/inference configs and prompt snapshots for archival.
- A short `README` in `notebooks/` describing the run order and expected outputs.
