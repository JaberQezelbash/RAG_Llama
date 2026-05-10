


# AI RAG Chatbot with Fine-Tuned Data — Llama 3.1 + BGE-M3 + FAISS

**RAG-first customer-support chatbot using [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), FAISS retrieval, and LoRA/QLoRA fine-tuning on multilingual IT support tickets.**



<!-- Optional: add a banner/diagram here -->
<!-- 
<img width="900" alt="banner" src="https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/blob/main/assets/banner.png">
<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.svg">
-->

<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.png">


This repository contains an end-to-end **Retrieval-Augmented Generation (RAG)** project for customer-support automation. The goal is to build a multilingual IT-support chatbot that retrieves relevant solved tickets, uses them as grounding evidence, and then generates a professional support response using a fine-tuned Llama model.

Unlike a simple fine-tuning project, this project intentionally makes **RAG the core component**. The model is not trained only to memorize support answers. Instead, each supervised fine-tuning example is built around retrieved context from a FAISS-based knowledge base, and the validation pipeline also follows the same retrieve → rerank → generate workflow.

> ⚠️ Disclaimer: This project is only for research, education, and portfolio demonstration.
---

## Motivation

Customer-support chatbots are a natural use case for RAG because support answers often depend on historical tickets, product details, troubleshooting patterns, queue metadata, and language-specific context.

Fine-tuning alone can improve style and task-following, but it has limitations:

- it may memorize outdated answers,
- it may hallucinate when it does not know the correct support policy,
- it cannot easily update its knowledge without retraining,
- and it may ignore metadata such as language, priority, queue, and ticket type.

This project addresses those issues by combining:

- **RAG retrieval** for grounding,
- **reranking** for better evidence selection,
- **LoRA/QLoRA fine-tuning** for response style and task adaptation,
- **multilingual ticket data** for English, German, Spanish, French, Portuguese, and related support scenarios,
- and **validation metrics** for both retrieval quality and generated response similarity.

The main learning objective is to build a system where the RAG pipeline is not an add-on, but the central mechanism.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Repository Structure](#repository-structure)
5. [Dataset](#dataset)
6. [Model Stack](#model-stack)
7. [Pipeline Stages](#pipeline-stages)
8. [Requirements](#requirements)
9. [Hugging Face Llama Access](#hugging-face-llama-access)
10. [How to Run](#how-to-run)
11. [Script 1: Token Login](#script-1-token-login)
12. [Script 2: RAG Training Data Builder](#script-2-rag-training-data-builder)
13. [Script 3: Llama Fine-Tuning](#script-3-llama-fine-tuning)
14. [Script 4: RAG Validation](#script-4-rag-validation)
15. [Outputs](#outputs)
16. [Configuration](#configuration)
17. [Evaluation Metrics](#evaluation-metrics)
18. [CPU vs GPU Notes](#cpu-vs-gpu-notes)
19. [Important Implementation Notes](#important-implementation-notes)
20. [Future Improvements](#future-improvements)
21. [Author’s Note](#authors-note)

---

## Project Overview

This project builds and evaluates a RAG-based multilingual customer-support chatbot.

The selected stack is:

- **Dataset:** Customer IT Support / Multilingual Customer Support Tickets
- **Retriever:** [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
- **Reranker:** [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Generator:** [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Fine-tuning method:** LoRA / QLoRA supervised fine-tuning
- **Vector database:** FAISS
- **Training target:** Customer-support response generation grounded in retrieved solved tickets

The workflow is divided into four main scripts:

```text
00_token_login.py
01_build_rag_training_data.py
02_train_llama_lora_or_qlora.py
03_validate_rag_lora_or_qlora.py
```

This separation makes the project easier to debug, reproduce, and extend.

---

## Key Features

✅ RAG-first project design  
✅ Multilingual customer-support dataset  
✅ Train-only FAISS knowledge base to reduce validation leakage  
✅ Dense retrieval using BGE-M3 embeddings  
✅ Reranking during validation using BGE reranker  
✅ RAG-grounded supervised fine-tuning records  
✅ Llama 3.1 8B Instruct generator  
✅ CUDA QLoRA support when GPU is available  
✅ CPU LoRA fallback when CUDA is not available  
✅ Hugging Face token prompt for gated Llama access  
✅ Safe reranker wrapper to avoid tokenizer compatibility issues  
✅ FAISS local vector search  
✅ Saves train/validation splits, corpus, FAISS index, SFT records, adapter, predictions, and metrics  
✅ Includes retrieval-quality metrics and generation-quality metrics  
✅ Designed for Jupyter Notebook or script-based execution  
✅ Modular 4-stage workflow for clean experimentation  

---

## Architecture

The system follows this high-level architecture:

```text
Customer Support Ticket Dataset
        │
        ▼
Data Cleaning + Deduplication
        │
        ▼
Answered Tickets Only
        │
        ▼
Train / Validation Split
        │
        ▼
Train-only Knowledge Base
        │
        ├── BGE-M3 Embeddings
        ├── FAISS Index
        └── Corpus JSONL
        │
        ▼
RAG-SFT Dataset Construction
        │
        ├── Query = subject + body + metadata + tags
        ├── Retrieve similar solved tickets
        ├── Build prompt with retrieved evidence
        └── Completion = original support answer
        │
        ▼
Llama 3.1 Adapter Fine-Tuning
        │
        ├── CUDA available: QLoRA
        └── No CUDA: CPU LoRA fallback
        │
        ▼
Validation
        │
        ├── Retrieve with BGE-M3 + FAISS
        ├── Rerank with BGE reranker
        ├── Generate answer with fine-tuned Llama adapter
        └── Save predictions + metrics
```

The key idea is that the generator is trained and evaluated with retrieved solved-ticket evidence available in the prompt.

---

## Repository Structure

Recommended repository structure:

```text
.
├── codes/
│   ├── 00_token_login.py
│   ├── 01_build_rag_training_data.py
│   ├── 02_train_llama_lora_or_qlora.py
│   └── 03_validate_rag_lora_or_qlora.py
│
├── assets/
│   ├── requirements.txt
│   ├── configurations.md
│   └── rag_architecture.png
│
├── dataset/
│   └── Customer IT Support -Ticket Dataset/
│       ├── aa_dataset-tickets-multi-lang-5-2-50-version.csv
│       ├── dataset-tickets-german_normalized.csv
│       ├── dataset-tickets-german_normalized_50_5_2.csv
│       ├── dataset-tickets-multi-lang3-4k.csv
│       └── dataset-tickets-multi-lang-4-20k.csv
│
├── outputs/
│   ├── processed/
│   ├── rag_store/
│   ├── sft_data/
│   ├── trainer_runs_lora_or_qlora/
│   ├── llama31_8b_rag_lora_or_qlora_adapter/
│   └── validation_results_lora_or_qlora/
│
└── README.md
```

Recommended `.gitignore` entries:

```text
dataset/
outputs/
*.safetensors
*.bin
*.pt
*.pth
*.index
*.npy
__pycache__/
.ipynb_checkpoints/
.env
```

The dataset and outputs are intentionally not committed because they can be large and may have their own license restrictions.

---

## Dataset

The project uses the **Customer IT Support / Multilingual Customer Support Tickets** dataset from Kaggle.

The dataset directory used in development was:

```text
https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?resource=download
```

The dataset contains five CSV files:

```text
aa_dataset-tickets-multi-lang-5-2-50-version.csv
dataset-tickets-german_normalized.csv
dataset-tickets-german_normalized_50_5_2.csv
dataset-tickets-multi-lang3-4k.csv
dataset-tickets-multi-lang-4-20k.csv
```

The main useful files for supervised generator fine-tuning are the files that contain an `answer` column. Rows without an answer can still be useful for future classification or routing tasks, but they are not used for supervised response-generation fine-tuning in the current pipeline.

Common columns include:

```text
subject
body
answer
type
queue
priority
language
business_type
version
tag_1
tag_2
tag_3
tag_4
tag_5
tag_6
tag_7
tag_8
tag_9
```

The script automatically:

1. loads all CSV files,
2. normalizes column names,
3. repairs common mojibake/encoding issues,
4. keeps rows with usable `subject`, `body`, and `answer`,
5. deduplicates rows,
6. assigns internal ticket IDs,
7. creates query text for retrieval,
8. creates context text for the RAG knowledge base,
9. and saves cleaned train/validation splits.

---

## Model Stack

### Generator

```text
meta-llama/Llama-3.1-8B-Instruct
```

The generator is used to produce final support responses.

The script supports:

```text
CUDA available     → QLoRA fine-tuning
CUDA unavailable   → CPU LoRA fallback
```

The same Llama model is used in both cases.

---

### Retriever

```text
BAAI/bge-m3
```

BGE-M3 is used to embed ticket queries and solved-ticket documents.

The FAISS index is built from the **training split only**:

```text
train tickets → BGE-M3 embeddings → FAISS index
```

This prevents validation tickets from being directly inserted into the retrieval knowledge base.

---

### Reranker

```text
BAAI/bge-reranker-v2-m3
```

The validation script uses a manual reranker wrapper based on:

```python
AutoTokenizer
AutoModelForSequenceClassification
```

This avoids a known compatibility issue where some `FlagEmbedding` reranker paths may call tokenizer methods that are unavailable in certain Transformers/tokenizer versions.

---

### Vector Store

```text
FAISS
```

FAISS is used for local dense-vector similarity search.

Saved artifacts include:

```text
outputs/rag_store/faiss_train.index
outputs/rag_store/corpus.jsonl
outputs/rag_store/train_vectors.npy
```

---

## Pipeline Stages

### Stage 1 — Token Login

Script:

```text
00_token_login.py
```

Purpose:

- ask for Hugging Face token,
- authenticate the session,
- verify Llama access.

This is needed because Llama 3.1 is a gated model on Hugging Face.

---

### Stage 2 — RAG Training Data Builder

Script:

```text
01_build_rag_training_data.py
```

Purpose:

- clean dataset,
- split train/validation,
- build train-only FAISS index,
- create RAG-grounded SFT records.

It does **not** load or fine-tune Llama.

---

### Stage 3 — Llama Fine-Tuning

Script:

```text
02_train_llama_lora_or_qlora.py
```

Purpose:

- load existing SFT JSONL files,
- load Llama,
- train LoRA/QLoRA adapter,
- save the adapter.

It does **not** rebuild RAG artifacts.

---

### Stage 4 — Validation

Script:

```text
03_validate_rag_lora_or_qlora.py
```

Purpose:

- retrieve from FAISS,
- rerank retrieved candidates,
- generate answers with fine-tuned Llama adapter,
- save predictions and metrics.

---

## Requirements

Recommended installation:

```bash
pip install -U pandas numpy scikit-learn tqdm ftfy faiss-cpu FlagEmbedding transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf safetensors huggingface_hub
```

Example `assets/requirements.txt`:

```txt
torch>=2.1.0
transformers>=4.40.0
datasets>=2.18.0
peft>=0.10.0
trl>=0.8.0
accelerate>=0.26.0
bitsandbytes>=0.43.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.66.0
ftfy>=6.1.0
faiss-cpu>=1.7.4
FlagEmbedding>=1.2.0
sentencepiece>=0.1.99
protobuf>=4.25.0
safetensors>=0.4.0
huggingface_hub>=0.23.0
```

If `bitsandbytes` causes issues on CPU-only systems, it is still safe to keep installed, but true 4-bit QLoRA requires CUDA.

---

## Hugging Face Llama Access

Because `meta-llama/Llama-3.1-8B-Instruct` is gated, access requires:

1. a Hugging Face account,
2. approval for the Llama 3.1 model,
3. a Hugging Face access token,
4. token login from Jupyter or Python.

The project includes:

```text
00_token_login.py
```

This script prompts for the token securely:

```python
getpass.getpass("Paste your Hugging Face token. It will not be shown: ")
```

Do **not** hard-code the token in the scripts.

Do **not** commit tokens to GitHub.

---

## How to Run

### Step 1 — Log in to Hugging Face

```bash
python codes/00_token_login.py
```

Or run it as a Jupyter cell.

Expected output:

```text
Token saved for this session.
Llama access works.
```

---

### Step 2 — Build RAG training data

```bash
python codes/01_build_rag_training_data.py
```

This creates:

```text
outputs/processed/train.csv
outputs/processed/validation.csv
outputs/rag_store/faiss_train.index
outputs/rag_store/corpus.jsonl
outputs/rag_store/train_vectors.npy
outputs/sft_data/train_sft_prompt_completion.jsonl
outputs/sft_data/validation_sft_prompt_completion.jsonl
outputs/sft_data/train_sft_audit.jsonl
outputs/sft_data/validation_sft_audit.jsonl
```

If these artifacts already exist, the script can skip rebuilding.

---

### Step 3 — Fine-tune Llama

```bash
python codes/02_train_llama_lora_or_qlora.py
```

If CUDA is available:

```text
Training mode: cuda_qlora
```

If CUDA is not available:

```text
Training mode: cpu_lora
```

On CPU, the default configuration uses only a subset of records:

```text
cpu_train_record_limit = 800
cpu_eval_record_limit = 100
cpu_max_seq_length = 1024
```

This is intentional because Llama 3.1 8B CPU training is extremely slow.

---

### Step 4 — Validate the RAG chatbot

```bash
python codes/03_validate_rag_lora_or_qlora.py
```

This creates:

```text
outputs/validation_results_lora_or_qlora/predictions.jsonl
outputs/validation_results_lora_or_qlora/predictions.csv
outputs/validation_results_lora_or_qlora/metrics_summary.json
```

---

## Script 1: Token Login

File:

```text
codes/00_token_login.py
```

Purpose:

- request token,
- log in to Hugging Face,
- check Llama tokenizer access.

This script should be run before fine-tuning or validation.

Main behavior:

```text
User enters token → token stored in session → Llama access tested
```

---

## Script 2: RAG Training Data Builder

File:

```text
codes/01_build_rag_training_data.py
```

This is the core RAG-development script.

It performs:

1. CSV loading
2. text cleaning
3. encoding repair
4. answered-ticket filtering
5. deduplication
6. train/validation split
7. BGE-M3 embedding
8. FAISS index creation
9. corpus JSONL creation
10. RAG-grounded SFT prompt construction

The SFT examples follow this structure:

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a multilingual IT customer-support assistant..."
    },
    {
      "role": "user",
      "content": "Customer ticket to answer...\nRetrieved solved-ticket evidence..."
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "The original support answer..."
    }
  ]
}
```

The retrieved context includes:

```text
Retrieved ticket ID
Subject
Customer issue
Resolved support answer
Type
Queue
Priority
Language
Business type
Tags
Relevance score
```

This means the Llama model is trained to answer with RAG context in the prompt.

---

## Script 3: Llama Fine-Tuning

File:

```text
codes/02_train_llama_lora_or_qlora.py
```

This script only handles model fine-tuning.

It expects these files to already exist:

```text
outputs/sft_data/train_sft_prompt_completion.jsonl
outputs/sft_data/validation_sft_prompt_completion.jsonl
```

If they do not exist, run:

```bash
python codes/01_build_rag_training_data.py
```

The fine-tuning script supports two modes:

### CUDA QLoRA mode

Used when:

```python
torch.cuda.is_available() == True
```

Configuration:

```text
4-bit quantization
NF4 quantization
LoRA adapters
paged_adamw_8bit optimizer
gradient checkpointing
```

### CPU LoRA fallback mode

Used when:

```python
torch.cuda.is_available() == False
```

Configuration:

```text
regular LoRA adapters
no 4-bit CUDA quantization
adamw_torch optimizer
reduced sequence length
limited training records by default
```

The CPU fallback keeps the same Llama model, but it can be very slow.

---

## Script 4: RAG Validation

File:

```text
codes/03_validate_rag_lora_or_qlora.py
```

The validation script evaluates the full RAG pipeline:

```text
validation ticket
    ↓
BGE-M3 query embedding
    ↓
FAISS top-k retrieval
    ↓
BGE reranking
    ↓
RAG prompt construction
    ↓
fine-tuned Llama generation
    ↓
metrics + saved predictions
```

Validation output includes:

```text
customer ticket
gold answer
generated answer
retrieved ticket IDs
retrieval scores
reranker scores
lexical F1
ROUGE-L F1
metadata retrieval metrics
```

---

## Outputs

After running all stages, the output directory should look like:

```text
outputs/
├── processed/
│   ├── combined_answered_tickets.csv
│   ├── train.csv
│   ├── validation.csv
│   └── csv_file_summary.json
│
├── rag_store/
│   ├── faiss_train.index
│   ├── corpus.jsonl
│   └── train_vectors.npy
│
├── sft_data/
│   ├── train_sft_prompt_completion.jsonl
│   ├── validation_sft_prompt_completion.jsonl
│   ├── train_sft_audit.jsonl
│   └── validation_sft_audit.jsonl
│
├── trainer_runs_lora_or_qlora/
│   └── checkpoint-*/
│
├── llama31_8b_rag_lora_or_qlora_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   └── tokenizer files...
│
└── validation_results_lora_or_qlora/
    ├── predictions.jsonl
    ├── predictions.csv
    ├── metrics_summary.json
    └── config_validate_lora_or_qlora.json
```

---

## Configuration

Important settings from the RAG-building script:

```python
faiss_candidate_k = 16
sft_context_top_k = 3
embedding_batch_size = 16
embedding_max_length = 8192
max_context_chars_per_ticket = 950
max_query_chars = 3500
```

Important settings from the fine-tuning script:

```python
generator_model_id = "meta-llama/Llama-3.1-8B-Instruct"

max_seq_length = 3072
num_train_epochs = 1.0
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
```

CPU fallback settings:

```python
cpu_train_record_limit = 800
cpu_eval_record_limit = 100
cpu_max_seq_length = 1024
cpu_torch_dtype = "float16"
```

Important settings from the validation script:

```python
max_validation_examples = 100
faiss_candidate_k = 30
rerank_top_k = 5
max_prompt_tokens = 4096
max_new_tokens = 384
```

---

## Evaluation Metrics

The validation script produces both generation and retrieval metrics.

### Generation metrics

```text
lexical_f1
rouge_l_f1
```

These compare the generated answer with the dataset’s reference support answer.

### Retrieval metadata metrics

```text
top1_same_queue
top1_same_language
topk_same_queue
topk_same_language
topk_tag_overlap
best_faiss_score
best_rerank_score
```

These measure whether the retrieved tickets are similar in metadata to the validation ticket.

The metrics are saved in:

```text
outputs/validation_results_lora_or_qlora/metrics_summary.json
```

Example structure:

```json
{
  "n": 100,
  "mean_lexical_f1": 0.32,
  "mean_rouge_l_f1": 0.28,
  "mean_top1_same_queue": 0.61,
  "mean_top1_same_language": 0.89,
  "mean_topk_same_queue": 0.78,
  "mean_topk_same_language": 0.95
}
```

The exact numbers will depend on hardware, configuration, training records, and generation settings.

---

## CPU vs GPU Notes

This project was designed to support both GPU and CPU environments.

### If CUDA is available

The training script uses QLoRA:

```text
4-bit quantization
LoRA adapters
paged_adamw_8bit optimizer
larger sequence length
faster training
```

This is the preferred mode.

### If CUDA is not available

The training script falls back to CPU LoRA:

```text
same Llama model
regular LoRA adapters
no CUDA 4-bit quantization
smaller sequence length
limited training examples by default
very slow training
```

For CPU-only systems, the default configuration is intentionally conservative:

```python
cpu_train_record_limit = 800
cpu_eval_record_limit = 100
cpu_max_seq_length = 1024
```

This avoids trying to train on all 35k+ records on CPU, which could take many days or longer.

---

## Important Implementation Notes

### 1. Train-only retrieval index

The FAISS index is built only from the training split:

```text
train.csv → BGE-M3 embeddings → FAISS
```

Validation tickets are not inserted into the knowledge base.

This reduces direct validation leakage.

---

### 2. RAG is used during SFT data creation

Each training example contains retrieved solved-ticket evidence.

The model sees prompts like:

```text
Customer ticket to answer:
Subject: ...
Message: ...

Ticket metadata:
type=...; queue=...; priority=...; language=...; tags=...

Retrieved solved-ticket evidence:
[Retrieved ticket 1]
...
[Retrieved ticket 2]
...
[Retrieved ticket 3]
...

Task:
Draft the best customer-support response grounded in the retrieved evidence.
```

The target completion is the original support answer.

---

### 3. RAG is also used during validation

Validation does not simply ask the fine-tuned model to answer directly.

It performs:

```text
retrieve → rerank → generate
```

This ensures the final project evaluates the RAG system, not only the fine-tuned model.

---

### 4. Reranker wrapper avoids tokenizer compatibility errors

The validation script uses:

```python
AutoTokenizer
AutoModelForSequenceClassification
```

instead of relying directly on the `FlagReranker` inference path.

This avoids compatibility issues such as:

```text
AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model
```

---

### 5. Hugging Face token is not hard-coded

The project uses:

```python
getpass.getpass(...)
```

to ask for the token securely.

Do not commit tokens.

Do not paste tokens into README files, notebooks, GitHub issues, or public logs.

---

### 6. Dataset encoding repair

Some dataset snippets may contain mojibake such as:

```text
mÃ¶chte
Ãœbersicht
```

The scripts use `ftfy` when available to repair common encoding issues.

---

## Future Improvements

Potential extensions:

- Add hybrid retrieval using dense + sparse retrieval.
- Add Qdrant or Milvus as a production-style vector database.
- Add metadata filtering by language, queue, priority, and ticket type.
- Add automatic language detection.
- Add ticket queue classification before retrieval.
- Add confidence scoring for low-quality retrieval results.
- Add a Gradio or Streamlit chatbot interface.
- Add human evaluation of generated support responses.
- Compare base Llama vs fine-tuned Llama vs RAG-only vs RAG + fine-tuned Llama.
- Add BLEU, BERTScore, semantic similarity, and LLM-as-judge evaluation.
- Add experiment tracking with Weights & Biases or MLflow.
- Add Docker support.
- Add configuration files instead of editing Python dataclasses.
- Add multi-vector retrieval or late-interaction retrieval.
- Add safety rules for sensitive customer data.
- Add tests for data cleaning, prompt construction, and retrieval.

---

## Example Inference Flow

A user asks:

```text
Our AWS deployment failed after a recent configuration update. The service is down and we need urgent help.
```

The system:

1. embeds the query with BGE-M3,
2. retrieves similar solved tickets from FAISS,
3. reranks candidates with BGE reranker,
4. builds a RAG prompt with the top tickets,
5. sends the prompt to the fine-tuned Llama adapter,
6. generates a grounded support response,
7. and optionally reports retrieved evidence IDs.

Example output style:

```text
Thank you for reporting the AWS deployment issue. We understand the urgency and the impact on your operations. To help diagnose the failure, please provide the deployment logs, recent configuration changes, affected services, and any CloudTrail or CloudFormation error messages. We recommend verifying IAM permissions, checking service limits, and reviewing recent infrastructure updates. Once we receive the logs, we can escalate this to the cloud support team for further investigation.
```

---

## Reproducibility Checklist

Before running:

```text
[ ] Dataset downloaded and placed in the expected dataset directory
[ ] Python environment created
[ ] Dependencies installed
[ ] Hugging Face account approved for Llama 3.1
[ ] Hugging Face token created
[ ] 00_token_login.py runs successfully
```

After RAG building:

```text
[ ] train.csv exists
[ ] validation.csv exists
[ ] faiss_train.index exists
[ ] corpus.jsonl exists
[ ] train_sft_prompt_completion.jsonl exists
[ ] validation_sft_prompt_completion.jsonl exists
```

After fine-tuning:

```text
[ ] adapter_config.json exists
[ ] adapter_model.safetensors exists
[ ] tokenizer files exist
```

After validation:

```text
[ ] predictions.jsonl exists
[ ] predictions.csv exists
[ ] metrics_summary.json exists
```

---

## Known Limitations

- CPU training with Llama 3.1 8B is extremely slow.
- The dataset is synthetic/generated-style customer-support data, so real-world performance may differ.
- The validation metrics are simple and should not be treated as full human-quality evaluation.
- Lexical metrics may penalize valid alternative answers.
- The current FAISS setup uses dense retrieval only.
- The RAG-building stage does not currently apply metadata filters before retrieval.
- The training-time RAG context uses FAISS top-k retrieval without reranking for speed and stability.
- The validation stage includes reranking.
- The generated answer quality depends heavily on retrieved evidence quality.
- Model access requires Hugging Face approval for Llama 3.1.

---

## Ethical and Practical Considerations

Customer-support automation can improve response speed, but it should be deployed carefully.

Important considerations:

- avoid exposing private customer information,
- avoid hallucinating policies, prices, refunds, legal claims, or security guarantees,
- clearly disclose AI assistance when used in a real support workflow,
- keep a human escalation path,
- avoid using generated answers for critical incidents without review,
- and ensure compliance with the dataset and model licenses.

---

## Author’s Note

Thanks for checking out this project. My goal was to create a clean, practical, and reproducible RAG-first pipeline that shows how retrieval and fine-tuning can work together for a real-world customer-support application.

This project demonstrates:

- multilingual ticket processing,
- dense retrieval with BGE-M3,
- local vector search with FAISS,
- RAG-grounded prompt construction,
- Llama 3.1 LoRA/QLoRA fine-tuning,
- CPU fallback for limited hardware,
- and validation through retrieve → rerank → generate.

The most important design choice is that **RAG remains central throughout the project**: during training data construction, during validation, and during final chatbot-style inference.
