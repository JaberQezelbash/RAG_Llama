# Technical Details

This document contains the deeper implementation notes for the **AI RAG Chatbot with Fine-Tuned Data** project.

The main README focuses on the project story and usage. This file explains the technical design, configuration, artifacts, and evaluation details.

---

## 1. Full Pipeline

The project is separated into four scripts:

```text
00_token_login.py
01_build_rag_training_data.py
02_train_llama_lora_or_qlora.py
03_validate_rag_lora_or_qlora.py
```

The pipeline is:

```text
Dataset CSV files
    ↓
Cleaning and deduplication
    ↓
Answered-ticket filtering
    ↓
Train / validation split
    ↓
Train-only BGE-M3 embeddings
    ↓
FAISS index
    ↓
RAG-grounded SFT prompt/completion records
    ↓
Llama LoRA/QLoRA adapter training
    ↓
Validation with retrieve → rerank → generate
```

---

## 2. Dataset Details

The dataset folder contains five CSV files:

```text
aa_dataset-tickets-multi-lang-5-2-50-version.csv
dataset-tickets-german_normalized.csv
dataset-tickets-german_normalized_50_5_2.csv
dataset-tickets-multi-lang3-4k.csv
dataset-tickets-multi-lang-4-20k.csv
```

The scripts automatically inspect all CSV files and normalize their columns.

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

Only rows with usable `subject`, `body`, and `answer` are used for supervised fine-tuning.

Rows without an `answer` column are not used for generator SFT, but they could be useful in future work for classification, routing, or unsupervised retrieval experiments.

---

## 3. Data Cleaning

The RAG-building script performs:

```text
1. CSV loading with multiple encodings
2. Column-name normalization
3. Empty-row filtering
4. Text cleanup
5. Mojibake repair
6. Deduplication
7. Ticket ID creation
8. Query/context text construction
```

The script tries these encodings:

```text
utf-8-sig
utf-8
cp1252
latin1
```

The project also uses `ftfy` when available to repair text artifacts such as:

```text
mÃ¶chte
Ãœbersicht
```

---

## 4. Train / Validation Split

The cleaned answered tickets are split into train and validation sets.

Default split:

```python
validation_size = 0.10
seed = 42
```

When possible, the split is stratified by language.

The resulting files are:

```text
outputs/processed/train.csv
outputs/processed/validation.csv
```

---

## 5. RAG Knowledge Base

The FAISS knowledge base is built **only from the training split**.

This is intentional:

```text
train.csv → BGE-M3 embeddings → FAISS index
```

Validation tickets are not inserted into the FAISS index. This helps reduce direct validation leakage.

Saved files:

```text
outputs/rag_store/faiss_train.index
outputs/rag_store/corpus.jsonl
outputs/rag_store/train_vectors.npy
```

---

## 6. Retriever

The retriever is:

```text
BAAI/bge-m3
```

The query text for each ticket is built from:

```text
subject
body
type
queue
priority
language
business_type
tags
```

The context text stored in the corpus includes:

```text
ticket_id
subject
body
answer
type
queue
priority
language
business_type
tags
```

Default embedding configuration:

```python
embedding_batch_size = 16
embedding_max_length = 8192
```

The FAISS index uses inner product similarity on normalized vectors.

---

## 7. RAG-Grounded SFT Dataset

The RAG-building script creates prompt/completion JSONL files:

```text
outputs/sft_data/train_sft_prompt_completion.jsonl
outputs/sft_data/validation_sft_prompt_completion.jsonl
```

Each SFT example has this structure:

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
      "content": "Reference support answer..."
    }
  ]
}
```

The user prompt contains:

```text
Customer ticket to answer:
Subject: ...
Message: ...

Ticket metadata:
type=...; queue=...; priority=...; language=...; business_type=...; tags=...

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

Default retrieval settings for SFT creation:

```python
faiss_candidate_k = 16
sft_context_top_k = 3
max_context_chars_per_ticket = 950
max_query_chars = 3500
```

The SFT construction currently uses FAISS top-k retrieval without reranking for speed and stability.

---

## 8. Generator Model

The generator is:

```text
meta-llama/Llama-3.1-8B-Instruct
```

This is a gated Hugging Face model. Users must:

1. request access on Hugging Face,
2. create a Hugging Face token,
3. run `00_token_login.py`.

The token is requested with:

```python
getpass.getpass(...)
```

The token should not be hard-coded or committed.

---

## 9. Fine-Tuning Modes

The fine-tuning script supports two modes.

---

### 9.1 CUDA QLoRA Mode

Used when:

```python
torch.cuda.is_available() == True
```

The script uses:

```text
4-bit quantized model loading
NF4 quantization
LoRA adapters
paged_adamw_8bit optimizer
gradient checkpointing
```

Default CUDA settings:

```python
max_seq_length = 3072
num_train_epochs = 1.0
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
warmup_ratio = 0.03
save_steps = 500
eval_steps = 250
```

---

### 9.2 CPU LoRA Fallback Mode

Used when:

```python
torch.cuda.is_available() == False
```

This keeps the same Llama model but does not use CUDA 4-bit quantization.

Default CPU settings:

```python
cpu_train_record_limit = 800
cpu_eval_record_limit = 100
cpu_max_seq_length = 1024
cpu_torch_dtype = "float16"
```

CPU training with Llama 3.1 8B is very slow. These defaults are conservative so that the script can still run on limited hardware.

To train on all SFT examples on CPU, set:

```python
cpu_train_record_limit = None
```

However, this can take several days or longer depending on hardware.

---

## 10. LoRA Configuration

Default LoRA settings:

```python
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
```

Target modules:

```python
(
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
```

These target common attention and MLP projection layers in Llama-style architectures.

The adapter is saved to:

```text
outputs/llama31_8b_rag_lora_or_qlora_adapter/
```

---

## 11. Validation Pipeline

The validation script evaluates the complete RAG pipeline:

```text
validation ticket
    ↓
BGE-M3 embedding
    ↓
FAISS retrieval
    ↓
BGE reranking
    ↓
RAG prompt construction
    ↓
fine-tuned Llama generation
    ↓
metrics and saved predictions
```

The validation script loads:

```text
outputs/processed/validation.csv
outputs/rag_store/faiss_train.index
outputs/rag_store/corpus.jsonl
outputs/llama31_8b_rag_lora_or_qlora_adapter/
```

Default validation settings:

```python
max_validation_examples = 100
faiss_candidate_k = 30
rerank_top_k = 5
max_prompt_tokens = 4096
max_new_tokens = 384
```

Set:

```python
max_validation_examples = None
```

to evaluate the full validation set.

---

## 12. Reranker Details

The reranker is:

```text
BAAI/bge-reranker-v2-m3
```

The validation script uses a manual wrapper based on:

```python
AutoTokenizer
AutoModelForSequenceClassification
```

This avoids a compatibility issue that can occur with some `FlagEmbedding` reranker calls.

The wrapper scores query/document pairs and sorts retrieved candidates by reranker score.

Default reranker settings:

```python
reranker_batch_size = 8
reranker_max_length = 512
reranker_max_passage_chars = 3500
```

---

## 13. Output Files

After the RAG-building stage:

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
└── sft_data/
    ├── train_sft_prompt_completion.jsonl
    ├── validation_sft_prompt_completion.jsonl
    ├── train_sft_audit.jsonl
    └── validation_sft_audit.jsonl
```

After fine-tuning:

```text
outputs/
├── trainer_runs_lora_or_qlora/
│   └── checkpoint-*/
│
└── llama31_8b_rag_lora_or_qlora_adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    └── tokenizer files
```

After validation:

```text
outputs/
└── validation_results_lora_or_qlora/
    ├── predictions.jsonl
    ├── predictions.csv
    ├── metrics_summary.json
    └── config_validate_lora_or_qlora.json
```

---

## 14. Validation Metrics

The validation script saves both generation and retrieval metrics.

### Generation Metrics

```text
lexical_f1
rouge_l_f1
```

These compare the generated answer with the reference support answer.

### Retrieval Metadata Metrics

```text
top1_same_queue
top1_same_language
topk_same_queue
topk_same_language
topk_tag_overlap
best_faiss_score
best_rerank_score
```

These measure whether the retrieved tickets are similar in language, queue, tags, and retrieval score.

Metrics are saved to:

```text
outputs/validation_results_lora_or_qlora/metrics_summary.json
```

---

## 15. Example Metrics File

Example structure:

```json
{
  "n": 100,
  "mean_lexical_f1": 0.32,
  "mean_rouge_l_f1": 0.28,
  "mean_top1_same_queue": 0.61,
  "mean_top1_same_language": 0.89,
  "mean_topk_same_queue": 0.78,
  "mean_topk_same_language": 0.95,
  "mean_topk_tag_overlap": 0.70,
  "mean_best_faiss_score": 0.64,
  "mean_best_rerank_score": 0.81
}
```

The exact values depend on data split, hardware, training duration, and generation settings.

---

## 16. Important Configuration Summary

### RAG Building

```python
faiss_candidate_k = 16
sft_context_top_k = 3
embedding_batch_size = 16
embedding_max_length = 8192
max_context_chars_per_ticket = 950
max_query_chars = 3500
```

### Llama Fine-Tuning

```python
generator_model_id = "meta-llama/Llama-3.1-8B-Instruct"

max_seq_length = 3072
num_train_epochs = 1.0
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
```

### CPU Fallback

```python
cpu_train_record_limit = 800
cpu_eval_record_limit = 100
cpu_max_seq_length = 1024
cpu_torch_dtype = "float16"
```

### Validation

```python
max_validation_examples = 100
faiss_candidate_k = 30
rerank_top_k = 5
max_prompt_tokens = 4096
max_new_tokens = 384
```

---

## 17. Troubleshooting

### Llama access error

Possible error:

```text
GatedRepoError
401 Unauthorized
Cannot access gated repo
```

Fix:

1. Make sure Hugging Face granted access to Llama 3.1.
2. Create a read token.
3. Run:

```bash
python codes/00_token_login.py
```

---

### No CUDA detected

Message:

```text
CUDA available: False
Training mode: cpu_lora
```

This is expected on CPU-only machines.

The project will still run, but it can be very slow.

---

### Reranker tokenizer error

Possible error:

```text
AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model
```

The validation script avoids this by using a custom `SafeBGEReranker` wrapper.

---

### Memory issues on CPU

If CPU model loading fails, try:

```python
cpu_torch_dtype = "float32"
```

or reduce:

```python
cpu_train_record_limit
cpu_max_seq_length
max_validation_examples
max_prompt_tokens
max_new_tokens
```

---

### Slow validation

CPU generation with Llama is slow. Reduce:

```python
max_validation_examples
max_new_tokens
rerank_top_k
```

---

## 18. Recommended `.gitignore`

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

Do not commit:

```text
Hugging Face tokens
dataset files
model weights
adapter checkpoints
large output artifacts
```

---

## 19. Future Improvements

Possible extensions:

```text
- Add sparse + dense hybrid retrieval
- Add metadata filtering before retrieval
- Add Qdrant, Chroma, or Milvus
- Add language detection
- Add ticket queue classification
- Add Gradio or Streamlit chatbot UI
- Add human evaluation
- Compare base Llama vs fine-tuned Llama vs RAG-only vs RAG + fine-tuned Llama
- Add BERTScore or semantic similarity metrics
- Add LLM-as-judge evaluation
- Add MLflow or Weights & Biases tracking
- Add Docker support
- Convert dataclass settings to YAML configs
- Add unit tests for data cleaning, retrieval, and prompt construction
```

---

## 20. Reproducibility Checklist

Before running:

```text
[ ] Dataset downloaded
[ ] Dataset placed in dataset/Customer IT Support -Ticket Dataset/
[ ] Python dependencies installed
[ ] Hugging Face access granted for Llama 3.1
[ ] Hugging Face token created
[ ] 00_token_login.py runs successfully
```

After RAG building:

```text
[ ] outputs/processed/train.csv exists
[ ] outputs/processed/validation.csv exists
[ ] outputs/rag_store/faiss_train.index exists
[ ] outputs/rag_store/corpus.jsonl exists
[ ] outputs/sft_data/train_sft_prompt_completion.jsonl exists
[ ] outputs/sft_data/validation_sft_prompt_completion.jsonl exists
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

## 21. Core Design Principle

The most important design principle of this project is:

> RAG is central at every stage.

RAG is used:

```text
during training-data construction,
during validation,
and during final chatbot-style generation.
```

The model is not simply fine-tuned to answer tickets from memory. It is trained and evaluated in a retrieval-grounded setting.