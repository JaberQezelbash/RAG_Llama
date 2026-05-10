


## AI RAG Chatbot with Fine-Tuned Data (Llama 3.1 + BGE-M3 + FAISS)

**RAG-based customer-support chatbot using [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), FAISS retrieval, and LoRA/QLoRA fine-tuning on multilingual IT support tickets.**



<!-- Optional: add a banner/diagram here -->
<!-- 
<img width="900" alt="banner" src="https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/blob/main/assets/banner.png">
<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.svg">
-->

<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.png">


This repository contains an end-to-end **Retrieval-Augmented Generation (RAG)** project for customer-support automation. The goal is to build a multilingual IT-support chatbot that retrieves relevant solved tickets, uses them as grounding evidence, and then generates a professional support response using a fine-tuned Llama model.

Unlike a simple fine-tuning project, this project intentionally makes **RAG the core component**. The model is not trained only to memorize support answers. Instead, each supervised fine-tuning example is built around retrieved context from a FAISS-based knowledge base, and the validation pipeline also follows the same retrieve → rerank → generate workflow.

> ⚠️ Disclaimer: This project is only for research, education, and portfolio demonstration.




## Motivation

Customer-support chatbots need to be accurate, consistent, and context-aware. A model that only relies on fine-tuning may learn response style, but it can still hallucinate or miss ticket-specific context.

This project combines:

- **retrieval** to find similar solved tickets,
- **reranking** to improve evidence quality,
- **fine-tuning** to adapt the model to support-style responses,
- and **validation** to test the full retrieve → rerank → generate pipeline.

The result is a practical RAG-based chatbot workflow for multilingual IT support tickets.



## Model Stack

| Component | Model / Tool |
|---|---|
| Generator | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Retriever | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) |
| Reranker | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Vector Store | FAISS |
| Fine-tuning | LoRA / QLoRA |
| Dataset | Customer IT Support / Multilingual Customer Support Tickets (from [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?resource=download)) |




### What This Model Does

The project is split into four clear stages:

```text
1. Hugging Face token login
2. RAG training-data construction
3. Llama LoRA/QLoRA fine-tuning
4. RAG validation and response generation
```

The full workflow:

```text
Support ticket dataset
        ↓
Clean and prepare answered tickets
        ↓
Build FAISS knowledge base from train tickets
        ↓
Retrieve similar solved tickets
        ↓
Create RAG-grounded SFT examples
        ↓
Fine-tune Llama adapter
        ↓
Validate using retrieve → rerank → generate
```



## Repository Structure

```text
.
├── codes/
│   ├── 00_token_login.py
│   ├── 01_build_rag_training_data.py
│   ├── 02_train_llama_lora_or_qlora.py
│   └── 03_validate_rag_lora_or_qlora.py
│
├── assets/
│   ├── technical_details.md
│   ├── requirements.txt
│   └── model_architecture.png
│
├── outputs/
│   ├── processed/
│   ├── rag_store/
│   ├── sft_data/
│   ├── llama31_8b_rag_lora_or_qlora_adapter/
│   └── validation_results_lora_or_qlora/
│
└── README.md
```

For implementation details, configurations, metrics, and troubleshooting notes, see:

[`assets/technical_details.md`](assets/technical_details.md)



## Dataset

This project uses the *Customer IT Support / Multilingual Customer Support Tickets* dataset available on [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?resource=download).

The dataset contains multilingual support tickets with fields such as:

```text
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

More dataset-processing details are available in [`assets/technical_details.md`](assets/technical_details.md).



## Key Features

✅ RAG-based chatbot pipeline  
✅ Multilingual support-ticket processing  
✅ FAISS-based dense retrieval  
✅ BGE-M3 embeddings  
✅ BGE reranking during validation  
✅ Llama 3.1 8B Instruct generator  
✅ LoRA/QLoRA fine-tuning  
✅ CPU fallback when CUDA is unavailable  
✅ Train-only retrieval index to reduce validation leakage  
✅ Grounded response generation from retrieved solved tickets  
✅ Saves predictions, metrics, retrieved evidence, and adapter files  



## Quick Start

### 0. Log in to Hugging Face

Llama 3.1 is a gated model, so you need Hugging Face access and a token. Run:

```bash
python codes/00_token_login.py
```

### 1. Building RAG

```bash
python codes/01_build_rag_training_data.py
```

This creates:

```text
outputs/processed/
outputs/rag_store/
outputs/sft_data/
```

These include the cleaned train/validation splits, FAISS index, corpus, and RAG-grounded supervised fine-tuning records.



### 2. Fine-tune Llama

```bash
python codes/02_train_llama_lora_or_qlora.py
```

### 3. Validation

```bash
python codes/03_validate_rag_lora_or_qlora.py
```

This runs:

```text
retrieve → rerank → generate
```

and saves predictions and metrics to:

```text
outputs/validation_results_lora_or_qlora/
```


## Output Highlights

After the full pipeline runs, the main outputs are:

```text
outputs/rag_store/faiss_train.index
outputs/rag_store/corpus.jsonl
outputs/sft_data/train_sft_prompt_completion.jsonl
outputs/sft_data/validation_sft_prompt_completion.jsonl
outputs/llama31_8b_rag_lora_or_qlora_adapter/
outputs/validation_results_lora_or_qlora/predictions.csv
outputs/validation_results_lora_or_qlora/metrics_summary.json
```

The validation output includes:

- customer ticket,
- retrieved ticket IDs,
- generated answer,
- reference answer,
- retrieval scores,
- reranker scores,
- and evaluation metrics.



## Example Use Case

A customer writes:

```text
Our AWS deployment failed after a recent configuration update.
The service is down and we need urgent help.
```

The system retrieves similar solved tickets about AWS deployment failures, reranks them, and generates a grounded response such as:

```text
Thank you for reporting the AWS deployment issue. We understand the urgency and the impact on your operations. Please provide the deployment logs, recent configuration changes, affected services, and any CloudTrail or CloudFormation error messages. We recommend checking IAM permissions, service limits, and recent infrastructure updates while our team investigates further.
```



## Notes on CPU Training

This project can run without a GPU, but Llama 3.1 8B is large.
On CPU, training is slow. The default CPU configuration intentionally limits the number of training examples to keep the experiment practical.
For faster and more complete fine-tuning, CUDA GPU training is strongly recommended.
See [`assets/technical_details.md`](assets/technical_details.md) for CPU/GPU configuration details.



## Technical Details

The README keeps the workflow high-level. For deeper implementation notes, see:
[`assets/technical_details.md`](assets/technical_details.md)

That file includes:
- dataset preprocessing details,
- RAG prompt format,
- FAISS index construction,
- LoRA/QLoRA configuration,
- CPU/GPU behavior,
- validation metrics,
- output files,
- troubleshooting notes,
- and future improvement ideas.



## Author’s Note

The goal was to build a clean and reproducible RAG-based pipeline that demonstrates how retrieval, reranking, and fine-tuning can work together in a realistic customer-support chatbot setting.
The most important design choice is that *RAG remains central* throughout the project: during training-data construction, dusing LLM finetuning, during validation, and during final chatbot-style response generation.
