## RAG-based finetuning of Llama-3.1-8B-Instruct



<!-- Optional: add a banner/diagram here -->
<!-- 
<img width="900" alt="banner" src="https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/blob/main/assets/banner.png">
<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.svg">
-->

<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.png">

This repository contains one of my end-to-end fine-tuning projects: a RAG-based multilingual IT support chatbot using Meta Llama-3.1-8B-Instruct as the generator, BGE-M3 for retrieval, BGE reranking, and FAISS vector search. Fine-tunes the generator with QLoRA SFT on the support-ticket data and evaluated performance using retrieval quality, reranks evidence relevance, lexical F1, ROUGE-L, and generates response quality metrics.
