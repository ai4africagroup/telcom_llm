# Phi-2 and Falcon-7B Models

## Overview

This repository contains the implementation and training code for our Phi-2 model and inference code for the Falcon-7B model. The Phi-2 model is fine-tuned using QLoRA and a custom fine-tuning script, while Falcon-7B is employed for efficient inference. The repository is organized into training, inference, and utility scripts for streamlined usage and development.

## Table of Contents

1. [Phi-2 Model](#phi-2-model)
    - [Training](#training)
    - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    - [Inference](#inference)
2. [Falcon-7B Model](#falcon-7b-model)
    - [Inference](#inference-1)
3. [Utility Functions](#utility-functions)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Phi-2 Model

### Training

The Phi-2 model training is managed through multiple scripts to facilitate different aspects of the fine-tuning process.

- **train.py**: The primary training file for the Phi-2 model using QLoRA.
- **misc/custom_finetuning.py**: We have written our own custom fine-tuning script for developed from scratch (LORA).
- **train_rough.py**: A rough version of the trainer code for preliminary tests and experiments.

### Retrieval-Augmented Generation (RAG)

The RAG implementation involves several custom-built scripts designed to handle document chunking, retrieval, and generation.

- **RAG.py**: Our RAG implementation, written completely from scratch.
- **data_preprocessing.py**: A document chunker that preprocesses text data by calling scripts from the `scripts` directory.
- **bm_25_extractor.py**: A BM25 extractor for efficient document retrieval.

### Inference

- **Phi_inference.py**: Custom inference code for the Phi-2 model.

## Falcon-7B Model

### Inference

- **inference_haystac_falcon.ipynb**: Notebook for performing inference using the Falcon-7B model.

## Utility Functions

- **utils.py**: Contains various utility functions used across the project.

## Installation

To get started with this repository, clone it to your local machine and install the required dependencies.

```bash
git clone https://github.com/ai4africagroup/telcom_llm.git
cd phi-2-falcon-7b
pip install -r requirements.txt
