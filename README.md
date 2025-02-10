## Minimal example of RAG for RCG24

This repository provides a minimal implementation of a Retrieval-Augmented Generation (RAG) process to answer questions based on the content of the OECD Regions and Cities at a Glance 2024 (RCG24) report.

### Prerequisites

- A [Hugging Face API Token](https://huggingface.co/settings/tokens)

### Usage

To install all dependencies run the following command:

```
uv pip install -r pyproject.toml
```

To run the RAG application, run the following command:

```
uv run main.py
```

### Project Structure

- `main.py`: Main script to run the RAG application.
- `input/rcg24.pdf`: The RCG24 document to be processed.
- `.env`: Environment variables file. Contains the Hugging Face API Token as 'HUGGINGFACEHUB_API_TOKEN='.
- `pyproject.toml`: Project dependencies and configuration.
- `README.md`: Project documentation.