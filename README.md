## Minimal example of RAG for RCG24

This repository contains an example of a minimal RAG for the RCG24 report.

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