#!/bin/bash

# Use this init file only within Onyxia

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the dependencies
uv pip install -r llm_rcg_rag/pyproject.toml
