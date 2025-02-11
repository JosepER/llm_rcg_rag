#!/bin/bash

# Use this init file only within Onyxia

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Change directory
cd llm_rcg_rag/

# Install the dependencies
uv pip install -r pyproject.toml
