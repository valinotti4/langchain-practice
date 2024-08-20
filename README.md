# Langchain practice

Repo developed in a [Langchain crash course](https://www.youtube.com/watch?v=yF9kGESAi3M&t=511s)

## Installation

### Create venv

`python3 -m venv langchain-practice`

### Activate venv

`source langchain-practice/bin/activate`

### Install libraries

`pip install <library_name>`

### Save dependencies

`pip freeze > requirements.txt`

### Install dependencies

`pip install -r requirements.txt`

### Deactivate venv

`deactivate`

## How to use

Run any file independently by: `python3 <filepath>`

- `./2-rag/basic_metadata_part1.py` is an example that creates a vector database from .txt books and persists it locally.
- `./2-rag/basic_metadata_part2.py` uses the persisted database created in the part 1 and creates a retriever to get relevant documents.
- `./2-rag/conversational_rag.py` RAG with conversational capabilities using the database persisted in the `basic_metadata_part1`
