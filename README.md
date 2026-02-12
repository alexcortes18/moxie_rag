# Moxie RAG

## Setup

1. Create a virtual environment (Python 3.10 or 3.11 recommended for `chromadb`):

```bash
python3.10 -m venv .venv
# or
python3.11 -m venv .venv
```

If you want to target a specific Python 3.10 interpreter (system or another venv), use its full path. Example:

```bash
/Users/alejandrocortes/Documents/Projects/AI\ Projects/rag_fundamentals/.venv/bin/python -m venv .venv
```

2. Activate your virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Create a `.env` file with your OpenAI key:

```bash
OPENAI_API_KEY=your_key_here
```

## Notes

- `os` and `uuid` are part of the Python standard library, so they are not listed in `requirements.txt`.
