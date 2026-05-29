.PHONY: run

run: 
	uv run ./src/main.py

install-torch:
	uv pip uninstall torch
	uv pip install torch --index-url https://download.pytorch.org/whl/cu130