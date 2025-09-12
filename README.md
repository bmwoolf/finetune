# Fine Tuning ESM2

1. rebuild dlfb package 
2. fine-tune for classifying proteins
3. rebuild in Rust


## Running
```bash
# install packages in pyproject.toml
uv sync

# execute the finetuning training loop and run an analysis
uv run python main.py
```