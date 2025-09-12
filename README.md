# Fine Tuning ESM2

1. rebuild dlfb package 
2. fine-tune for classifying proteins
3. rebuild in Rust


## Running
```bash
# install packages in pyproject.toml
uv sync

# install datasets
cd src/ && mkdir data
uv run dlfb-provision --chapter proteins --destination ./data

# execute the finetuning training loop and run an analysis
cd ../
uv run python main.py
```