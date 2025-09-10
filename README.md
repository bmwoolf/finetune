# finetune

Fine-tune ESM2 to classify proteins.


ESM2 outputs representations- which are embeddings you can use for downstream tasks (classification, regression, PTM site prediction).

We can use the `transformers` library because ESM2 looks like NLP with 1D sparse vectors as outputs, we can't use it on AlphaFold/Boltz because their outputs look like geometric physics.


## running
```bash
uv sync
uv run python main.py
```