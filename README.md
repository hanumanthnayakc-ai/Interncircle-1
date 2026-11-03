# Tumour classifier: interactive Streamlit app

Sliders for six cell measurements; a decision tree re-predicts on every change.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py        # only needed if model.joblib is missing or scikit-learn differs
streamlit run app.py
```

The model file is tied to the scikit-learn version it was trained with (1.9.1). If you use a
different version, re-run `python train_model.py` to rebuild it.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Picks the 6 most important features, tunes the tree, saves `model.joblib` and `model_meta.json` |
| `predictor.py` | Loads the model; slider ranges, predictions, decision-path explanation (no UI code) |
| `app.py` | The Streamlit page |
| `test_predictor.py` | Checks for the prediction logic (`python test_predictor.py`) |

## Deploy

- **Streamlit Community Cloud**: push this folder to a GitHub repo, then create an app pointing
  at `app.py`. Commit `model.joblib`, `model_meta.json` and `requirements.txt`.
- **Hugging Face Spaces**: create a Space with the Streamlit SDK and upload the same files.

Educational demo only. Not a medical device.
