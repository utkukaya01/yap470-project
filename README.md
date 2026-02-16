# Fake News Detection (ISOT + LIAR)

This repository contains a two-stage fake news detection study built on the **ISOT** and **LIAR** datasets.

- **Method 1**: classical ML pipeline (BoW / TF-IDF + SVD / Chi2 + GBM / SVM / MLP)
- **Method 2**: hybrid deep model (Word2Vec + Bi-LSTM + best Method 1 auxiliary features)

The project is notebook-first and already includes processed data, extracted features, and many trained models.

## Project Structure

```text
.
|-- method1_training.ipynb
|-- method1_testing.ipynb
|-- method2_training.ipynb
|-- method2_testing.ipynb
|-- data/
|   |-- raw/
|   |   |-- ISOT/{Fake.csv, True.csv}
|   |   `-- LIAR/{train.tsv, valid.tsv, test.tsv}
|   |-- processed/{ISOT, LIAR}/*.pkl
|   `-- features/{ISOT, LIAR}/*.joblib
`-- saved_models/
    |-- method1/*.{joblib,keras}
    `-- method2/{w2v,bilstm,tokenizer}/*
```

## Environment

- Tested on **Python 3.12.3**
- Recommended: virtual environment (`venv` or `conda`)
- Optional but helpful: GPU for Method 2 training
- If cloning from Git, install **Git LFS** first (model/feature files are LFS-tracked)

## Installation

### 1) Core dependencies (Method 1)

```bash
pip install tensorflow scikit-learn pandas matplotlib nltk spacy ipywidgets joblib scipy tqdm
python -m spacy download en_core_web_sm
```

### 2) Additional dependency for Method 2

```bash
pip install gensim
```

Note: `gensim` may update `numpy` and can affect compatibility with other packages.

## How to Run

Run notebooks in this order:

1. `method1_training.ipynb`
2. `method1_testing.ipynb`
3. `method2_training.ipynb`
4. `method2_testing.ipynb`

Why this order:

- Method 1 generates processed datasets and feature objects under `data/processed` and `data/features`.
- Method 2 consumes those Method 1 outputs as auxiliary inputs.

## Data Notes

- **ISOT**: binary labels are created as `Fake -> 0`, `True -> 1`.
- **LIAR**: 6-class labels are mapped to binary:
  - fake side: `pants-fire`, `false`, `barely-true` -> `0`
  - true side: `half-true`, `mostly-true`, `true` -> `1`
- Both datasets are split with stratification and `random_state=42`.

## Current Artifacts in Repository

Repository already contains:

- Processed train/test `.pkl` files
- Feature extraction objects (`.joblib`)
- Method 1 trained models (`GBM`, `SVM`, `MLP`)
- Method 2 Word2Vec models, tokenizers, and Bi-LSTM checkpoints

So you can directly run testing notebooks without retraining everything.

## Reference Results (from notebook outputs)

- **Method 1 (best observed F1-macro)**
  - ISOT: `0.9968` (`bow min_df=30 + chi2=700 + MLP`)
  - LIAR: `0.6090` (`tfidf min_df=40 + chi2=700 + SVM`)
- **Method 2 (best observed macro F1 in testing notebook)**
  - ISOT: `0.9989` (`w2v_dim=200`, `window=8`)
  - LIAR: `0.6115` (`w2v_dim=200`, `window=5`)

## Common Issues

- `Missing model: ...` in testing notebooks:
  - Run related training notebook first, or verify files under `saved_models/`.
- LFS pointer files instead of actual model files:
  - Ensure Git LFS is installed and LFS objects are pulled.
- Package version conflicts after installing `gensim`:
  - Recreate a fresh virtual environment and reinstall dependencies in order.

## Dataset and Usage

- LIAR dataset readme is available at `data/raw/LIAR/README`.
- Follow original dataset licenses/terms for usage and distribution.
