# English-Arabic Bidirectional Translation

This project fine-tunes a pretrained neural machine translation model to translate between English and Arabic in both directions.

## Features
- Supports both English → Arabic and Arabic → English translation
- Fine-tunes a MarianMT model with language tokens
- Includes a simple GUI for user interaction
- Evaluates translation quality with test samples

## Dataset
- Tatoeba (English-Arabic parallel corpus)
- Preprocessing with language tokens (">>ar<<" for Arabic output, ">>en<<" for English output)

## Model
- Based on Helsinki-NLP MarianMT pretrained models
- Fine-tuned for bidirectional translation

## Project structure
```bash
english-arabic-bidirectional-translation/
│
├── data/
│ ├── Tatoeba.ar-en.ar
│ ├── Tatoeba.ar-en.en
│ └── translation_dataset.tsv
├── scripts/
│ └── prepare_dataset.py
├── models/
├── gui/
└── README.md
```
## How to run
1. Install dependencies
2. Run dataset preparation
3. Fine-tune the model
4. Launch the GUI

---
