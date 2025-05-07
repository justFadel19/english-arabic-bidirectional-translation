# Quick Start Guide

This guide will help you get started with the English-Arabic Bidirectional Translation project.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB of RAM (16GB recommended)
- Internet connection to download the pretrained model

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/english-arabic-bidirectional-translation.git
   cd english-arabic-bidirectional-translation
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The project uses the Tatoeba English-Arabic parallel corpus. The data files are already included in the `data/` directory.

To prepare the dataset for model fine-tuning:

```bash
python scripts/prepare_dataset.py
```

This will create processed dataset files in the `data/` directory.

## Model Fine-tuning

To fine-tune the Facebook BART base model on the prepared dataset:

```bash
python scripts/finetune_model.py
```

The fine-tuning process has been optimized for faster training. By default, it:

- Uses a suitable training dataset
- Runs for 10 epochs with a learning rate of 5e-5
- Uses a batch size of 12
- Enables mixed precision training (FP16) when a GPU is available
- Adds warmup_ratio and attention_dropout for improved performance
- Uses gradient accumulation for larger effective batch size
- Shows example translations after training is complete

To modify these settings, you can edit the parameters at the top of the `finetune_model.py` file.
```

The fine-tuned model will be saved in the `models/bart-english-arabic-final/` directory.

## Evaluation

To evaluate the model's performance on the test set:

```bash
python scripts/evaluate_model.py
```

This will:
- Calculate BLEU scores for both translation directions
- Generate example translations for a sample of test cases
- Save the evaluation results in the `results/` directory

The evaluation script is optimized to use a smaller test set (10 samples by default) for faster results.

## Using the Translator

### GUI Application

To launch the graphical user interface:

```bash
python gui/translation_app.py
```

The GUI provides an easy way to translate text between English and Arabic with the following optimized features:

- Fast model loading with TorchScript optimization when available
- Half-precision inference on GPU for faster translation
- Enhanced nucleus sampling for better translation quality

## Project Structure

- `data/`: Contains the parallel corpus and processed datasets
- `scripts/`: Python scripts for data preparation, model training, and evaluation
- `models/`: Directory where trained models are saved
- `results/`: Evaluation results and visualizations
- `gui/`: GUI application for translation
- `docs/`: Documentation files

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size or MAX_TRAIN_SAMPLES in `scripts/finetune_model.py`
- **CUDA Not Found**: Ensure you have CUDA installed and compatible with your PyTorch version
- **Slow Training**: Enable mixed precision by ensuring `USE_FP16` is set to True
- **TorchScript Errors**: If you encounter errors with optimized model loading, the code will fall back to standard loading
- **Long Translation Times**: Consider reducing MAX_TARGET_LENGTH or adjusting the nucleus sampling parameters
- **Model Not Found**: Make sure you've completed the training step before evaluation or using the GUI
