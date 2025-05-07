# Model Files

This directory contains the fine-tuned BART model for English-Arabic translation.

Due to the large size of the model files, they are not included in the Git repository.

## How to obtain the model files

### Option 1: Fine-tune the model yourself
Run the fine-tuning script to create your own model:
```bash
python scripts/finetune_model.py
```

### Option 2: Download pre-trained model
The fine-tuned model can be downloaded from [Hugging Face Hub](https://huggingface.co/) or other file sharing services.

Once downloaded, extract the model files to:
```
models/bart-english-arabic-final/
```

The model directory should contain the following files:
- config.json
- generation_config.json
- merges.txt
- model.safetensors
- special_tokens_map.json
- tokenizer_config.json
- vocab.json
