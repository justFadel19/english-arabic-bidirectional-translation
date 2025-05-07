import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Model parameters
MODEL_NAME = "facebook/bart-base"
MAX_LENGTH = 64
BATCH_SIZE = 12  # Reduced batch size for better generalization
LEARNING_RATE = 5e-5  # Further reduced learning rate for fine-tuning
NUM_EPOCHS = 10  # Increased number of epochs for better learning
WARMUP_RATIO = 0.1  # Warm up learning rate
ATTENTION_DROPOUT = 0.15  # Increased dropout for better generalization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def translate_example(model, tokenizer, text, src_lang, tgt_lang, device):
    """Translate a single example text."""
    # Create the input prompt based on the source and target languages
    if src_lang == "en" and tgt_lang == "ar":
        input_text = f"en2ar: {text}"
    else:
        input_text = f"ar2en: {text}"
    
    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,  # Avoid repeating the same n-grams
            do_sample=True,  # Enable sampling
            top_k=50,  # More diverse outputs
            top_p=0.95  # Nucleus sampling for better quality
        )
    
    # Decode and return the translation
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def show_translation_examples(model, tokenizer, device):
    """Show some example translations to demonstrate model capabilities."""
    print("\n===== Translation Examples =====")
    
    # English to Arabic examples
    english_examples = [
        "Hello, how are you?",
        "I love learning languages.",
        "What time is it?",
        "My name is John and I'm a student.",
        "The weather is nice today.",
        "I want to travel to Egypt next year."
    ]
    
    print("\nEnglish to Arabic Examples:")
    for text in english_examples:
        translation = translate_example(model, tokenizer, text, "en", "ar", device)
        print(f"English: {text}")
        print(f"Arabic:  {translation}")
        print("-" * 40)
    
    # Arabic to English examples
    arabic_examples = [
        "مرحبا، كيف حالك؟",
        "أنا أحب تعلم اللغات.",
        "كم الساعة الآن؟",
        "اسمي محمد وأنا طالب.",
        "الطقس جميل اليوم.",
        "أريد السفر إلى مصر العام المقبل."
    ]
    
    print("\nArabic to English Examples:")
    for text in arabic_examples:
        translation = translate_example(model, tokenizer, text, "ar", "en", device)
        print(f"Arabic:  {text}")
        print(f"English: {translation}")
        print("-" * 40)

def main():
    print("Starting BART fine-tuning for English-Arabic translation...")
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    
    # Load dataset from TSV
    print("Loading dataset...")
    dataset = load_dataset('csv', data_files='data/translation_dataset.tsv', delimiter='\t')
    
    # Split into train/validation
    split_data = dataset['train'].train_test_split(test_size=0.1)
    dataset = DatasetDict({
        'train': split_data['train'],
        'validation': split_data['test']
    })
    
    # Train on the full dataset
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # Load tokenizer and model
    print(f"Loading {MODEL_NAME} tokenizer and model...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        attention_dropout=ATTENTION_DROPOUT  # Apply dropout for better generalization
    )
    model.to(DEVICE)
    
    # Simple tokenization function
    def preprocess_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=MAX_LENGTH, 
                truncation=True,
                padding="max_length"
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    # Apply tokenization
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=["input_text", "target_text", "direction"]
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="results",
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_ratio=WARMUP_RATIO,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,  # Increased for larger effective batch size
        logging_steps=100,
        logging_dir="results/logs"
    )
    
    # Initialize the Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    model.save_pretrained("models/bart-english-arabic-final")
    tokenizer.save_pretrained("models/bart-english-arabic-final")
    
    # Show translation examples
    print("\nGenerating translation examples...")
    show_translation_examples(model, tokenizer, DEVICE)
            
    # Print completion message
    print("\nTraining complete! Model saved to models/bart-english-arabic-final")

if __name__ == "__main__":
    main()
