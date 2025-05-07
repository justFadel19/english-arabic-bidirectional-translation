import os
import pandas as pd
import torch
import evaluate
from transformers import BartForConditionalGeneration, BartTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
MODEL_PATH = os.path.join(MODEL_DIR, "bart-english-arabic-final")
MAX_SOURCE_LENGTH = 64
MAX_TARGET_LENGTH = 64

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {MODEL_PATH}")
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device

def translate(text, source_lang, target_lang, model, tokenizer, device):
    """Translate text using the fine-tuned model."""
    # Create the input text with the appropriate prefix
    if source_lang == "en" and target_lang == "ar":
        input_text = f"en2ar: {text}"
    elif source_lang == "ar" and target_lang == "en":
        input_text = f"ar2en: {text}"
    else:
        raise ValueError("Unsupported language pair")
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_SOURCE_LENGTH, padding="max_length", truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Generate translation
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    # Decode the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation

def evaluate_model():
    """Evaluate the model on the test set and save results."""
    model, tokenizer, device = load_model_and_tokenizer()
    # Load dataset from single TSV file and select test subset
    dataset_file = os.path.join(DATA_DIR, 'translation_dataset.tsv')
    full_df = pd.read_csv(dataset_file, sep='\t')
    
    # Split dataset to get test set (10% for testing)
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(full_df, test_size=0.1, random_state=42)
    print(f"Loaded full dataset from {dataset_file} and selected {len(test_df)} test samples")
    
    # Load BLEU metric
    bleu = evaluate.load("sacrebleu")
    # Separate test data by direction
    en2ar_data = test_df[test_df["direction"] == "en2ar"].to_dict('records')
    ar2en_data = test_df[test_df["direction"] == "ar2en"].to_dict('records')
    # Evaluate with a smaller test set for faster evaluation
    MAX_TEST_SAMPLES = 10  # Further reduced for quick demonstration
    # Evaluate English to Arabic
    en2ar_predictions = []
    en2ar_references = []
    
    print(f"Evaluating English to Arabic translations (using {min(MAX_TEST_SAMPLES, len(en2ar_data))} samples)...")
    for item in en2ar_data[:MAX_TEST_SAMPLES]:
        source = item["input_text"].replace("en2ar: ", "")
        target = item["target_text"]
        prediction = translate(source, "en", "ar", model, tokenizer, device)
        en2ar_predictions.append(prediction)
        en2ar_references.append([target])
    
    en2ar_bleu = bleu.compute(predictions=en2ar_predictions, references=en2ar_references)
    print(f"English to Arabic BLEU: {en2ar_bleu['score']:.2f}")
    # Evaluate Arabic to English
    ar2en_predictions = []
    ar2en_references = []
    
    print(f"Evaluating Arabic to English translations (using {min(MAX_TEST_SAMPLES, len(ar2en_data))} samples)...")
    for item in ar2en_data[:MAX_TEST_SAMPLES]:  # Limit samples for speed
        source = item["input_text"].replace("ar2en: ", "")
        target = item["target_text"]
        prediction = translate(source, "ar", "en", model, tokenizer, device)
        ar2en_predictions.append(prediction)
        ar2en_references.append([target])
    
    ar2en_bleu = bleu.compute(predictions=ar2en_predictions, references=ar2en_references)
    print(f"Arabic to English BLEU: {ar2en_bleu['score']:.2f}")
    
    # Save evaluation results
    results = {
        "english_to_arabic_bleu": en2ar_bleu["score"],
        "arabic_to_english_bleu": ar2en_bleu["score"],
        "average_bleu": (en2ar_bleu["score"] + ar2en_bleu["score"]) / 2
    }
    
    # Save results as text file
    with open(os.path.join(RESULTS_DIR, "evaluation_results.txt"), "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    # Create sample translations dataframe
    sample_translations = []
    
    # Sample English to Arabic translations
    for i, (source, target, pred) in enumerate(zip(
        [item["input_text"].replace("en2ar: ", "") for item in en2ar_data[:5]],
        [item["target_text"] for item in en2ar_data[:5]],
        en2ar_predictions[:5]
    )):
        sample_translations.append({
            "direction": "English to Arabic",
            "source": source,
            "reference": target,
            "translation": pred        })
    
    # Sample Arabic to English translations
    for i, (source, target, pred) in enumerate(zip(
        [item["input_text"].replace("ar2en: ", "") for item in ar2en_data[:5]],
        [item["target_text"] for item in ar2en_data[:5]],
        ar2en_predictions[:5]
    )):
        sample_translations.append({
            "direction": "Arabic to English",
            "source": source,
            "reference": target,
            "translation": pred
        })
    
    # Save sample translations
    pd.DataFrame(sample_translations).to_csv(os.path.join(RESULTS_DIR, "sample_translations.csv"), index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=["English→Arabic", "Arabic→English", "Average"], 
                y=[results["english_to_arabic_bleu"], results["arabic_to_english_bleu"], results["average_bleu"]])
    plt.title("BLEU Scores for BART Translation Model")
    plt.ylabel("BLEU Score")
    plt.ylim(0, 100)
    plt.savefig(os.path.join(RESULTS_DIR, "bleu_scores.png"))
    
    return results



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BART translation model")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test set")
    
    args = parser.parse_args()
    
    if args.evaluate or not args:
        evaluate_model()
    else:
        parser.print_help()
