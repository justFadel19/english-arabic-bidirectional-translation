import csv
import os
import re

def clean_text(text):
    """Basic text cleaning."""
    return re.sub(r'\s+', ' ', text).strip()

def prepare_dataset():
    print("Preparing dataset for BART model fine-tuning...")
    
    # Check if input files exist
    if not os.path.exists('data/Tatoeba.ar-en.en') or not os.path.exists('data/Tatoeba.ar-en.ar'):
        print("Error: Source files not found. Please make sure Tatoeba files exist in the data folder.")
        return
    
    # Read source files
    with open('data/Tatoeba.ar-en.en', 'r', encoding='utf-8') as en_file, \
         open('data/Tatoeba.ar-en.ar', 'r', encoding='utf-8') as ar_file:
        
        english_lines = en_file.readlines()
        arabic_lines = ar_file.readlines()
    
    # Handle mismatched line counts
    if len(arabic_lines) != len(english_lines):
        print(f"Warning: Line count mismatch! Arabic: {len(arabic_lines)}, English: {len(english_lines)}")
        min_len = min(len(arabic_lines), len(english_lines))
        arabic_lines = arabic_lines[:min_len]
        english_lines = english_lines[:min_len]
    
    print(f"Loaded {len(arabic_lines)} parallel sentences")
        
    # Create bidirectional dataset with proper BART format
    output = []
    
    # Add the header row
    output.append(['input_text', 'target_text', 'direction'])
    
    en2ar_count = 0
    ar2en_count = 0
    
    # Process each pair of sentences
    for en, ar in zip(english_lines, arabic_lines):
        en = clean_text(en.strip())
        ar = clean_text(ar.strip())
        
        if en and ar:  # Make sure neither string is empty
            # Format for BART: English to Arabic
            output.append([f"en2ar: {en}", ar, "en2ar"])
            en2ar_count += 1
            
            # Format for BART: Arabic to English
            output.append([f"ar2en: {ar}", en, "ar2en"])
            ar2en_count += 1
    
    # Save to TSV file
    with open('data/translation_dataset.tsv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(output)
    
    print(f"Dataset preparation complete: {len(output)-1} translation pairs created")
    print(f"English to Arabic: {en2ar_count} samples")
    print(f"Arabic to English: {ar2en_count} samples") 
    print("File saved as: data/translation_dataset.tsv")

if __name__ == "__main__":
    prepare_dataset()
