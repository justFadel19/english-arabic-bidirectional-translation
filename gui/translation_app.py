import os
import sys
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "bart-english-arabic-final")

# Model parameters
MAX_SOURCE_LENGTH = 64
MAX_TARGET_LENGTH = 64

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("English-Arabic Translator")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize model and tokenizer
        self.load_model_thread = None
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Create UI
        self.create_widgets()
        
        # Start loading model
        self.load_model()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Translation direction selector
        direction_frame = ttk.Frame(main_frame)
        direction_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(direction_frame, text="Translation Direction:").pack(side=tk.LEFT)
        
        self.direction_var = tk.StringVar(value="en2ar")
        en2ar_radio = ttk.Radiobutton(direction_frame, text="English → Arabic", 
                                      variable=self.direction_var, value="en2ar")
        ar2en_radio = ttk.Radiobutton(direction_frame, text="Arabic → English", 
                                      variable=self.direction_var, value="ar2en")
        
        en2ar_radio.pack(side=tk.LEFT, padx=10)
        ar2en_radio.pack(side=tk.LEFT)
        
        # Paned window for input and output
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Input frame
        input_frame = ttk.LabelFrame(paned, text="Input Text")
        paned.add(input_frame, weight=1)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, 
                                                  width=40, height=10, font=("Arial", 12))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.translate_btn = ttk.Button(button_frame, text="Translate", 
                                       command=self.translate_text, state=tk.DISABLED)
        self.translate_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Loading model...")
        status_label = ttk.Label(button_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(paned, text="Translation")
        paned.add(output_frame, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, 
                                                   width=40, height=10, font=("Arial", 12))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)
    
    def load_model(self):
        """Load the BART model and tokenizer."""
        try:
            self.status_var.set("Loading model...")
            # Check if model exists
            if not os.path.exists(MODEL_PATH):
                self.status_var.set("Model not found. Please train the model first.")
                messagebox.showerror("Model Not Found", 
                                    f"Could not find the model at {MODEL_PATH}. "
                                    "Please train the model first.")
                return
            
            # Load tokenizer and model with optimizations
            try:
                # Try to load with better performance settings
                self.tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
                
                # Set device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Load with optimizations for inference
                self.model = BartForConditionalGeneration.from_pretrained(
                    MODEL_PATH,
                    torchscript=True if torch.cuda.is_available() else False,  # Enable TorchScript for GPU
                    low_cpu_mem_usage=True  # More memory-efficient loading
                )
                
                # Move model to device
                self.model.to(self.device)
                
                # Extra optimization for inference
                if torch.cuda.is_available():
                    self.model = self.model.half()  # Use half precision on GPU for faster inference
                
                self.model.eval()  # Set to evaluation mode
                
            except Exception as e:
                # Fallback to standard loading
                print(f"Optimized loading failed, using standard loading: {str(e)}")
                self.tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
                self.model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
            
            # Update UI
            self.translate_btn.config(state=tk.NORMAL)
            device_name = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
            self.status_var.set(f"Model loaded (using {device_name})")
            
        except Exception as e:
            self.status_var.set("Error loading model")
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
    
    def translate_text(self):
        """Translate text using the loaded model."""
        if not self.model or not self.tokenizer:
            messagebox.showerror("Error", "Model not loaded yet")
            return
        
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            return
        
        try:
            self.status_var.set("Translating...")
            self.root.update()
            
            # Determine translation direction
            direction = self.direction_var.get()
            source_lang = "en" if direction == "en2ar" else "ar"
            target_lang = "ar" if direction == "en2ar" else "en"
            
            # Create the input text with the appropriate prefix
            if source_lang == "en" and target_lang == "ar":
                prefix_text = f"en2ar: {input_text}"
            else:
                prefix_text = f"ar2en: {input_text}"
            
            # Tokenize the input text
            inputs = self.tokenizer(prefix_text, return_tensors="pt", 
                                   max_length=MAX_SOURCE_LENGTH, padding="max_length", truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Generate translation
            with torch.no_grad():  # No gradient tracking for inference
                outputs = self.model.generate(
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
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Update output text
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, translation)
            self.output_text.config(state=tk.DISABLED)
            
            # Update status
            self.status_var.set("Translation complete")
            
        except Exception as e:
            self.status_var.set("Translation error")
            messagebox.showerror("Error", f"Translation error: {str(e)}")
    
    def clear_text(self):
        """Clear input and output text areas."""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.status_var.set("Ready")

def main():
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
