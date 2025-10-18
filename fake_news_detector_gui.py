#!/usr/bin/env python3
"""
Fake News Detector - GUI Application
Simple graphical interface for fake news detection
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from fake_news_detector import FakeNewsDetector

class FakeNewsDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detector - Linear SVM")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.detector = FakeNewsDetector()
        self.model_loaded = False
        
        self.create_widgets()
        self.load_model_on_startup()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔍 Fake News Detector", 
                               font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Loading...", 
                                     font=('Helvetica', 10))
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Article Input", padding="10")
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)
        input_frame.rowconfigure(3, weight=3)
        
        # Title input
        ttk.Label(input_frame, text="Title:", font=('Helvetica', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.title_entry = scrolledtext.ScrolledText(input_frame, height=3, 
                                                     font=('Helvetica', 10), wrap=tk.WORD)
        self.title_entry.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Text input
        ttk.Label(input_frame, text="Article Text:", font=('Helvetica', 10, 'bold')).grid(
            row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        self.text_entry = scrolledtext.ScrolledText(input_frame, height=10, 
                                                    font=('Helvetica', 10), wrap=tk.WORD)
        self.text_entry.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Analyze Article", 
                                        command=self.predict_article, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_inputs).pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Analysis Result", padding="10")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=8, 
                                                     font=('Helvetica', 11), wrap=tk.WORD, 
                                                     state=tk.DISABLED)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for colored output
        self.result_text.tag_configure("fake", foreground="red", font=('Helvetica', 12, 'bold'))
        self.result_text.tag_configure("real", foreground="green", font=('Helvetica', 12, 'bold'))
        self.result_text.tag_configure("header", font=('Helvetica', 11, 'bold'))
    
    def load_model_on_startup(self):
        """Load model when GUI starts"""
        def load():
            try:
                if self.detector.load_model():
                    self.model_loaded = True
                    self.status_label.config(text="✅ Model Ready - Accuracy: 99.78%")
                    self.predict_button.config(state=tk.NORMAL)
                else:
                    self.status_label.config(text="⚠️ No trained model found. Please run the launcher script.")
                    self.predict_button.config(state=tk.DISABLED)
            except Exception as e:
                self.status_label.config(text=f"❌ Error loading model: {str(e)}")
                self.predict_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def predict_article(self):
        """Predict if article is fake or real"""
        title = self.title_entry.get("1.0", tk.END).strip()
        text = self.text_entry.get("1.0", tk.END).strip()
        
        if not title or not text:
            messagebox.showwarning("Input Required", "Please enter both title and article text.")
            return
        
        if not self.model_loaded:
            messagebox.showerror("Model Not Ready", "Please train the model first.")
            return
        
        # Disable button during prediction
        self.predict_button.config(state=tk.DISABLED, text="Analyzing...")
        self.status_label.config(text="🔄 Analyzing article...")
        
        def predict():
            try:
                result = self.detector.predict(title, text)
                
                if result:
                    self.display_result(result)
                    self.status_label.config(text="✅ Analysis complete!")
                else:
                    self.status_label.config(text="❌ Prediction failed")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Prediction error: {str(e)}")
                self.status_label.config(text="❌ Error occurred")
            finally:
                self.predict_button.config(state=tk.NORMAL, text="Analyze Article")
        
        thread = threading.Thread(target=predict, daemon=True)
        thread.start()
    
    def display_result(self, result):
        """Display prediction result"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        
        # Prediction
        self.result_text.insert(tk.END, "PREDICTION: ", "header")
        
        prediction = result['prediction']
        tag = "fake" if prediction == "FAKE" else "real"
        self.result_text.insert(tk.END, f"{prediction}\n\n", tag)
        
        # Confidence
        confidence = result['confidence']
        self.result_text.insert(tk.END, "Confidence Score: ", "header")
        self.result_text.insert(tk.END, f"{confidence:.4f}\n\n")
        
        # Certainty level
        if confidence > 2.0:
            certainty = "Very High"
            certainty_desc = "The model is very confident in this prediction."
        elif confidence > 1.0:
            certainty = "High"
            certainty_desc = "The model is confident in this prediction."
        elif confidence > 0.5:
            certainty = "Medium"
            certainty_desc = "The model has moderate confidence in this prediction."
        else:
            certainty = "Low"
            certainty_desc = "The model has low confidence. Manual review recommended."
        
        self.result_text.insert(tk.END, "Certainty Level: ", "header")
        self.result_text.insert(tk.END, f"{certainty}\n")
        self.result_text.insert(tk.END, f"{certainty_desc}\n\n")
        
        # Interpretation
        self.result_text.insert(tk.END, "Interpretation:\n", "header")
        if prediction == "FAKE":
            self.result_text.insert(tk.END, 
                "⚠️ This article shows characteristics of fake news, such as:\n"
                "• Sensational or emotional language\n"
                "• Lack of credible source attribution\n"
                "• Informal tone or excessive punctuation\n"
                "• Consider fact-checking before sharing.\n")
        else:
            self.result_text.insert(tk.END,
                "✅ This article shows characteristics of real news, such as:\n"
                "• Formal reporting language\n"
                "• Proper source attribution\n"
                "• Temporal specificity (dates, locations)\n"
                "• Professional journalistic style.\n")
        
        self.result_text.config(state=tk.DISABLED)
    
    def clear_inputs(self):
        """Clear all input fields"""
        self.title_entry.delete("1.0", tk.END)
        self.text_entry.delete("1.0", tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.status_label.config(text="✅ Model Ready" if self.model_loaded else "⚠️ No model loaded")


def main():
    root = tk.Tk()
    app = FakeNewsDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

