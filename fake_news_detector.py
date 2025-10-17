#!/usr/bin/env python3
"""
Fake News Detector - Linear SVM Model
One-click training and prediction system
"""

import pandas as pd
import numpy as np
import re
import joblib
import nltk
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self, data_dir='notebooks'):
        """Initialize the Fake News Detector"""
        self.data_dir = Path(data_dir)
        self.model = None
        self.vectorizer = None
        self.model_path = Path('fake_news_detector_svm.joblib')
        
        # Download required NLTK data
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        print("📦 Setting up NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            print("✅ NLTK data ready!\n")
        except Exception as e:
            print(f"⚠️ NLTK setup warning: {e}\n")
    
    def _clean_text(self, text):
        """Clean text data"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters (keep apostrophes)
        text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)
        
        return text
    
    def load_and_preprocess_data(self):
        """Load and preprocess the datasets"""
        print("="*60)
        print("📊 LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        try:
            # Load datasets
            print("Loading datasets...")
            fake_df = pd.read_csv(self.data_dir / 'Fake.csv')
            fake_df['label'] = 0
            
            true_df = pd.read_csv(self.data_dir / 'True.csv')
            true_df['label'] = 1
            
            print(f"✅ Fake news: {len(fake_df):,} articles")
            print(f"✅ True news: {len(true_df):,} articles")
            
            # Combine datasets
            df = pd.concat([fake_df, true_df], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"✅ Combined dataset: {len(df):,} articles\n")
            
            # Clean text
            print("🧹 Cleaning text data...")
            df['title_clean'] = df['title'].apply(self._clean_text)
            df['text_clean'] = df['text'].apply(self._clean_text)
            df['combined_text'] = df['title_clean'] + ' ' + df['text_clean']
            
            print("✅ Text cleaning completed!\n")
            
            return df
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find data files in {self.data_dir}")
            print(f"   Please ensure Fake.csv and True.csv are in the {self.data_dir} directory")
            raise
    
    def train_model(self, df):
        """Train the Linear SVM model"""
        print("="*60)
        print("🚀 TRAINING LINEAR SVM MODEL")
        print("="*60)
        
        # Prepare data
        X = df['combined_text']
        y = df['label']
        
        # Train-test split
        print("📊 Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}\n")
        
        # TF-IDF Vectorization
        print("🔤 Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            stop_words='english',
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"✅ TF-IDF features: {X_train_tfidf.shape}")
        print(f"   Vocabulary size: {len(self.vectorizer.get_feature_names_out()):,}\n")
        
        # Train model
        print("🤖 Training Linear SVM...")
        self.model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=2000,
            dual=False
        )
        
        import time
        start_time = time.time()
        self.model.fit(X_train_tfidf, y_train)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds!\n")
        
        # Evaluate model
        print("📈 EVALUATING MODEL")
        print("-"*60)
        
        y_pred = self.model.predict(X_test_tfidf)
        y_scores = self.model.decision_function(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🎯 ROC-AUC Score: {roc_auc:.4f}\n")
        
        print("📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Fake News', 'True News']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("🔍 Confusion Matrix:")
        print(f"                Predicted")
        print(f"             Fake    True")
        print(f"Actual Fake  {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"      True   {cm[1,0]:5d}   {cm[1,1]:5d}\n")
        
        # Cross-validation
        print("🔄 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(self.model, X_train_tfidf, y_train, 
                                    cv=5, scoring='f1_macro')
        print(f"   CV Scores: {cv_scores}")
        print(f"   Mean: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"   Stability: {'Excellent' if cv_scores.std() < 0.01 else 'Good'}\n")
        
        return accuracy, roc_auc, training_time
    
    def save_model(self, accuracy, roc_auc, training_time):
        """Save the trained model"""
        print("💾 Saving model...")
        
        from datetime import datetime
        
        model_package = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'training_time': training_time,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Linear SVM',
            'feature_count': len(self.vectorizer.get_feature_names_out())
        }
        
        joblib.dump(model_package, self.model_path)
        print(f"✅ Model saved to: {self.model_path}\n")
    
    def load_model(self):
        """Load a trained model"""
        if not self.model_path.exists():
            print(f"❌ Model file not found: {self.model_path}")
            print("   Please train a model first using: python fake_news_detector.py train")
            return False
        
        print("📦 Loading trained model...")
        model_package = joblib.load(self.model_path)
        
        self.model = model_package['model']
        self.vectorizer = model_package['vectorizer']
        
        print(f"✅ Model loaded successfully!")
        print(f"   Accuracy: {model_package['accuracy']:.4f}")
        print(f"   ROC-AUC: {model_package['roc_auc']:.4f}")
        print(f"   Created: {model_package['created_date']}\n")
        
        return True
    
    def predict(self, title, text):
        """Predict if a news article is fake or real"""
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                return None
        
        # Clean and combine text
        combined_text = f"{title} {text}"
        combined_text = self._clean_text(combined_text)
        
        # Transform and predict
        text_tfidf = self.vectorizer.transform([combined_text])
        prediction = self.model.predict(text_tfidf)[0]
        confidence = abs(self.model.decision_function(text_tfidf)[0])
        
        result = {
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': confidence,
            'label': prediction
        }
        
        return result
    
    def interactive_predict(self):
        """Interactive prediction interface"""
        print("="*60)
        print("🔍 FAKE NEWS DETECTOR - INTERACTIVE MODE")
        print("="*60)
        print("Enter 'quit' to exit\n")
        
        while True:
            try:
                title = input("📰 Article Title: ").strip()
                
                if title.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                text = input("📄 Article Text: ").strip()
                
                if not title or not text:
                    print("⚠️ Please provide both title and text\n")
                    continue
                
                # Make prediction
                result = self.predict(title, text)
                
                if result:
                    print(f"\n{'='*60}")
                    print(f"🎯 PREDICTION: {result['prediction']}")
                    print(f"📊 Confidence: {result['confidence']:.4f}")
                    
                    if result['confidence'] > 2.0:
                        certainty = "Very High"
                    elif result['confidence'] > 1.0:
                        certainty = "High"
                    elif result['confidence'] > 0.5:
                        certainty = "Medium"
                    else:
                        certainty = "Low"
                    
                    print(f"✨ Certainty: {certainty}")
                    print(f"{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def main():
    """Main execution function"""
    import sys
    
    detector = FakeNewsDetector()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Training mode
        print("🚀 FAKE NEWS DETECTOR - TRAINING MODE\n")
        
        # Load and preprocess data
        df = detector.load_and_preprocess_data()
        
        # Train model
        accuracy, roc_auc, training_time = detector.train_model(df)
        
        # Save model
        detector.save_model(accuracy, roc_auc, training_time)
        
        print("="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🎯 ROC-AUC: {roc_auc:.4f}")
        print(f"⚡ Training Time: {training_time:.2f} seconds")
        print("\nYou can now use the model for predictions!")
        print("Run: python fake_news_detector.py predict\n")
        
    elif len(sys.argv) > 1 and sys.argv[1] == 'predict':
        # Prediction mode
        detector.interactive_predict()
        
    else:
        # Show usage
        print("="*60)
        print("🔍 FAKE NEWS DETECTOR - LINEAR SVM")
        print("="*60)
        print("\nUsage:")
        print("  python fake_news_detector.py train     # Train the model")
        print("  python fake_news_detector.py predict   # Make predictions")
        print("\nFirst-time users should run 'train' to create the model.\n")


if __name__ == "__main__":
    main()

