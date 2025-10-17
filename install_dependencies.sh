#!/bin/bash
# Install Dependencies for Fake News Detector

echo "📦 Installing Fake News Detector Dependencies"
echo "=============================================="
echo ""

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo ""
echo "Installing required packages..."
echo ""

# Install packages directly
pip3 install pandas numpy scikit-learn nltk joblib matplotlib seaborn

echo ""
echo "✅ Installation complete!"
echo ""
echo "Testing imports..."
python3 -c "
import pandas as pd
import numpy as np
import sklearn
import nltk
import joblib
print('✅ All packages imported successfully!')
print(f'   pandas: {pd.__version__}')
print(f'   numpy: {np.__version__}')
print(f'   scikit-learn: {sklearn.__version__}')
"

echo ""
echo "Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print('✅ NLTK data downloaded!')
"

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   python3 fake_news_detector.py train"
echo "   python3 fake_news_detector_gui.py"

