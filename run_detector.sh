#!/bin/bash
# Fake News Detector - Quick Launch Script

echo "🔍 Fake News Detector - Quick Launch"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if required files exist
if [ ! -f "notebooks/Fake.csv" ] || [ ! -f "notebooks/True.csv" ]; then
    echo "❌ Error: Data files not found"
    echo "Please ensure Fake.csv and True.csv are in the notebooks/ directory"
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Installation complete!"
else
    source venv/bin/activate
fi

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import pandas, numpy, sklearn, nltk, joblib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Dependencies missing. Installing..."
    pip install pandas numpy scikit-learn nltk joblib
fi

# Check if model exists
if [ ! -f "fake_news_detector_svm.joblib" ]; then
    echo ""
    echo "⚠️ No trained model found. Training new model..."
    echo ""
    python3 fake_news_detector.py train
fi

# Launch GUI
echo ""
echo "🚀 Launching Fake News Detector GUI..."
echo ""
python3 fake_news_detector_gui.py

