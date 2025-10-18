# Fake News Detection - Group 9

A complete machine learning project for detecting fake news with **99.78% accuracy**. This project includes a production-ready application with GUI, comprehensive model comparisons, and detailed research notebooks.

## 🎯 Project Overview

This project explores multiple approaches to fake news detection and provides a one-click application for real-world use:

- **Production Application**: Easy-to-use GUI and command-line tool
- **Best Model**: Linear SVM with 99.78% accuracy, <1 second training time
- **Multiple Models Tested**: Logistic Regression, SVM, Naive Bayes, CNN, Deep Learning
- **Complete Research**: Detailed model comparisons and performance analysis
- **Large Dataset**: 44,898 articles (23,481 fake, 21,417 real news)

## ✨ Key Features

- 🚀 **One-Click Launch**: Simple scripts for Mac/Windows
- 🎯 **99.78% Accuracy**: State-of-the-art performance
- ⚡ **Lightning Fast**: Train in 0.22 seconds, predict in <0.01 seconds
- 🖥️ **User-Friendly GUI**: No coding required
- 📊 **Comprehensive Analysis**: Complete model comparison with visualizations
- 🔬 **Research-Ready**: Multiple Jupyter notebooks with experiments

## 📁 Project Structure

```
fake-news-group9/
├── README.md                           # This file
├── QUICK_START.md                      # Detailed user guide
├── requirements.txt                    # Python dependencies
│
├── 🚀 Production Application
├── fake_news_detector.py               # Core detection engine (CLI)
├── fake_news_detector_gui.py           # GUI application (Tkinter)
├── fake_news_detector_svm.joblib       # Pre-trained model (99.78% accuracy)
├── run_detector.sh                     # Mac/Linux launcher
├── run_detector.bat                    # Windows launcher
├── install_dependencies.sh             # Dependency installer
├── quick_test.py                       # Quick testing script
│
└── 📊 Research & Notebooks
    └── notebooks/
        ├── Fake.csv                              # Fake news dataset (23,481 articles)
        ├── True.csv                              # Real news dataset (21,417 articles)
        │
        ├── 📈 Model Implementations
        ├── 01_linearSVM.ipynb                    # ⭐ Best Model: Linear SVM (99.78%)
        ├── 02_logisticReg.ipynb                  # Logistic Regression (99.40%)
        ├── Naive Bayes Execution.ipynb           # Naive Bayes (95-96%)
        ├── s2_preprocess_cnn.ipynb               # CNN Deep Learning
        │
        ├── 🔬 Preprocessing & Pipelines
        ├── 01_preprocess.ipynb                   # Data preprocessing pipeline
        ├── s1_preprocess_LogisticRegression.ipynb # Alternative pipeline
        ├── s1_no_reuters_pipeline.ipynb          # Pipeline without Reuters
        │
        └── 📊 Analysis & Results
            ├── Model_Comparison_Analysis.md      # Complete model comparison
            ├── Model_Results_Comparison.ipynb    # Interactive results analysis
            ├── Model_Results_Comparison.html     # Results visualization
            └── Model_Results_Comparison.pptx     # Presentation slides
```

## 🚀 Quick Start

### Option 1: One-Click Launch (Recommended)

**Mac/Linux:**
```bash
chmod +x run_detector.sh
./run_detector.sh
```

**Windows:**
```bash
run_detector.bat
```

That's it! The script will:
1. ✅ Create virtual environment
2. ✅ Install dependencies
3. ✅ Download NLTK data
4. ✅ Train model (first time only)
5. ✅ Launch GUI application

### Option 2: Manual Setup

1. **Clone and navigate:**
```bash
git clone https://github.com/krati-2024aiml121/fake-news-group9.git
cd fake-news-group9
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Train the model:**
```bash
python fake_news_detector.py train
```

5. **Launch GUI:**
```bash
python fake_news_detector_gui.py
```

For detailed instructions, see [QUICK_START.md](QUICK_START.md)

## 💻 Usage

### GUI Application (Easiest)

1. Launch the application using `./run_detector.sh` (Mac/Linux) or `run_detector.bat` (Windows)
2. Enter article title and text
3. Click "Analyze Article"
4. View results: FAKE (red) or REAL (green) with confidence score

**Example Test - Fake News:**
- **Title:** `Scientists Shocked: Earth is Actually Flat, NASA Admits!`
- **Text:** `In a shocking revelation today, NASA scientists have finally admitted...`
- **Result:** FAKE with Very High confidence

**Example Test - Real News:**
- **Title:** `Federal Reserve Announces Interest Rate Decision`
- **Text:** `Washington (Reuters) - The Federal Reserve announced on Wednesday...`
- **Result:** REAL with High confidence

### Command Line Interface

**Interactive Predictions:**
```bash
python fake_news_detector.py predict
```

**Training:**
```bash
python fake_news_detector.py train
```

### Python API

```python
from fake_news_detector import FakeNewsDetector

# Initialize and load model
detector = FakeNewsDetector()
detector.load_model()

# Make prediction
result = detector.predict(
    title="Your Article Title",
    text="Your article text..."
)

print(f"Prediction: {result['prediction']}")  # FAKE or REAL
print(f"Confidence: {result['confidence']:.4f}")
```

## 📊 Model Performance

### Best Model: Linear SVM

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.78% |
| **ROC-AUC Score** | 1.0000 (Perfect) |
| **Training Time** | 0.22 seconds |
| **Prediction Time** | <0.01 seconds |
| **Model Size** | ~8 MB |
| **Dataset Size** | 44,898 articles |

**Cross-Validation (5-Fold):**
- Mean Score: 99.38% ± 0.13%
- Stability: Excellent

### Model Comparison

| Model | Accuracy | ROC-AUC | Training Time | Notes |
|-------|----------|---------|---------------|-------|
| **Linear SVM** ⭐ | **99.78%** | **1.0000** | **0.22s** | Best overall |
| Logistic Regression | 99.40% | 0.998 | Fast | Good interpretability |
| Naive Bayes | 95-96% | Good | Very Fast | Baseline |
| CNN Deep Learning | High | High | Slow | Complex patterns |

See [Model_Comparison_Analysis.md](notebooks/Model_Comparison_Analysis.md) for detailed analysis.

## 🔬 Research Notebooks

### Model Implementations
- **[01_linearSVM.ipynb](notebooks/01_linearSVM.ipynb)** - Best performing model (99.78% accuracy)
- **[02_logisticReg.ipynb](notebooks/02_logisticReg.ipynb)** - Logistic Regression approach
- **[Naive Bayes Execution.ipynb](notebooks/Naive%20Bayes%20Execution.ipynb)** - Probabilistic classifier
- **[s2_preprocess_cnn.ipynb](notebooks/s2_preprocess_cnn.ipynb)** - Deep learning CNN models

### Preprocessing & Analysis
- **[01_preprocess.ipynb](notebooks/01_preprocess.ipynb)** - Data preprocessing pipeline
- **[Model_Results_Comparison.ipynb](notebooks/Model_Results_Comparison.ipynb)** - Interactive analysis
- **[Model_Comparison_Analysis.md](notebooks/Model_Comparison_Analysis.md)** - Complete comparison

## 🎯 How It Works

### Feature Detection

**Fake News Indicators:**
- ❌ Sensational language (excessive !, ?)
- ❌ Lack of credible sources
- ❌ Informal tone ("just", "people", "shocking")
- ❌ Social media references
- ❌ Emotional manipulation

**Real News Indicators:**
- ✅ Source attribution ("Reuters", "Associated Press")
- ✅ Temporal specificity (dates, locations)
- ✅ Formal language ("announced", "according to")
- ✅ Institutional references
- ✅ Professional journalistic style

### Technical Pipeline

1. **Text Cleaning**: Lowercase, remove URLs/emails/HTML tags
2. **TF-IDF Vectorization**: Extract 8,000 features with 1-2 n-grams
3. **Linear SVM Classification**: Balanced class weights, C=1.0
4. **Confidence Scoring**: Decision function distance from hyperplane

## 🛠️ Technical Details

### Requirements
- Python 3.8+
- 2GB RAM minimum
- 500MB disk space

### Dependencies
- pandas, numpy - Data processing
- scikit-learn - Machine learning
- nltk - Natural language processing
- joblib - Model serialization
- tkinter - GUI (included with Python)
- matplotlib, seaborn, plotly - Visualizations (for notebooks)
- torch, tensorflow - Deep learning (for CNN notebooks)

### Model Architecture
- **Algorithm**: Linear Support Vector Machine (LinearSVC)
- **Features**: 8,000 TF-IDF features (unigrams + bigrams)
- **Regularization**: C=1.0, balanced class weights
- **Training**: 35,918 samples (80% split)
- **Testing**: 8,980 samples (20% split)

## 📚 Dataset

- **Source**: Kaggle Fake News Dataset
- **Total Articles**: 44,898
  - Fake News: 23,481 articles
  - True News: 21,417 articles
- **Columns**: title, text, subject, date
- **Location**: `notebooks/Fake.csv` and `notebooks/True.csv`

## 🎓 Key Learnings

1. **Linear models** can outperform deep learning for text classification with proper feature engineering
2. **TF-IDF vectorization** with bigrams captures enough context for fake news detection
3. **99.78% accuracy** is achievable with simple, fast models
4. **Balanced datasets** and proper preprocessing are crucial
5. **Cross-validation** confirms model stability and generalization

## 🔧 Advanced Features

### Confidence Interpretation
- **>2.0**: Very High Confidence (Model is very certain)
- **1.0-2.0**: High Confidence (Model is confident)
- **0.5-1.0**: Medium Confidence (Reasonable prediction)
- **<0.5**: Low Confidence (Manual review recommended)

### Retraining with Custom Data
1. Add articles to `notebooks/Fake.csv` and `notebooks/True.csv`
2. Keep columns: `title`, `text`, `subject`, `date`
3. Run: `python fake_news_detector.py train`
4. New model saved automatically



## 👥 Team

- **Krati**
- **Nitin**
- **Karan**
- **Sumantha**

## 📄 Files Summary

### Core Application Files
- `fake_news_detector.py` - Main detection engine with training and prediction
- `fake_news_detector_gui.py` - Graphical user interface
- `fake_news_detector_svm.joblib` - Pre-trained Linear SVM model
- `run_detector.sh` / `run_detector.bat` - Cross-platform launchers
- `requirements.txt` - Python package dependencies

### Documentation
- `README.md` - This file (project overview)
- `QUICK_START.md` - Detailed usage guide
- `notebooks/Model_Comparison_Analysis.md` - Complete model analysis

### Research Notebooks
- `notebooks/01_linearSVM.ipynb` - Best model implementation
- `notebooks/02_logisticReg.ipynb` - Logistic Regression
- `notebooks/Naive Bayes Execution.ipynb` - Naive Bayes classifier
- `notebooks/s2_preprocess_cnn.ipynb` - CNN deep learning
- `notebooks/Model_Results_Comparison.ipynb` - Results visualization

## 🎯 Quick Commands Reference

```bash
# First time setup
./run_detector.sh              # Launch everything (Mac/Linux)
run_detector.bat               # Launch everything (Windows)

# Manual training
python fake_news_detector.py train

# Interactive predictions
python fake_news_detector.py predict

# Launch GUI
python fake_news_detector_gui.py

# Jupyter notebooks
jupyter notebook
# Then open notebooks/01_linearSVM.ipynb
```

## 🌟 Highlights

- ✅ **Production-Ready**: Complete GUI and CLI applications
- ✅ **State-of-the-Art**: 99.78% accuracy with Linear SVM
- ✅ **Fast**: 0.22s training, <0.01s prediction
- ✅ **User-Friendly**: One-click installation and launch
- ✅ **Well-Documented**: Comprehensive guides and notebooks
- ✅ **Reproducible**: All experiments documented in notebooks
- ✅ **Extensible**: Easy to retrain with new data

## 📖 Getting Started Paths

**For End Users:**
1. Read [QUICK_START.md](QUICK_START.md) - Non-technical guide
2. Run `./run_detector.sh` to launch GUI
3. Test with sample articles

**For Developers:**
1. Explore `fake_news_detector.py` for the core engine
2. Review `notebooks/01_linearSVM.ipynb` for model details
3. Use Python API for integration

**For Researchers:**
1. Open `notebooks/Model_Comparison_Analysis.md`
2. Review `notebooks/Model_Results_Comparison.ipynb`
3. Experiment with different models in notebooks

## 🙏 Acknowledgments

- **Dataset**: Kaggle Fake News Dataset
- **Libraries**: scikit-learn, NLTK, pandas, numpy
- **Inspiration**: State-of-the-art research in fake news detection
- **Tools**: Python, Jupyter, Tkinter

---

**🚀 Ready to detect fake news? Start with `./run_detector.sh` or see [QUICK_START.md](QUICK_START.md)!**
