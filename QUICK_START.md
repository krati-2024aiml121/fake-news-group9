# 🔍 Fake News Detector - Quick Start Guide

## ⭐ For Non-Technical Users (Start Here!)

### What This Tool Does
This application analyzes news articles and tells you whether they are likely **REAL** or **FAKE** with 99.78% accuracy. Just paste a headline and article text, click a button, and get instant results!

### Simple Steps to Get Started

#### **Step 1: Find and Open Your Terminal**
- On your Mac, press `Cmd + Space` (Command + Spacebar)
- Type "Terminal" and press Enter
- A black or white window will open - this is your Terminal

#### **Step 2: Navigate to the Project Folder**
In the Terminal, type this command and press Enter:
```bash
cd Desktop/capstone_bits/fake-news-group9
```

#### **Step 3: Make the Launch Script Ready** (First Time Only)
Type this command and press Enter:
```bash
chmod +x run_detector.sh
```
*Note: This makes the program executable. You only need to do this once.*

#### **Step 4: Launch the Application**
Type this command and press Enter:
```bash
./run_detector.sh
```

**What happens next:**
- ⏳ First time: The program installs necessary tools (takes 5-10 minutes)
- ⏳ It trains the AI model (takes 1-2 minutes)
- 🚀 A window opens with the Fake News Detector interface
- ✅ Future runs: Opens instantly (5-10 seconds)!

#### **Step 5: Use the Application**

When the window opens, you'll see:

1. **Title Box** (small text area at top)
   - Enter your news headline here
   
2. **Article Text Box** (large text area below)
   - Paste the full news article text here
   
3. **Analyze Article Button** (below the text boxes)
   - Click this to check if the news is fake or real

4. **Results Section** (bottom of window)
   - Shows whether the article is FAKE (red) or REAL (green)
   - Displays confidence score and explanation

#### **Example Test Article**

Try this fake news example to test the detector:

**Title:**
```
Scientists Shocked: Earth is Actually Flat, NASA Admits!
```

**Article Text:**
```
In a shocking revelation today, NASA scientists have finally admitted what conspiracy theorists have known all along - the Earth is actually flat! Government officials are scrambling to cover up this massive lie that has fooled billions of people. An insider source who wished to remain anonymous revealed these shocking details. This is what they don't want you to know! Share this before it gets deleted!!!
```

Click "Analyze Article" and watch it detect this as **FAKE**!

Now try a real news example:

**Title:**
```
Federal Reserve Announces Interest Rate Decision
```

**Article Text:**
```
Washington (Reuters) - The Federal Reserve announced on Wednesday that it would maintain interest rates at their current levels, citing concerns about inflation and economic stability. Fed Chair Jerome Powell stated in a press conference that the committee carefully reviewed economic indicators before reaching this decision. The announcement came after a two-day policy meeting in Washington. Market analysts had widely expected this outcome based on recent economic data.
```

Click "Analyze Article" and watch it detect this as **REAL**!

#### **Understanding the Results**

After clicking "Analyze Article," you'll see:

- **PREDICTION:** Either "FAKE" (in red) or "REAL" (in green)
- **Confidence Score:** A number indicating how certain the AI is (higher = more certain)
  - Above 2.0 = Very confident
  - 1.0-2.0 = Confident
  - 0.5-1.0 = Moderate confidence
  - Below 0.5 = Low confidence (manually verify)
- **Interpretation:** Explanation of why the article was classified this way

#### **Buttons Explained**

- **Analyze Article:** Click this to check your news article
- **Clear:** Clears all text boxes to start fresh

#### **Tips for Best Results**

✅ **Do:**
- Paste complete article text (at least a few sentences)
- Include the original headline
- Test with different news sources

❌ **Don't:**
- Submit just one or two words
- Leave title or text empty
- Expect perfect results on very short texts

#### **When You're Done**

- Simply close the application window
- To run it again, just repeat Step 4: `./run_detector.sh`

---

## 🔬 For Technical Users

### How to Run Locally

The code is written in **Python 3.8+**. If you don't have Python installed, you can find it [here](https://www.python.org/downloads/). Ensure you have Python 3.8 or higher installed.

1. **Clone the repo and `cd` into the folder**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the GUI app:**
   ```bash
   python fake_news_detector_gui.py
   ```
   Or use the launcher script: `./run_detector.sh` (Mac/Linux) or `run_detector.bat` (Windows)

### Review Research & Model Results

- `notebooks/Model_Comparison_Analysis.md` - Complete analysis of all models (Logistic Regression, Linear SVM, Naive Bayes, CNN)
- `notebooks/Model_Results_Comparison.ipynb` - Interactive results with visualizations
- `notebooks/01_linearSVM.ipynb` - Best model implementation (99.78% accuracy)
- `notebooks/01_preprocess.ipynb` - Data preprocessing pipeline

### Retrain Model

1. **Modify data (optional):** Edit `notebooks/Fake.csv` and `notebooks/True.csv` (keep columns: `title`, `text`, `subject`, `date`)
2. **Retrain:**
   ```bash
   python fake_news_detector.py train
   ```
3. Model saved as `fake_news_detector_svm.joblib`

**Key Files:** `fake_news_detector.py` (core engine), `fake_news_detector_gui.py` (GUI), `fake_news_detector_svm.joblib` (trained model)

### Project Structure

```
fake-news-group9/
├── fake_news_detector.py           # Core detection engine (training & prediction)
├── fake_news_detector_gui.py       # Tkinter GUI application
├── fake_news_detector_svm.joblib   # Trained Linear SVM model (99.78% accuracy)
├── requirements.txt                # Python dependencies
├── run_detector.sh                 # Mac/Linux launcher script
├── run_detector.bat                # Windows launcher script
├── install_dependencies.sh         # Dependency installation script
├── test_detector.py                # Unit tests
├── quick_test.py                   # Quick testing script
├── demo_prediction.txt             # Sample predictions output
└── notebooks/
    ├── Fake.csv                              # Fake news dataset (23,481 articles)
    ├── True.csv                              # Real news dataset (21,417 articles)
    ├── 01_preprocess.ipynb                   # Data preprocessing pipeline
    ├── 01_linearSVM.ipynb                    # Linear SVM model (best: 99.78%)
    ├── 02_logisticReg.ipynb                  # Logistic Regression model
    ├── Naive Bayes Execution.ipynb           # Naive Bayes classifier
    ├── s1_preprocess_LogisticRegression.ipynb # Alternative Logistic Regression
    ├── s1_no_reuters_pipeline.ipynb          # Pipeline without Reuters data
    ├── s2_preprocess_cnn.ipynb               # CNN deep learning model
    ├── Model_Results_Comparison.ipynb        # Interactive results & visualizations
    └── Model_Results_Comparison.html         # HTML export of results
```

---

## One-Click Installation and Usage

This package combines the preprocessing (`01_preprocess.ipynb`) and Linear SVM model (`01_linearSVM.ipynb`) into an easy-to-use application.

---

## 📦 Installation (First Time Only)

### Prerequisites
- Python 3.8 or higher installed
- Data files: `notebooks/Fake.csv` and `notebooks/True.csv`

### Option 1: Automatic Setup (Recommended)

#### For Mac/Linux:
```bash
chmod +x run_detector.sh
./run_detector.sh
```

#### For Windows:
```bash
run_detector.bat
```

**That's it!** The script will:
1. Create a virtual environment
2. Install all dependencies
3. Train the model (if not already trained)
4. Launch the GUI application

---

## 🚀 Usage

### Method 1: GUI Application (Easiest)

**Mac/Linux:**
```bash
./run_detector.sh
```

**Windows:**
```bash
run_detector.bat
```

**Using the GUI:**
1. Enter article title in the "Title" box
2. Enter article text in the "Article Text" box
3. Click "Analyze Article"
4. View results showing FAKE or REAL prediction with confidence score

![GUI Screenshot](gui_example.png)

---

### Method 2: Command Line

#### Train the Model
```bash
python fake_news_detector.py train
```

**Output:**
- Loads 44,898 articles (23,481 fake, 21,417 real)
- Trains Linear SVM in ~0.22 seconds
- Achieves 99.78% accuracy
- Saves model to `fake_news_detector_svm.joblib`

#### Interactive Predictions
```bash
python fake_news_detector.py predict
```

**Example:**
```
📰 Article Title: Scientists Discover Aliens on Moon!
📄 Article Text: Government has been hiding this shocking truth...

🎯 PREDICTION: FAKE
📊 Confidence: 2.7023
✨ Certainty: Very High
```

---

### Method 3: Python API

```python
from fake_news_detector import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Load trained model
detector.load_model()

# Predict
result = detector.predict(
    title="Federal Reserve Announces Interest Rate Decision",
    text="Washington (Reuters) - The Federal Reserve announced today..."
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.78% |
| **ROC-AUC** | 1.0000 (Perfect) |
| **Training Time** | 0.22 seconds |
| **Prediction Time** | < 0.01 seconds |
| **Model Size** | ~8 MB |

**Cross-Validation (5-Fold):**
- Mean Score: 99.38% ± 0.13%
- Stability: Excellent

---

## 🎯 What the Model Detects

### Fake News Indicators
- ❌ Sensational language (excessive !, ?)
- ❌ Lack of source attribution
- ❌ Informal tone ("just", "people", "video")
- ❌ Social media references ("pic twitter", "featured image")

### Real News Indicators
- ✅ Source attribution ("Reuters", "said")
- ✅ Temporal specificity (days of week, dates)
- ✅ Formal language ("President", "Washington")
- ✅ Institutional references

---

## 📁 Project Structure

```
fake-news-group9/
├── fake_news_detector.py           # Core detection engine
├── fake_news_detector_gui.py       # GUI application
├── run_detector.sh                 # Mac/Linux launcher
├── run_detector.bat                # Windows launcher
├── fake_news_detector_svm.joblib   # Trained model (created after training)
├── requirements.txt                # Python dependencies
├── notebooks/
│   ├── Fake.csv                    # Fake news dataset
│   ├── True.csv                    # Real news dataset
│   ├── 01_preprocess.ipynb         # Original preprocessing notebook
│   └── 01_linearSVM.ipynb          # Original model training notebook
└── QUICK_START.md                  # This file
```

---

## 🔧 Manual Installation (If Automatic Fails)

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 4: Train Model
```bash
python fake_news_detector.py train
```

### Step 5: Run Application
```bash
python fake_news_detector_gui.py
```

---

## 📋 Requirements

**Python Packages:**
- pandas
- numpy
- scikit-learn
- nltk
- joblib
- tkinter (usually included with Python)

**System Requirements:**
- Python 3.8+
- 2GB RAM minimum
- 500MB free disk space

---

## 🐛 Troubleshooting

### Error: "Data files not found"
**Solution:** Ensure `Fake.csv` and `True.csv` are in the `notebooks/` directory

### Error: "Module not found"
**Solution:** Run `pip install -r requirements.txt`

### Error: "Model file not found"
**Solution:** Run `python fake_news_detector.py train` first

### GUI doesn't open on Mac
**Solution:** Install tkinter: `brew install python-tk`

### Slow training on older computers
**Expected:** Training takes 0.2-2 seconds depending on hardware. This is normal.

---

## 💡 Tips

1. **First-time users:** Let the script train the model automatically
2. **Faster startup:** The model file is saved and reused (no retraining needed)
3. **Batch processing:** Use the Python API for analyzing multiple articles
4. **Confidence scores:** Higher scores (>2.0) indicate very confident predictions
5. **Low confidence (<0.5):** Manually review these articles

---

## 🎓 Technical Details

**Preprocessing Pipeline:**
1. Text cleaning (lowercase, remove URLs/emails/HTML)
2. TF-IDF vectorization (8,000 features, n-gram 1-2)
3. Feature normalization

**Model Architecture:**
- Algorithm: Linear Support Vector Machine (LinearSVC)
- Regularization: C=1.0, balanced class weights
- Training: 35,918 samples (80% of data)
- Testing: 8,980 samples (20% of data)

**Feature Extraction:**
- Vocabulary: 8,000 most important terms
- N-grams: Unigrams + Bigrams
- Stop words: Removed (English)
- Min document frequency: 3
- Max document frequency: 0.95

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the original notebooks: `01_preprocess.ipynb` and `01_linearSVM.ipynb`
3. Ensure all data files are present and Python version is correct

---

## 🎉 Success!

You now have a working fake news detector with:
- ✅ 99.78% accuracy
- ✅ Sub-second predictions
- ✅ Easy-to-use GUI
- ✅ Command-line interface
- ✅ Python API

**Start detecting fake news with a single click!**

