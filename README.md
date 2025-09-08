# Fake News Detection - Group 9

A machine learning project for detecting fake news using various classification approaches. This project is designed for experimentation with different models through Jupyter notebooks.

## 🎯 Project Overview

This project explores multiple approaches to fake news detection:
- Traditional machine learning models (Logistic Regression, SVM, Random Forest, XGBoost, etc.)
- Deep learning models (LSTM, CNN, BERT, Transformers)
- Feature engineering and text preprocessing techniques
- Model comparison and performance analysis

## 📁 Project Structure

```
fake-news-group9/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_model1.ipynb
│   ├── Fake.csv          # Fake news dataset
│   └── True.csv          # Real news dataset
├── data/
│   ├── raw/
│   └── processed/
└── models/
    └── saved_models/
```

## 🚀 Quick Start

### Setup with Conda (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/kratig/fake-news-group9.git
cd fake-news-group9
```

2. **Create conda environment:**
```bash
conda create -n fake-news-detection python=3.9 -y
```

3. **Activate the environment:**
```bash
conda activate fake-news-detection
```

4. **Install dependencies:**

**Option A: Using requirements.txt**
```bash
pip install -r requirements.txt
```


5. **Start Jupyter Server:**
```bash
jupyter notebook
```
or
```bash
jupyter lab
```

6. **Navigate to notebooks:**
   - Open your browser and go to the Jupyter interface
   - Navigate to `notebooks/01_model1.ipynb` and start experimenting!

### Alternative Setup with Virtual Environment

If you prefer using Python's built-in venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

### Additional Setup for NLP Libraries

If you encounter issues with NLTK or spaCy, run these commands in a notebook cell:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# For spaCy (if using):
# !python -m spacy download en_core_web_sm
```

### Usage

1. **Start with `01_model1.ipynb`**: This contains your first model implementation
2. **Data files**: The datasets (`Fake.csv` and `True.csv`) are available in the notebooks directory
3. **Experiment**: Create additional notebooks for different model approaches
4. **Save models**: Use the `models/saved_models/` directory for trained models

## 📊 Models to Explore

### Classical Machine Learning
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes


### Deep Learning
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional LSTM)
- CNN (Convolutional Neural Networks)
- CNN-LSTM Hybrid

### Transformer Models
- BERT (Bidirectional Encoder Representations from Transformers)
- RoBERTa
- DistilBERT
- ALBERT

### Ensemble Methods
- Voting Classifier
- Stacking
- Bagging

## 🔧 Features

- **Text Preprocessing**: Text cleaning, normalization, and tokenization
- **Feature Engineering**: TF-IDF, Word2Vec, GloVe embeddings, BERT embeddings
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, ROC-AUC
- **Visualization**: Model performance plots, confusion matrices, word clouds
- **Hyperparameter Tuning**: Grid search and random search for optimization

## 📈 Workflow

1. **Data Loading**: Load and explore your fake news dataset
2. **Preprocessing**: Clean and prepare text data for modeling
3. **Feature Engineering**: Extract relevant features from text
4. **Model Training**: Train different types of models
5. **Evaluation**: Compare model performance using various metrics
6. **Model Selection**: Choose the best performing model
7. **Prediction**: Make predictions on new data

## 🎯 Getting Started

1. **Setup environment**: Follow the installation steps above
2. **Open Jupyter**: Start with `jupyter notebook` or `jupyter lab`
3. **Run the notebook**: Navigate to `notebooks/01_model1.ipynb`
4. **Load data**: The datasets (`Fake.csv` and `True.csv`) are already in the notebooks directory
5. **Experiment**: Run cells step by step and modify as needed
6. **Create new notebooks**: Add more notebooks for different model approaches
7. **Save results**: Store trained models in `models/saved_models/` directory

### Quick Commands Summary:
```bash
# Navigate to project
cd fake-news-group9

# Setup conda environment
conda create -n fake-news-detection python=3.9 -y
conda activate fake-news-detection

# Install packages
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Open notebooks/01_model1.ipynb and start coding!
```

## 📝 Notes

- Each notebook is self-contained and can be run independently
- Models are saved automatically for later use
- All required libraries are listed in `requirements.txt`
- Experiment tracking and comparison built into the notebooks

## 👥 Team

- **Krati Bansal** - Project Lead - kratibansal2006@gmail.com
- **Nitin**
- **Karan**
- **Sumantha**

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing ML libraries
- Inspired by state-of-the-art research in fake news detection