# NLP-Based Sentiment Analysis System

## Project Overview

This project implements a beginner-level Artificial Intelligence system focused on Natural Language Processing (NLP). The system automatically classifies textual movie reviews as **Positive** or **Negative** using machine learning techniques.

### Objective

To build an AI system that can understand textual data and accurately predict sentiment polarity using NLP and machine learning models without relying on deep learning frameworks or external AI APIs.

---

## Dataset Description

### Dataset: IMDb Large Movie Review Dataset

- **Dataset Name**: IMDb Large Movie Review Dataset
- **Source**: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size**:
  - **Total**: Approximately 50,000 labeled text reviews
  - **Positive Reviews**: 25,000
  - **Negative Reviews**: 25,000
  - **Training Set**: 40,000 reviews (80%)
  - **Testing Set**: 10,000 reviews (20%)
- **Classes**:
  - **Positive**: Reviews with sentiment score ≥ 7
  - **Negative**: Reviews with sentiment score ≤ 4
- **Data Format**: CSV file with two columns:
  - `review`: The text content of the movie review
  - `sentiment`: Binary label (positive / negative)

### Data Distribution

| Set            | Positive   | Negative   | Total      |
| -------------- | ---------- | ---------- | ---------- |
| Training (80%) | 20,000     | 20,000     | 40,000     |
| Testing (20%)  | 5,000      | 5,000      | 10,000     |
| **Total**      | **25,000** | **25,000** | **50,000** |

**Note**: The dataset uses stratified splitting to ensure balanced representation of both classes in training and testing sets.

---

## Methodology

### 1. Text Preprocessing

The following preprocessing steps are applied to clean the text data:

1. **Lowercase Conversion**: All text is converted to lowercase for consistency
2. **HTML Tag Removal**: Removes any HTML tags present in web-scraped data
3. **URL Removal**: Removes URLs from the text
4. **Punctuation Removal**: Removes all punctuation and special characters
5. **Stop Word Removal**: Common English stop words are removed using NLTK
6. **Tokenization**: Text is split into individual tokens (words)

### 2. Feature Extraction

Two feature extraction techniques are implemented:

#### Bag of Words (BoW)

- Counts the frequency of each word in the document
- Creates a sparse matrix of word counts
- Simple but effective for text classification

#### TF-IDF (Term Frequency-Inverse Document Frequency)

- Weights words based on their importance
- Reduces the weight of common words
- Increases the weight of rare, informative words

### 3. Machine Learning Models

Three classification algorithms are trained and evaluated:

#### Naive Bayes (MultinomialNB)

- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast and efficient for text classification

#### Logistic Regression

- Linear classifier that predicts class probabilities
- Highly interpretable
- Works well with high-dimensional sparse data

#### Support Vector Machine (LinearSVC)

- Finds optimal hyperplane separating classes
- Maximizes margin between classes
- Effective for high-dimensional text data

---

## Results and Evaluation Metrics

### Evaluation Metrics Used

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among predicted positives
- **Recall**: Proportion of true positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of prediction results

### Model Comparison

| Model                   | Feature      | Accuracy   | Precision  | Recall     | F1-Score   |
| ----------------------- | ------------ | ---------- | ---------- | ---------- | ---------- |
| Naive Bayes             | Bag of Words | 84.99%     | 84.69%     | 85.42%     | 85.05%     |
| Naive Bayes             | TF-IDF       | 85.86%     | 84.96%     | 87.14%     | 86.04%     |
| Logistic Regression     | Bag of Words | 87.02%     | 86.58%     | 87.62%     | 87.10%     |
| **Logistic Regression** | **TF-IDF**   | **88.73%** | **88.11%** | **89.54%** | **88.82%** |
| SVM                     | Bag of Words | 86.37%     | 86.23%     | 86.56%     | 86.40%     |
| SVM                     | TF-IDF       | 87.89%     | 87.24%     | 88.76%     | 87.99%     |

### Key Results Summary

**🏆 Best Performing Model: Logistic Regression with TF-IDF**

- **Accuracy**: 88.73%
  - Correctly classified 8,873 out of 10,000 test reviews
- **Precision**: 88.11%
  - When model predicts "Positive", it's correct 88.11% of the time
- **Recall**: 89.54%
  - Successfully identifies 89.54% of actual positive reviews
- **F1-Score**: 88.82%
  - Excellent balanced performance across precision and recall

**Model Performance Insights:**

1. **TF-IDF outperforms Bag of Words** across all models

   - Average improvement: +1.5% accuracy
   - TF-IDF better captures sentiment-bearing words by reducing common word importance

2. **Logistic Regression is most effective** for this task

   - Superior to Naive Bayes by ~3% accuracy
   - Better than SVM by ~1% accuracy
   - Efficient and interpretable for high-dimensional text data

3. **All models achieve >84% accuracy**
   - Demonstrates task feasibility with traditional ML approaches
   - No need for deep learning for this dataset

## Model Training Results & Visualizations

### Sentiment Distribution

![Distribution of Sentiment Classes](images/Distribution%20of%20Sentiment%20Classes.png)

The dataset contains balanced distribution of positive and negative reviews:

- **Training Set (80%)**: 20,000 positive + 20,000 negative = 40,000 reviews
- **Testing Set (20%)**: 5,000 positive + 5,000 negative = 10,000 reviews

### Model Performance Comparison

![Model Performance Comparison](images/Model%20Performance%20Comparison.png)

The chart above shows comparative performance across all six model-feature combinations. Logistic Regression with TF-IDF achieves the highest performance across all metrics.

### Confusion Matrices

#### Naive Bayes Models

**Naive Bayes with Bag of Words**

![Confusion Matrix - Naive Bayes (BoW)](<images/Confusion%20Matrix%20-%20Naive%20Bayes%20(BoW).png>)

**Naive Bayes with TF-IDF**

![Confusion Matrix - Naive Bayes (TF-IDF)](<images/Confusion%20Matrix%20-%20Naive%20Bayes%20(TF-IDF).png>)

#### Logistic Regression Models

**Logistic Regression with Bag of Words**

![Confusion Matrix - Logistic Regression (BoW)](<images/Confusion%20Matrix%20-%20Logistic%20Regression%20(BoW).png>)

**Logistic Regression with TF-IDF (Best Model)**

![Confusion Matrix - Logistic Regression (TF-IDF)](<images/Confusion%20Matrix%20-%20Logistic%20Regression%20(TF-IDF).png>)

#### Support Vector Machine Models

**SVM with Bag of Words**

![Confusion Matrix - SVM (BoW)](<images/Confusion%20Matrix%20-%20SVM%20(BoW).png>)

**SVM with TF-IDF**

![Confusion Matrix - SVM (TF-IDF)](<images/Confusion%20Matrix%20-%20SVM%20(TF-IDF).png>)

### Additional Visualizations

For interactive visualizations and detailed training output, open `sentiment_analysis.html` in your browser.

---

## Project Structure

```
sentiment-analysis-nlp/
│
├── data/
│   └── imdb_reviews.csv              # IMDb dataset (50,000 labeled reviews)
│
├── images/                           # Visualization charts and confusion matrices
│   ├── Distribution of Sentiment Classes.png
│   ├── Model Performance Comparison.png
│   ├── Confusion Matrix - Naive Bayes (BoW).png
│   ├── Confusion Matrix - Naive Bayes (TF-IDF).png
│   ├── Confusion Matrix - Logistic Regression (BoW).png
│   ├── Confusion Matrix - Logistic Regression (TF-IDF).png
│   ├── Confusion Matrix - SVM (BoW).png
│   └── Confusion Matrix - SVM (TF-IDF).png
│
├── saved_models/                     # Trained models
│   ├── best_model.joblib             # Best model: Logistic Regression (TF-IDF)
│   ├── tfidf_vectorizer.joblib       # TF-IDF vectorizer for feature extraction
│   ├── bow_vectorizer.joblib         # Bag of Words vectorizer
│   ├── naive_bayes_tfidf.joblib      # Trained Naive Bayes model (TF-IDF)
│   └── svm_tfidf.joblib              # Trained SVM model (TF-IDF)
│
├── sentiment_analysis.ipynb          # Main Jupyter notebook with complete pipeline
│
├── sentiment_analysis.html           # HTML export of notebook with visualizations
│
├── requirements.txt                  # Python package dependencies
│
├── README.md                         # Project documentation
│
└── PROJECT_DOCUMENTATION.md          # Detailed technical documentation
```

### File Descriptions

| File                                   | Purpose                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| `sentiment_analysis.ipynb`             | Complete project pipeline: data loading, preprocessing, training, evaluation |
| `sentiment_analysis.html`              | HTML export with visualizations and confusion matrices                       |
| `data/imdb_reviews.csv`                | Input dataset (50,000 labeled movie reviews)                                 |
| `images/`                              | All training visualizations and confusion matrices                           |
| `saved_models/best_model.joblib`       | Pre-trained Logistic Regression model (TF-IDF)                               |
| `saved_models/tfidf_vectorizer.joblib` | TF-IDF feature extractor                                                     |
| `requirements.txt`                     | Python dependencies and versions                                             |
| `README.md`                            | Project overview and setup instructions                                      |
| `PROJECT_DOCUMENTATION.md`             | In-depth technical documentation                                             |

## Instructions to Run the Project

### Prerequisites

- **Python 3.8** or higher
- **pip** (Python package manager)
- **Jupyter Notebook** or **VS Code** with Jupyter extension
- **Disk space**: ~500MB (for dataset and models)

### Complete Setup Guide

#### **Option A: Full Training (From Scratch)**

Follow these steps to download data, train models, and generate results:

**Step 1: Navigate to Project Directory**

```bash
cd sentiment-analysis-nlp
```

**Step 2: Create Python Virtual Environment**

```bash
python -m venv venv
```

**Step 3: Activate Virtual Environment**

**On Windows:**

```bash
.\venv\Scripts\activate
```

**On Linux/Mac:**

```bash
source venv/bin/activate
```

**Step 4: Install Python Dependencies**

```bash
pip install -r requirements.txt
```

This installs:

- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning models)
- nltk (text processing)
- matplotlib & seaborn (visualization)

**Step 5: Download NLTK Language Data**

```python
python -c "
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
print('NLTK data downloaded successfully!')
"
```

**Step 6: Prepare Dataset**

Download the IMDb dataset from: https://ai.stanford.edu/~amaas/data/sentiment/

Extract and place `imdb_reviews.csv` in the `data/` folder with these columns:

- `review`: Text content of the review
- `sentiment`: Label ("positive" or "negative")

**Step 7: Open and Run Notebook**

```bash
jupyter notebook sentiment_analysis.ipynb
```

Or open in **VS Code** → Right-click notebook → "Open with Jupyter"

**Step 8: Execute All Cells in Order**

Run cells sequentially:

1. **Cell 1-2**: Import libraries (3-5 seconds)
2. **Cell 3**: Quick Load check (pass if no saved models exist)
3. **Cell 4-9**: Load and explore dataset (5-10 seconds)
4. **Cell 10-15**: Text preprocessing (30-60 seconds)
5. **Cell 16-18**: Feature extraction (15-30 seconds)
6. **Cell 19-32**: Model training and evaluation (2-5 minutes)
7. **Cell 33-39**: Model comparison and visualization (1-2 minutes)

**Total Runtime: ~5-10 minutes** depending on system

**Step 9: View Results**

- Check console output for accuracy, precision, recall, F1-score
- View confusion matrices and comparison charts
- Models are automatically saved to `saved_models/` folder

---

#### **Option B: Quick Start (Using Pre-trained Models)**

If models are already trained and saved in `saved_models/` folder:

**Step 1: Activate Virtual Environment**

```bash
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

**Step 2: Open Notebook**

```bash
jupyter notebook sentiment_analysis.ipynb
```

**Step 3: Run "Quick Load" Cell**

- Run **Cell 1** (Quick Load - Load Pre-trained Models)
- This loads saved models in ~2 seconds

**Step 4: Test on New Reviews**

- Go to **last cell** (Cell 39)
- Use: `analyze_sentiment("Your review text here")`
- Get instant predictions

**Time Saved: Skip 5-10 minute training, go straight to predictions!**

---

### Troubleshooting

| Issue                                         | Solution                                                       |
| --------------------------------------------- | -------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'nltk'` | Run: `pip install -r requirements.txt`                         |
| `FileNotFoundError: data/imdb_reviews.csv`    | Download dataset from Stanford link and save to `data/` folder |
| `NLTK punkt tokenizer not found`              | Run the NLTK download command from Step 5                      |
| `Jupyter not found`                           | Run: `pip install jupyter`                                     |
| Memory error during training                  | Close other applications or use smaller dataset sample         |

---

## Technology Stack

| Component               | Technology          |
| ----------------------- | ------------------- |
| Programming Language    | Python 3.x          |
| Data Manipulation       | pandas, numpy       |
| Text Processing         | NLTK, regex         |
| Machine Learning        | scikit-learn        |
| Visualization           | matplotlib, seaborn |
| Development Environment | Jupyter Notebook    |

---

## Limitations

1. **Binary Classification Only**: Limited to positive/negative sentiment; neutral sentiment not considered
2. **Domain Specific**: Trained on movie reviews; may not generalize well to other domains
3. **Language**: English language only
4. **Sarcasm/Irony**: May not accurately detect sarcastic or ironic statements
5. **Context**: Does not consider context or sequence of words (bag-of-words approach)

---

## Future Improvements

1. **Multi-class Classification**: Add neutral sentiment class
2. **Word Embeddings**: Implement Word2Vec or GloVe for better feature representation
3. **Cross-Validation**: Use k-fold cross-validation for more robust evaluation
4. **Hyperparameter Tuning**: Use GridSearchCV for optimal model parameters
5. **Ensemble Methods**: Combine multiple models for improved accuracy
6. **Web Application**: Create Flask/Streamlit app for real-time predictions
7. **Domain Adaptation**: Train on multiple domains (product reviews, social media)

---

## References

1. Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.
2. IMDb Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
3. scikit-learn Documentation: https://scikit-learn.org/
4. NLTK Documentation: https://www.nltk.org/

---

## Author

This project was developed as part of an MS in Artificial Intelligence application, demonstrating understanding of NLP concepts, machine learning workflows, and model evaluation techniques.

---

## License

This project is for educational purposes only.
