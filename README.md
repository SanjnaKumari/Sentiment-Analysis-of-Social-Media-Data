# 📊 Sentiment Analysis of Social Media Tweets during the Stock Market Crash (2022)

This project implements sentiment analysis on Twitter data using various text vectorization and embedding techniques. Tweets collected under the hashtag **#stockmarketcrash** are classified into three sentiment categories: **Positive**, **Neutral**, and **Negative**. The project compares multiple embedding strategies to determine which is most effective for sentiment classification using a **Support Vector Machine (SVM)** classifier.

---

## 🧠 Objective

To analyze and classify tweet sentiments using multiple embedding techniques, and identify which representation leads to the highest accuracy using a consistent classification model (SVM).

---

## 🗃 Dataset

The dataset, `dataset.csv`, contains 33,946 tweets, distributed as:

- **Positive**: 12,542 tweets  
- **Neutral**: 11,498 tweets  
- **Negative**: 9,906 tweets  

Collected using the hashtag `#stockmarketcrash` from Twitter.

---

## ⚙️ Methods and Techniques

### 1. Preprocessing
Each tweet is cleaned and normalized using the following steps:
- Lowercasing
- Removing mentions, hashtags, URLs, special characters, and numbers
- Lemmatization using WordNetLemmatizer and POS tagging

### 2. Embedding Techniques

| Embedding Method | Description |
|------------------|-------------|
| **Bag of Words (BoW)** | Counts word frequency. Top 100 features selected via chi-square. |
| **TF-IDF** | Weighs rare but informative words more heavily. Top 100 features selected via chi-square. |
| **Word2Vec** | Uses pre-trained embeddings (`word2vec-google-news-300`) for word-level vectorization. |
| **GloVe** | Uses `glove.6B.300d` embeddings via `torchtext`. |
| **Universal Sentence Encoder (USE)** | Optional advanced technique using Google's pre-trained model for sentence-level embeddings. |

### 3. Classification Model
- **Support Vector Machine (SVM)** with a linear kernel used across all embedding methods
- Data split: 80% training, 20% testing
- Evaluation metric: **Accuracy**

---

## 📈 Results

| Embedding Method         | Accuracy |
|--------------------------|----------|
| Bag of Words (BoW)       | 71.50%   |
| TF-IDF                   | **72.06%** ✅ Best performer |
| Word2Vec (Google News)   | 65.00%   |
| GloVe (Torchtext 6B.300d)| 66.50%   |
| Universal Sentence Encoder (USE) | 68.00% |

TF-IDF outperformed all other methods due to its ability to highlight important yet rare words in tweets.

---

## 🛠 Requirements

- Python 3.x
- Libraries:
  - pandas, numpy, nltk, scikit-learn
  - gensim, torch, torchtext
  - tensorflow, tensorflow-hub (for USE)
  - matplotlib (optional for visualization)

> Run `pip install -r requirements.txt` if a requirements file is added.

---

## 🧩 Key Takeaways

- **TF-IDF** proved to be the best performing embedding method on this dataset.
- **USE** shows promise, but might require more sophisticated models to outperform TF-IDF.
- Word embeddings like **Word2Vec** and **GloVe** require more context-aware models to be fully effective in sentence-level sentiment tasks.

---

## 📜 License

This project is for educational and academic use.

---

## 🙌 Acknowledgments

- Virginia Tech – CS 5804 Assignment  
- Twitter for real-time data  
- Google, Gensim, and Torchtext for pre-trained embeddings


