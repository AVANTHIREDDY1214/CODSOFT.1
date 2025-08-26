# CODSOFT.1
# 🎬 Movie Genre Prediction

This project builds a **machine learning model** to predict the **genre of a movie** based on its **plot summary** using Natural Language Processing (NLP) techniques like **TF-IDF** and classifiers such as **Naive Bayes, Logistic Regression, and SVM**.

---

## 📂 Dataset

The dataset consists of four files:

* `description.txt` → Metadata about the dataset (not used for training)
* `train_data.txt` → Training data (contains `plot` and `genre`)
* `test_data.txt` → Test data (contains only `plot`)
* `test_data_solution.txt` → Ground truth genres for the test set

The `.txt` files are **tab-separated**.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AVANTHIREDDY1214/CODSOFT.1
cd codsoft.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with:

```
pandas
numpy
scikit-learn
```

---

## 🚀 Usage

Run the training and evaluation script:

```bash
python movie_genre_prediction.ipynb
```

The script will:

1. Train models (**Naive Bayes, Logistic Regression, SVM**) on `train_data.txt`.
2. Evaluate models using `test_data.txt` and `test_data_solution.txt`.
3. Print accuracy and classification reports.
4. Predict genres for new unseen plots.

---

## 📝 Example Output

```
----- Logistic Regression -----
Accuracy: 0.82
              precision    recall  f1-score   support

Action            0.85      0.81      0.83       120
Comedy            0.80      0.84      0.82       150
Drama             0.81      0.80      0.81       130
...

Example Prediction: Fantasy
```

---

## 📊 Models Used

* **Naive Bayes** – Fast baseline model for text classification
* **Logistic Regression** – Strong linear classifier for text
* **Linear SVM** – Works well with sparse high-dimensional TF-IDF vectors

---

## 🔮 Future Improvements

* Use **Word Embeddings** (Word2Vec, GloVe)
* Fine-tune **Transformers (BERT, RoBERTa)** for higher accuracy
* Hyperparameter tuning with **GridSearchCV**

