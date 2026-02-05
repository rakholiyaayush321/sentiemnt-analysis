import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC  # Faster than SVC
from statistics import mode
from nltk.classify import ClassifierI
import mlflow
import mlflow.sklearn

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', text.lower())
    tokenized = word_tokenize(cleaned)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokenized if word not in stop_words and word.isalpha()]
    return ' '.join(lemmatized)

# Load data (assuming train/pos, train/neg, test/pos, test/neg folders)
def load_data():
    documents = []
    for category in ['pos', 'neg']:
        for folder in ['train', 'test']:
            path = f'{folder}/{category}'
            if os.path.exists(path):
                for file in os.listdir(path):
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                        review = f.read()
                        documents.append((preprocess(review), category))
    return documents

documents = load_data()
random.shuffle(documents)

texts = [doc[0] for doc in documents]
labels = [1 if doc[1] == 'pos' else 0 for doc in documents]  # 1: pos, 0: neg

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Define VoteClassifier
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = [c.classify(features) for c in self._classifiers]
        return mode(votes)

    def confidence(self, features):
        votes = [c.classify(features) for c in self._classifiers]
        choice_votes = votes.count(mode(votes))
        return choice_votes / len(votes)

# Train individual classifiers with SklearnClassifier wrapper
def train_classifier(model):
    return SklearnClassifier(model).train([(X_train[i].toarray()[0], y_train[i]) for i in range(X_train.shape[0])])

MNB_clf = train_classifier(MultinomialNB())
BNB_clf = train_classifier(BernoulliNB())
LogReg_clf = train_classifier(LogisticRegression(max_iter=200))
SGD_clf = train_classifier(SGDClassifier())
SVC_clf = train_classifier(LinearSVC(max_iter=2000))

voted_classifier = VoteClassifier(MNB_clf, BNB_clf, LogReg_clf, SGD_clf, SVC_clf)

# Evaluate
def classify(features):
    return voted_classifier.classify(features.toarray()[0])

predictions = [classify(X_test[i]) for i in range(X_test.shape[0])]
new_f1 = f1_score(y_test, predictions)
print(f"Improved Model F1-Score: {new_f1}")

# Save improved models and vectorizer
with open('pickled_algos/improved_vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
# Save each classifier similarly...
# (Add code to pickle each: pickle.dump(MNB_clf, open('pickled_algos/improved_MNB_clf.pickle', 'wb')) etc.)
