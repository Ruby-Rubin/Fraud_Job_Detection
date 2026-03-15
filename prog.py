import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# download stopwords first time
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("fake_job_postings.csv")

# Fill missing values
data['description'] = data['description'].fillna("")

# Text cleaning function
def clean_text(text):

    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]

    return " ".join(words)

# Apply preprocessing
data['clean_description'] = data['description'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['clean_description'])

# Target column
y = data['fraudulent']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")