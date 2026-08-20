import joblib
from sklearn.dummy import DummyClassifier
import numpy as np

# This script creates a small placeholder model.pkl for local testing.
# Run: python create_dummy_model.py

clf = DummyClassifier(strategy="most_frequent")
# Minimal training data so the classifier can be saved
X = np.array([[0], [1]])
y = np.array([0, 0])
clf.fit(X, y)
joblib.dump(clf, "model.pkl")
print("Created placeholder model.pkl in the current directory.")
