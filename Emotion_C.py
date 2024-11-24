#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox

# Load the dataset
file_path = 'C:\\Users\\Bely\\Documents\\Ml_excercise\\emo1.csv'
df = pd.read_csv(file_path)

# Data preprocessing
df['text'] = df['text'].str.lower()
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['text']).toarray()
y = df.drop(columns=['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)

# Calculate accuracy
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)

# GUI Application
def classify_text():
    for widget in table_frame.winfo_children():
        widget.destroy()  # Clear previous results
    
    user_input = entry.get()
    if user_input.strip() == "":
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    
    transformed_input = tfidf.transform([user_input]).toarray()
    predictions = model.predict(transformed_input)
    emotion_labels = y.columns
    
    # Update the table with predictions
    for i, emotion in enumerate(emotion_labels):
        ticked = "✓" if predictions[0][i] == 1 else ""
        tk.Label(table_frame, text=emotion, borderwidth=2, relief="groove", width=20, anchor='w').grid(row=i // 3, column=(i % 3) * 2, padx=5, pady=5)
        tk.Label(table_frame, text=ticked, borderwidth=2, relief="groove", width=5).grid(row=i // 3, column=(i % 3) * 2 + 1, padx=5, pady=5)
    
    # Display accuracy at the bottom
   # accuracy_label.config(text=f"Accuracy: {accuracy:.4f}")

# Create the main application window
root = tk.Tk()
root.title("Emotion Classification")

# Create input field
label = tk.Label(root, text="Enter text:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Create classify button
button = tk.Button(root, text="Classify Emotions", command=classify_text)
button.pack(pady=20)

# Create frame for the table
table_frame = tk.Frame(root)
table_frame.pack(pady=20)

# Create and pack accuracy label
#accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.4f}", borderwidth=2, relief="groove", width=30)
#accuracy_label.pack(pady=10)

# Display all labels initially
emotion_labels = y.columns
for i, emotion in enumerate(emotion_labels):
    tk.Label(table_frame, text=emotion, borderwidth=2, relief="groove", width=20, anchor='w').grid(row=i // 3, column=(i % 3) * 2, padx=5, pady=5)
    tk.Label(table_frame, text="", borderwidth=2, relief="groove", width=5).grid(row=i // 3, column=(i % 3) * 2 + 1, padx=5, pady=5)

# Start the GUI event loop
root.mainloop()





# In[ ]:


import numpy as np
from sklearn.metrics import classification_report

# Convert to numpy arrays and ensure they are of integer type
y_test = np.array(y_test).astype(int)
y_pred = np.array(y_pred).astype(int)

# Generate and print the classification report
report = classification_report(y_test, y_pred, zero_division=0)
print(report)



# In[ ]:




