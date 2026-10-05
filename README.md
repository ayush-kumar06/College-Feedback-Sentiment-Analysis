# 🎓 College Feedback Sentiment Analysis

A Machine Learning and NLP-based project that analyzes college student feedback and classifies it as **Positive** or **Negative**. The project uses **TF-IDF** for text feature extraction and **Multinomial Naive Bayes** for sentiment classification. An interactive **Streamlit dashboard** is provided to visualize and explore student feedback insights.

---

## 📌 Project Overview

Student feedback is an important source of information for understanding the quality of teaching, infrastructure, facilities, administration, and other college services.

This project analyzes textual student feedback and automatically determines its sentiment. The results are presented through an interactive dashboard that makes it easy to understand overall student opinions and identify areas that may require improvement.

---

## 🎯 Objectives

* Clean and preprocess student feedback
* Convert text into numerical features using TF-IDF
* Train a sentiment classification model
* Identify positive and negative feedback
* Analyze feedback category-wise
* Visualize sentiment insights
* Provide real-time sentiment prediction
* Evaluate machine learning model performance

---

## 🚀 Features

* 📝 Student feedback preprocessing
* 🔤 TF-IDF feature extraction
* 🤖 Machine Learning sentiment classification
* 😊 Positive sentiment detection
* ⚠️ Negative sentiment detection
* 📊 Sentiment distribution visualization
* 📈 Category-wise feedback analysis
* 🔍 Real-time feedback sentiment prediction
* 🎯 Accuracy, Precision, Recall and F1 Score
* 📋 Searchable and filterable feedback dataset
* 🎨 Interactive Streamlit dashboard

---

## 🛠️ Technologies Used

| Technology              | Purpose                  |
| ----------------------- | ------------------------ |
| Python                  | Programming Language     |
| Pandas                  | Data Processing          |
| NumPy                   | Numerical Operations     |
| Scikit-learn            | Machine Learning         |
| NLP                     | Text Processing          |
| TF-IDF                  | Text Feature Extraction  |
| Multinomial Naive Bayes | Sentiment Classification |
| Streamlit               | Interactive Dashboard    |

---

## 🔄 Project Workflow

```text
Student Feedback
       ↓
Text Cleaning & Preprocessing
       ↓
TF-IDF Feature Extraction
       ↓
Train / Test Split
       ↓
Multinomial Naive Bayes
       ↓
Sentiment Prediction
       ↓
Dashboard & Visualization
```

---

## 📊 Dataset

The project uses a structured dataset containing **300 college student feedback records**.

### Dataset Columns

* `feedback_id`
* `category`
* `feedback`
* `sentiment`

### Sentiment Classes

* 🟢 Positive
* 🔴 Negative

### Feedback Categories

The dataset contains feedback related to areas such as:

* Faculty
* Infrastructure
* Library
* Laboratory
* Hostel
* Canteen
* WiFi
* Administration
* Placement
* Campus

---

## 🤖 Machine Learning Model

The project uses **Multinomial Naive Bayes** for sentiment classification.

### Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert textual feedback into numerical features that can be processed by the machine learning model.

### Train-Test Split

* **Training Data:** 80%
* **Testing Data:** 20%

---

## 📈 Model Evaluation

The model is evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The trained model achieved **100% accuracy on the current test split** of the project dataset.

> Note: This result is based on the current 300-record dataset and test split. Performance on a larger or real-world dataset may differ.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard with multiple sections.

### 🏠 Dashboard

The dashboard displays:

* Total feedback
* Positive feedback
* Negative feedback
* Model accuracy
* Sentiment distribution
* Category-wise feedback
* Recent student feedback

### 🔍 Analyze Feedback

Users can enter new student feedback and get:

* Predicted sentiment
* Prediction confidence
* Original feedback
* Cleaned feedback

### 📋 Student Feedback

Users can:

* View feedback records
* Search feedback
* Filter feedback by sentiment
* Explore feedback categories

### 📊 Sentiment Analytics

Provides:

* Positive feedback percentage
* Negative feedback percentage
* Sentiment distribution
* Category-wise sentiment analysis
* Category summary

### 🤖 Model Performance

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Model configuration

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/college-feedback-sentiment-analysis.git
```

### 2. Open the Project Folder

```bash
cd college-feedback-sentiment-analysis
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
College-Feedback-Sentiment-Analysis/
│
├── app.py
├── college_feedback_sentiment_dataset_300.csv
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

The main dependencies used in this project are:

```text
streamlit
pandas
numpy
scikit-learn
```

---

## 🔮 Future Scope

The project can be further improved by:

* 🌐 Adding multilingual sentiment analysis
* 📚 Using larger real-world datasets
* 🧠 Implementing advanced deep learning models
* 🔎 Adding aspect-based sentiment analysis
* 📝 Adding automated feedback collection
* ☁️ Deploying the application online
* 🔐 Adding authentication for college administrators
* 📊 Adding advanced interactive visualizations
* 📈 Tracking sentiment trends over time

---

## 🎓 Project Objective

The main objective of this project is to demonstrate how **Natural Language Processing and Machine Learning** can be used to analyze student opinions and generate meaningful insights from textual college feedback.

The system can help educational institutions better understand student satisfaction and identify areas that may need improvement.

---

## 👨‍💻 Author

### Ayush Kumar

This project was developed as a **Minor Project** focused on Natural Language Processing, Machine Learning, and Sentiment Analysis.

### 📬 Connect With Me

* **GitHub:** https://github.com/ayush-kumar06
* **LinkedIn:** https://linkedin.com/in/Ayush Kumar

---

## ⭐ If You Like This Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub!
