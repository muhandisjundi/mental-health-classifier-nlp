# Automated Mental Health Classification Engine with Multilingual Pipeline

An end-to-end Natural Language Processing (NLP) pipeline and machine learning framework designed to classify mental health-related narratives and text into 7 distinct psychological categories. This project serves as an automated first-aid screening assistant to flag or categorize potential mental health concerns from user-submitted text.

---

## Project Overview

Mental health filtering systems often struggle with cross-lingual nuances and informal internet slang. This engine implements a robust **Single-Pipe Architecture** that ingests raw multi-dialect text (specifically focusing on Indonesian-English contexts), normalizes colloquial expressions, and utilizes a machine-learning backend to deliver fast and reliable classification.

### Target Classes (7 Categories):
*   `Anxiety`
*   `Bipolar`
*   `Depression`
*   `Normal`
*   `Personality Disorder`
*   `Stress`
*   `Suicidal`

---

## ⚙️ Core Architecture & Methodology

The system is built on a streamlined sequential pipeline:

[Raw Input Text]
│
▼
[Text Preprocessing] ──► (Case Folding & RegEx Noise Removal)
│
▼
[Character Squashing] ──► (Reduces elongated slang e.g., "capeeeeekkkkk" ➔ "capek")
│
▼
[Translation Layer] ──► (Translates Indonesian slang/formal to English via deep-translator)
│
▼
[Feature Extraction] ──► (TF-IDF Vectorization - Max 5,000 Features)
│
▼
[Classification Engine] ──► (Linear Support Vector Classifier - Linear SVC)
│
▼
[Interactive CLI Engine] ──► (Terminal-based loop interface for user input)


1. **Text Preprocessing & Character Squashing:** Cleans input noise, strips special characters, and squashes repeated characters (e.g., transforming "capeeeeekkkkk" into its root word) to preserve context.
2. **Translation Layer:** Employs a multilingual automated translator to standardize textual representations into English before vectorization.
3. **Feature Engineering:** Employs a Term Frequency-Inverse Document Frequency (TF-IDF) configuration capped at 5,000 maximum lexical features.
4. **Classifier:** Powered by a Linear Support Vector Machine (Linear SVC) optimized for high-dimensional text data.

---

## Key Results & Performance Insights
The framework was evaluated using a **Hold-Out Validation Method (80:20 Split)** across a large-scale dataset, evaluating **10,537 test samples**.

* **Global Classification Accuracy:** **78%**
* **Class Peak Performance:** The model achieved an **F1-Score of 0.91** (Recall: 0.95) for the `Normal` category due to data abundance.
* **Lexical Challenges:** The `Depression` category yielded an F1-Score of 0.72. Analytical evaluation showed that this performance threshold is caused by significant *lexical overlaps* (shared vocabulary like "hampa", "lelah", or "help") between the *Depression*, *Stress*, and *Suicidal* classes.

---

## Quick Start & Installation
### Prerequisites
Make sure you have Python installed along with the required libraries:

```bash
pip install scikit-learn pandas numpy deep-translator ipywidgets
