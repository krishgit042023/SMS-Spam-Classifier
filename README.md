# SMS-Spam-Classifier
Classifying genuine and fake emails using NLP and Naive Bayes Classifier

# Genuine and Fake Email Classification

## Overview

This project demonstrates the use of Natural Language Processing (NLP) and the Naive Bayes Classifier to classify SMS/emails as genuine or fake. The repository contains the complete pipeline for preprocessing, training, testing, and evaluating the model, making it a valuable resource for understanding how to approach email classification problems using machine learning.

---

## Features

- Preprocessing emails with tokenization, stopword removal, and vectorization.
- Implementation of Naive Bayes Classifier for classification.
- Evaluation of the model using metrics like accuracy, precision, recall, and F1-score.
- Clean and modular code for easy customization and understanding.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - NLTK

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/email-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd email-classification
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```
email-classification/
├── data/
│   ├── train.csv       # Training dataset
│   ├── test.csv        # Testing dataset
├── src/
│   ├── preprocess.py   # Email preprocessing scripts
│   ├── model.py        # Model training and evaluation scripts
│   ├── predict.py      # Email classification prediction scripts
├── notebooks/
│   ├── EDA.ipynb       # Exploratory Data Analysis notebook
│   ├── ModelTraining.ipynb  # Model training notebook
├── requirements.txt    # Required Python libraries
├── README.md           # Project readme file
├── LICENSE             # License file
└── .gitignore          # Files to ignore in the repo
```

---

## Usage

### 1. Data Preprocessing

Preprocess the email dataset by tokenizing, removing stopwords, and vectorizing the text data:

```bash
python src/preprocess.py
```

### 2. Model Training

Train the Naive Bayes Classifier using the preprocessed data:

```bash
python src/model.py
```

### 3. Prediction

Classify new emails as genuine or fake:

```bash
python src/predict.py --input email_text_here
```

---

## Dataset

The dataset used in this project consists of email samples labeled as genuine or fake. You can use your own dataset by placing it in the `data/` directory and ensuring it follows the structure:

| Email Text        | Label   |
| ----------------- | ------- |
| "This is a test"  | genuine |
| "You won a prize" | fake    |

---

## Results

The model achieved the following metrics:

- **Accuracy**: 92%
- **Precision**: 91%
- **Recall**: 90%
- **F1-Score**: 90.5%

---

## Future Improvements

- Explore advanced NLP techniques like TF-IDF and word embeddings.
- Implement deep learning models for improved accuracy.
- Enhance preprocessing for handling emojis and non-standard text.

---

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue for any suggestions or bugs.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments

- The Scikit-learn library for the Naive Bayes implementation.
- The NLTK library for text preprocessing.

---

## Contact

For any queries or issues, please contact (halderkrishnendu187@gmail.com).

