# Text-Classification-using-DistillBERT-Classification

## Overview  
This project focuses on **text classification** using the **DistilBERT** model, a lightweight and efficient version of BERT developed by Hugging Face. The notebook walks through the entire pipeline of natural language processing — from loading and preprocessing a text dataset to fine-tuning the transformer model and evaluating its performance. The aim is to build a robust, high-performing classifier that can generalize well to unseen text data using minimal resources and training time.

---

## Objectives  
- Load and prepare text data for classification tasks  
- Tokenize and encode text using **DistilBERT tokenizer**  
- Fine-tune the **`distilbert-base-uncased`** model on the dataset  
- Evaluate model performance using key NLP metrics  
- Predict outcomes on new, custom text samples  

---

##  Project Workflow  

### 1. Data Preparation  
- Load a labeled text dataset (e.g., binary/multiclass)  
- Encode categorical labels into numerical format  
- Split dataset into **training** and **testing** sets  

### 2. Tokenization  
- Use Hugging Face's **DistilBERT tokenizer**  
- Convert text into token IDs and attention masks  
- Ensure proper padding and truncation  

### 3. Model Implementation - DistilBERT  
- Use `transformers` library to load **`distilbert-base-uncased`**  
- Add a classification head on top (e.g., linear layer)  
- Fine-tune the model using **PyTorch** or **Trainer API**  

### 4. Evaluation  
- Compute metrics including:  
  - **Accuracy**  
  - **Precision & Recall**  
  - **F1-Score**  
  - **Confusion Matrix**  
- Visualize results for deeper insights  

### 5. Inference  
- Pass custom text inputs to the trained model  
- Return predicted labels with confidence scores  

---

## Libraries & Tools  
- `transformers` – Hugging Face model & tokenizer  
-  `sklearn` – Evaluation metrics  
-  `pandas`, `numpy` – Data handling  
-  `torch` – Model training (or optionally, TensorFlow)

---

##  Sample Code Snippet

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("This movie was fantastic!", return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
