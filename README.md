# NameEntityRemover
Idenity and remove personal identifier from Medical Notes 


## Description
This project demonstrates how to fine-tune a pre-trained de-identification model (`obi/deid_roberta_i2b2`) to detect Protected Health Information (PII) in clinical text. The model was trained on **5,000 synthetic clinical notes** generated using the `Faker` library.

The system identifies the following entities:
- **NAME** (Patient/Doctor Names)
- **DOB** (Date of Birth)
- **SSN** (Social Security Number)
- **LOCATION** (Addresses, Cities)
- **CONTACT** (Phone Numbers)
- **EMAIL** (Email Addresses)
- **TRACKER** (Digital identifiers like IP addresses)

## Installation
Install the required dependencies:
```bash
pip install transformers datasets faker seqeval accelerate scikit-learn
```

## Usage
To use the fine-tuned model for inference, load it from the saved directory:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load Model and Tokenizer
model_path = "./fine_tuned_deid_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Initialize Pipeline
nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Run Inference
text = "Patient John Doe (DOB: 1980-01-01) lives at 123 Main St."
results = nlp(text)
print(results)
```

## Results
The fine-tuned model achieved exceptional performance on the test set, with **Precision, Recall, and F1-scores > 0.99** across all entity types.

## Disclaimer
All data used in this project is **synthetic** and generated for educational and demonstration purposes only. No real patient data was used.
"""
