
# Ryanair Customer Query Classifier

This project classifies customer queries into predefined categories (e.g., Flight Changes, Mobile App Issues) using a machine learning model. It includes a deployable FastAPI microservice for real-time and batch inference.

---

## Deliverables

1. A well-documented Jupyter notebook or Python script that includes all the steps from data preprocessing to solution evaluation.
   - Solution.ipynb
2. A detailed report:
   - Report.pdf
3. A README file providing instructions on how to run your code and reproduce the results.
   - README.MD

---

## Repository Structure

```
.
├── data/
│   ├── customer_queries_data.csv
│   ├── customer_queries_test.csv
│   ├── customer_queries_test_with_predictions.csv
│   └── response_customer_queries_test_with_predictions.json
│
├── media/
│   └── [plots]
│
├── inference_api/
│   ├── app.py
│   ├── requirements.txt
│   ├── LR_model_lemmatized_removeRyanair.joblib
│   └── LR_label_encoder_lemmatized_removeRyanair.joblib
│
├── saved_models/
│   └── [older model snapshots]
│
├── Solution.ipynb
├── Report.pdf
└── README.md
```

---

## Project Summary

- **Problem**: Automatically categorize customer support queries into 30 categories.
- **Approach**: TF-IDF + Logistic Regression (with class weighting for imbalance).
- **Preprocessing**:
  - Lemmatization (e.g., converting “airports” → “airport”)
  - Stopword removal
  - Custom stopword: `"ryanair"` excluded due to lack of discriminative value
- **Deployment**: Served via FastAPI with batch and real-time capabilities.

---

## Data

- `customer_queries_data.csv`: Labeled training set (20,000 examples)
- `customer_queries_test.csv`: Test set for inference
- `customer_queries_test_with_predictions.csv`: Example predictions
- `response_customer_queries_test_with_predictions.json`: API output snapshot

---

## Notebook

The Jupyter notebook `Solution.ipynb` includes:

- Exploratory data analysis
- Query length and keyword distribution
- Class imbalance handling
- TF-IDF and Logistic Regression pipeline
- Interpretability analysis (coefficients, top features)
- Confusion matrix and per-class performance
- Comparison of 3 model versions:
  1. Raw model
  2. Lemmatized input
  3. Lemmatized + `"ryanair"` removed (final model)
- Interpretation of probabilities and top features per class



---

## Saved Models

- `LR_model_lemmatized_removeRyanair.joblib`: Final model with lemmatization and custom stopword removal
- `LR_label_encoder_lemmatized_removeRyanair.joblib`: Matching label encoder 
- `LR_model_lemmatized.joblib`: Final model with lemmatization
- `LR_label_encoder_lemmatized.joblib`: Matching label encoder
- `LR_model.joblib`: Final model without preprocessing
- `LR_label_encoder.joblib`: Matching label encoder

---

## How to Run the Inference API

### 1. Install dependencies

```bash
cd inference_api
pip install -r requirements.txt
```

### 2. Launch the API

```bash
uvicorn app:app --reload
```

API will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

Access the API documentation through your browser at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

You can interact with the two available endpoints directly in the Swagger UI:

- `POST /predict` — Send raw text queries and receive top predictions.
- `POST /predict-file` — Upload a `.csv` file with a `query` column for batch predictions.

### How to Use:

1. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Expand either `POST /predict` or `POST /predict-file`
3. Click **"Try it out"**
4. Provide the input (either JSON or CSV upload)
5. Click **"Execute"** to run the request
6. View or download the structured response including predictions and probabilities

---

### `/predict` (POST)

Send a list of customer queries and get back the top-5 predicted categories (with probabilities), lemmatized text, and final predicted label.

#### Request:

```json
{
  "queries": [
    "I need to reset my password",
    "Can I change my flight date?"
  ]
}
```

#### Response:

```json
{
  "results": [
    {
      "query": "I need to reset my password",
      "lemmatized_query": "need reset password",
      "inference": "Customer Account Issues",
      "top_5_predictions": [
        { "label": "Customer Account Issues", "probability": 0.3119 },
        { "label": "Travel Documentation", "probability": 0.1142 },
        ...
      ]
    }
  ]
}
```

---

### `/predict-file` (POST)

Upload a `.csv` with a `query` column and receive the predicted label, lemmatized query, and top-5 class probabilities.

#### Example CSV:

```
query
Why was my credit card declined?
Can I bring a stroller on board?
```

#### Response:

```json
[
  {
    "query": "Why was my credit card declined?",
    "lemmatized_query": "credit card declined",
    "inference": "Payment Issues",
    "label": "...",
    "probability": 0.91,
    ...
  }
]
```

---

## How to Stop the Server

- Press `Ctrl + C` in terminal
- Or manually:
```bash
ps aux | grep uvicorn
kill <PID>
```

---

## Results

- Accuracy: **0.99**
- Macro F1: **0.99**
- Balanced performance across 30 categories
- Interpretability: Top model features analyzed per class
- Real-time predictions are near-instantaneous

---

## Final Notes

Three model versions were tested:

1. **Baseline** (no text normalization)
2. **Lemmatized**: Better generalization by reducing plurals/singulars
3. **Lemmatized + "ryanair" removed**: Removes high-frequency but non-informative token

The third model was chosen for deployment. It offers top interpretability and performance while eliminating non-discriminative features.

**Data is not included in the repo due to confidentiality purposes**
---
