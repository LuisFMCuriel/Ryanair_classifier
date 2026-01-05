from fastapi import FastAPI, File, UploadFile
import pandas as pd
from pydantic import BaseModel
from typing import List
import joblib
from fastapi.responses import JSONResponse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re



# Load model and label encoder
pipeline = joblib.load("LR_model_lemmatized_removeRyanair.joblib")
label_encoder = joblib.load("LR_label_encoder_lemmatized_removeRyanair.joblib")

app = FastAPI()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
"""
def lemmatize_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)
"""
def lemmatize_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    lemmatized = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word != "ryanair"
    ]
    return ' '.join(lemmatized)

class QueryInput(BaseModel):
    queries: List[str]

"""
@app.post("/predict")
def predict(input_data: QueryInput):
    lemmatized_queries = [lemmatize_text(q) for q in input_data.queries]
    predictions_encoded = pipeline.predict(lemmatized_queries)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    return {"predictions": predictions.tolist()}
"""
# Added top-k predictions with probabilities
@app.post("/predict")
def predict(input_data: QueryInput):
    lemmatized_queries = [lemmatize_text(q) for q in input_data.queries]
    probas = pipeline.predict_proba(lemmatized_queries)
    top_k = 5
    class_labels = label_encoder.inverse_transform(pipeline.classes_)

    results = []
    for i, query_proba in enumerate(probas):
        top_indices = query_proba.argsort()[::-1][:top_k]
        top_classes = class_labels[top_indices]
        top_probs = query_proba[top_indices]
        results.append({
            "query": input_data.queries[i],
            "lemmatized_query": lemmatized_queries[i],
            "inference": top_classes[0],  # Most probable class
            "top_5_predictions": [
                {"label": str(top_classes[j]), "probability": float(round(top_probs[j], 4))}
                for j in range(top_k)
            ]
        })

    return {"results": results}




"""
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if 'query' not in df.columns:
        return JSONResponse(status_code=400, content={"error": "CSV must contain a 'query' column."})

    df['lemmatized_query'] = df['query'].apply(lemmatize_text)
    predictions_encoded = pipeline.predict(df['lemmatized_query'])
    predictions = label_encoder.inverse_transform(predictions_encoded)
    df['predicted_label'] = predictions

    return df.to_dict(orient="records")
"""

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if 'query' not in df.columns:
        return JSONResponse(status_code=400, content={"error": "CSV must contain a 'query' column."})

    df['lemmatized_query'] = df['query'].apply(lemmatize_text)
    probas = pipeline.predict_proba(df['lemmatized_query'])
    top_k = 5
    class_labels = label_encoder.inverse_transform(pipeline.classes_)

    top_k_results = []
    for i, query_proba in enumerate(probas):
        top_indices = query_proba.argsort()[::-1][:top_k]
        row_result = {
            "query": df.loc[i, 'query'],
            "lemmatized_query": df.loc[i, 'lemmatized_query'],
            "inference": class_labels[top_indices[0]]  # Top predicted class
        }
        for j in range(top_k):
            row_result[f"top_{j+1}_label"] = class_labels[top_indices[j]]
            row_result[f"top_{j+1}_prob"] = float(round(query_proba[top_indices[j]], 4))
        top_k_results.append(row_result)

    return top_k_results








