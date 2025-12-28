import uvicorn
from fastapi import FastAPI
from nltk.corpus import stopwords
import pickle 
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import os

# for UI:-
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, Form

import nltk

nltk.download('stopwords')
nltk.download('wordnet')



app = FastAPI(title="IMDB Sentiment Analysis API")

templates = Jinja2Templates(directory="templates")
# load model & vectorizer

with open("sentiment_analysis.pkl" , "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def home(request: Request ,  sentiment: str = None):
    return templates.TemplateResponse(
        "index.html", {"request": request,"sentiment": sentiment}
    )

@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(request: Request, text: str = Form(...)):
    
    
    lr = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z0-9]', ' ',text)
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [lr.lemmatize(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    

    vector = vectorizer.transform([review])
    prediction = model.predict(vector)[0]

    sentiment = "Positive Review" if prediction == 1 else "Negative review"
        
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sentiment": sentiment
        }
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
