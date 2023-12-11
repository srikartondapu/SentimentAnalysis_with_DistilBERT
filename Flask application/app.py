from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from transformers import AutoTokenizer
app = Flask(__name__)

# Loading the fine-tuned BERT model and tokenizer
model_path = '//Users/srikartondapu/Desktop/NCRI_assessment/distilbert_imdb150000_best_f1/best_model_e_3'
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Function to get sentiment prediction
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)[0].tolist()
    sentiment = "Positive" if probabilities[1] > probabilities[0] else "Negative"
    return sentiment, probabilities

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment, probabilities = get_sentiment(user_input)
        return render_template('index.html', user_input=user_input, sentiment=sentiment, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)
