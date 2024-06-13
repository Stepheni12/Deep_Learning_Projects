from flask import Flask, render_template, request, jsonify
import torch
import re
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

model = torch.jit.load('./model.pth')
model.eval()

def preprocess(tweet):
    # Only leave necessary characters in tweets
    res = re.sub(r'[^a-zA-Z0-9@ ]', '', tweet)

    # Convert all tweets to lowercase
    res = res.lower()

    # Convert tweets with more than 2 repeating characters to just 2
    res = re.sub(r'(.)(\1+)', r'\1\1', res)

    # Convert all urls to 'URL'
    res = re.sub(r'http\S+', 'URL', res)

    # Convert all @ user mentions to 'USER'
    res = re.sub(r'@\S+', 'USER', res)

    sent_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    emb = sent_model.encode(res)
    emb = torch.tensor(emb).unsqueeze(0)

    return emb

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    tweet = request.form['tweet']
    emb = preprocess(tweet)
    raw_output = model(emb).item()

    if raw_output > 0.5:
        prediction = "Tweet indicates a disaster"
    else:
        prediction = "Tweet does not indicate a disaster"

    return jsonify({'tweet': tweet, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)