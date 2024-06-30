from flask import Flask, request, jsonify, send_from_directory
import pdfplumber
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration
import re
import torch
import os

app = Flask(__name__, static_folder='static')

# Function to read PDF and extract text using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to read Word document and extract text
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to preprocess text (cleaning)
def preprocess_text(text):
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text)
    text = re.sub(r'\b(?:doi|arxiv|fig|table|hal|ids|isbn|issn|pp)\b.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:submitted|accepted|published|conference|journal|doi)\b.*', '', text, flags=re.IGNORECASE)
    return text

# Function to estimate the number of tokens for a given word count
def estimate_token_length(word_count):
    return int(word_count * 1.3)

# Function to generate summary using BART model
def generate_summary_bart(text, word_length=150):
    max_length = 1024
    token_length = min(estimate_token_length(word_length), max_length)
    min_length = max(50, int(token_length / 2))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)

    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=token_length, min_length=min_length, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Endpoint to summarize a paper
@app.route('/summarize', methods=['POST'])
def summarize_paper():
    file = request.files['file']
    word_length = int(request.form['word_length'])

    if file.filename.endswith('.pdf'):
        paper_text = extract_text_from_pdf(file)
    elif file.filename.endswith('.docx'):
        paper_text = extract_text_from_docx(file)
    else:
        return jsonify({"error": "Unsupported file format. Please provide a PDF or DOCX file."}), 400

    clean_text = preprocess_text(paper_text)
    summary = generate_summary_bart(clean_text, word_length)
    
    return jsonify({"summary": summary})

@app.route('/', methods=['GET'])
def serve_homepage():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static_file(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
