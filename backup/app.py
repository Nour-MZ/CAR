from flask import Flask, jsonify, request, render_template
import csv

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from fuzzywuzzy import process
# import google.generativeai as genai


# API_KEY="AIzaSyBLFXyxwRNYR8EcJZi7jbpXRJfdDw68Tc0"
# genai.configure(api_key=API_KEY)

# model = genai.GenerativeModel('gemini-1.5-flash')


app = Flask(__name__)


cars = []
with open('car_models.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        cars.append(row)

df = pd.DataFrame(cars)
df['combined'] = df['name'] + ' ' + df['year'].astype(str)

# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf.fit_transform(df['combined'])
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def create_combined_string(car):
    return f"{car['name']} {car['year']} {car['model']} {car['color']} {car['brand']}"

combined_strings = [create_combined_string(car) for car in cars]



# def chunk_message(message, max_length):
#     return [message[i:i + max_length] for i in range(0, len(message), max_length)]

# max_chunk_length = 100 


# chunks = chunk_message(combined_strings, max_chunk_length)

def array_to_string(arr):
    return ' '.join(arr)  # Join elements with a space

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_cars():
    try:
        query = request.args.get('query')
        
        
        matches = difflib.get_close_matches(query, combined_strings, n=20, cutoff=0.2)
    

        match_indices = [combined_strings.index(match) for match in matches]


        best_matches = [cars[idx] for idx in match_indices]
        print(best_matches)

        return jsonify(best_matches)
    
    except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return jsonify({"error": "An internal error occurred"}), 500

@app.route('/recommend', methods=['GET'])
def recommend_cars():
    query = request.args.get('items')
    
    matches = difflib.get_close_matches(query, combined_strings, n=20, cutoff=0.1)
  

    match_indices = [combined_strings.index(match) for match in matches]
    best_matches = [cars[idx] for idx in match_indices]

    print(best_matches)
    return jsonify(best_matches)


if __name__ == '__main__':
    app.run(debug=True)