from flask import Flask, jsonify, request, render_template
import csv

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import difflib
from fuzzywuzzy import fuzz

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




def create_combined_string(car):
    return f"{car['name']} {car['year']} {car['model']} {car['color']} {car['brand']}"

combined_strings = [create_combined_string(car) for car in cars]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_strings)

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
        
        results = []
        for car in cars:
            name_score = fuzz.partial_ratio(query.lower(), car['name'].lower()) * 2  # Double weight for name
            model_score = fuzz.partial_ratio(query.lower(), car['model'].lower())
            year_score = fuzz.partial_ratio(query.lower(), car['year'].lower())
            color_score = fuzz.partial_ratio(query.lower(), car['color'].lower())
            brand_score = fuzz.partial_ratio(query.lower(), car['brand'].lower())
            
            total_score = (name_score + model_score + year_score + color_score + brand_score) / 6  # Adjusted for name's double weight
            
            results.append((car, total_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        results= results[:5] 

        return jsonify([{
        'car': {
            'name': car['name'],
            'model': car['model'],
            'year': car['year'],
            'color': car['color'],
            'brand': car['brand'],
            'acceleration': car['acceleration'],
            'cylinders': car['cylinders'],
            'displacement': car['displacement'],
            'horsepower': car['horsepower'],
            'mpg': car['mpg'],
            'origin': car['origin'],
            'weight':car['weight']
            
        },
        'score': score
    } for car, score in results])
       

    
    except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return jsonify({"error": "An internal error occurred"}), 500

@app.route('/recommend', methods=['GET'])
def recommend_cars():
    query = request.args.get('items')
    
    matches = difflib.get_close_matches(query, combined_strings, n=20, cutoff=0.1)
  

    match_indices = [combined_strings.index(match) for match in matches]
    best_matches = [cars[idx] for idx in match_indices]

    
    return jsonify(best_matches)


if __name__ == '__main__':
    app.run(debug=True)