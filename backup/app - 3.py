from flask import Flask, jsonify, request, render_template
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib



app = Flask(__name__)


cars = []
with open('car_models.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        cars.append(row)



def create_combined_string(car):
    return f"{car['name']} {car['year']} {car['model']} {car['color']} {car['brand']}"

combined_strings = [create_combined_string(car) for car in cars]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_strings)

car_data = pd.read_csv('car_models.csv')
car_data['combined'] = car_data['name'] + ' ' + car_data['model'] + ' ' + car_data['year'].astype(str) + ' ' + car_data['color'] + ' ' + car_data['brand']



def array_to_string(arr):
    return ' '.join(arr)  

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_cars():
    query = request.args.get('query', '')
    
    results = []
    querio = query.lower().split()
    print(querio)
    matches = []
    
    
    for car in cars:      
        if all(query.lower() in create_combined_string(car) for query in querio):
            matches.append(car)
    print(matches)
    
    if query:
        query_tfidf = tfidf_vectorizer.transform([query])

        cosine_similarities = np.dot(tfidf_matrix, query_tfidf.T).toarray().flatten()
        
        

        for idx, score in enumerate(cosine_similarities):
            if score > 0.2:
                car_info = car_data.iloc[idx].to_dict()
                sanitized_car_info = {key: (value if not pd.isna(value) else None) for key, value in car_info.items()}
                results.append({'car': sanitized_car_info, 'score': score})

       
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        results = results[:10]  
        

    return jsonify(results)

@app.route('/recommend', methods=['GET'])
def recommend_cars():
    query = request.args.get('items')
    
    matches = difflib.get_close_matches(query, combined_strings, n=20, cutoff=0.1)
  

    match_indices = [combined_strings.index(match) for match in matches]
    best_matches = [cars[idx] for idx in match_indices]

    
    return jsonify(best_matches)


if __name__ == '__main__':
    app.run(debug=True)