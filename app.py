from flask import Flask, jsonify, request, render_template
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_cars():
    query = request.args.get('query', '')

    if query == "":
        return 
    
    querio = query.lower().split()
    matches = []
     
    for car in cars:      
        if all(query.lower() in create_combined_string(car) for query in querio): 
            matches.append(car)
    return jsonify(matches)


@app.route('/recommend', methods=['GET'])
def recommend_cars():
    query = request.args.get('items')
      
    query = query.lower().split(",")
    print(query)

    results = []
    if query:
    
        similarities = []
        for word in query:
            query_vector = tfidf_vectorizer.transform([word]).toarray()[0]
            similarity = cosine_similarity(tfidf_matrix.toarray(), [query_vector])
            similarities.append(similarity.flatten())
        print(similarities)

        # Combine the similarities with more weight for repeated words
        combined_similarities = np.sum(similarities, axis=0)

        # Normalize the combined similarities
        combined_similarities = combined_similarities / len(query)
      

        top_results = np.argsort(combined_similarities)[::-1][:20]
        

        for idx in top_results:
            car_info = car_data.iloc[idx].to_dict()
            # This function when csv file has
            # sanitized_car_info = {key: (value if not pd.isna(value) else None) for key, value in car_info.items()}
            results.append({'car': car_info, 'score': float(combined_similarities[idx])})

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        results = results[:20]

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)