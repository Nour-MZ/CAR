from flask import Flask, jsonify, request, render_template
import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = Flask(__name__)


cars = []
with open('car_models.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        cars.append(row)

df = pd.DataFrame(cars)
df['combined'] = df['name'] + ' ' + df['year'].astype(str)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def create_combined_string(car):
    return f"{car['name']} {car['year']} {car['model']} {car['color']} {car['brand']}"

combined_strings = [create_combined_string(car) for car in cars]
print(combined_strings)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_cars():
    query = request.args.get('query')
    
    matches = difflib.get_close_matches(query, combined_strings, n=20, cutoff=0.2)

    # Get the best match indices
    match_indices = [combined_strings.index(match) for match in matches]

    # Retrieve the original car dictionaries for the matches
    best_matches = [cars[idx] for idx in match_indices]
    print(best_matches)
    return jsonify(best_matches)

@app.route('/recommend', methods=['GET'])
def recommend_cars():
 
    items = request.args.get('items').split(',')
    recommendations = get_recommendations(items)
   
    return jsonify(recommendations.to_dict('records'))

def get_recommendations(items, cosine_sim=cosine_sim, topn=2):
    recommendations = []
    for item in items:
        matches = df[df['combined'].str.lower().str.contains(item.lower())]
        
        if not matches.empty:
            idx = matches.index[0]
        else:
            print(f"No match found for '{item}'. Skipping.")
            continue

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:topn+1]
        if not sim_scores:
            topn=4
        else:
            topn=2
        item_indices = [i[0] for i in sim_scores]
        rec = df.iloc[item_indices]
        recommendations.append(rec)

    if recommendations:
        recommendations = pd.concat(recommendations, ignore_index=True)
        return recommendations.drop_duplicates()
    else:
        return pd.DataFrame()

if __name__ == '__main__':
    app.run(debug=True)