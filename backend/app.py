from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = pickle.load(open('model/cricket_model.pkl', 'rb'))
team_encoder = pickle.load(open('model/team_encoder.pkl', 'rb'))
venue_encoder = pickle.load(open('model/venue_encoder.pkl', 'rb'))

# Load dataset
df = pd.read_csv('dataset/matches.csv')  

# Extract unique teams and venues
teams = list(set(df['team1'].unique().tolist() + df['team2'].unique().tolist()))
venues = df['venue'].dropna().unique().tolist()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    team1 = data.get('team1')
    team2 = data.get('team2')
    venue = data.get('venue')

    if not team1 or not team2 or not venue:
        return jsonify({'error': 'Missing input fields.'}), 400

    try:
        team1_encoded = team_encoder.transform([team1])[0]
        team2_encoded = team_encoder.transform([team2])[0]
        venue_encoded = venue_encoder.transform([venue])[0]
    except Exception as e:
        return jsonify({'error': f'Encoding error: {str(e)}'}), 400

    input_data = [[team1_encoded, team2_encoded, venue_encoded]]

    # Predict
    prediction_encoded = model.predict(input_data)[0]
    winner = team_encoder.inverse_transform([prediction_encoded])[0]

    return jsonify({
        'winner': winner
    })

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'teams': teams,
        'venues': venues
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
