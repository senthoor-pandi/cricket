# app.py
import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open('model/cricket_model.pkl', 'rb'))
team_encoder = pickle.load(open('model/team_encoder.pkl', 'rb'))
venue_encoder = pickle.load(open('model/venue_encoder.pkl', 'rb'))

# Load dataset
df = pd.read_csv('dataset/matches.csv')

# Extract unique teams and venues
teams = list(set(df['team1'].unique().tolist() + df['team2'].unique().tolist()))
venues = df['venue'].dropna().unique().tolist()

# Streamlit UI
st.title("🏏 Cricket Winner Prediction")

st.markdown("### Select Match Details:")

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)
venue = st.selectbox("Select Venue", venues)

if st.button('Predict Winner'):
    if team1 == team2:
        st.error("Team 1 and Team 2 must be different!")
    else:
        # Encode inputs
        try:
            team1_encoded = team_encoder.transform([team1])[0]
            team2_encoded = team_encoder.transform([team2])[0]
            venue_encoded = venue_encoder.transform([venue])[0]

            input_data = [[team1_encoded, team2_encoded, venue_encoded]]

            # Predict
            prediction_encoded = model.predict(input_data)[0]
            winner = team_encoder.inverse_transform([prediction_encoded])[0]

            st.success(f"🏆 Predicted Winner: **{winner}**")

            # Confetti effect using emojis
            st.balloons()

        except Exception as e:
            st.error(f"Encoding error: {str(e)}")
