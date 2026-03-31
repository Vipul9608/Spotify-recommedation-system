import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("spotify (1).xls")  # your file
    return df

df = load_data()

st.title("🎵 Spotify Recommendation System")

# -------------------------------
# Show dataset
# -------------------------------
st.write("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Feature Selection
# -------------------------------
# IMPORTANT: Update column names based on your dataset
features = ['danceability', 'energy', 'loudness', 'tempo', 'valence']

# Drop missing values
df = df.dropna(subset=features)

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# -------------------------------
# Similarity Matrix
# -------------------------------
similarity = cosine_similarity(scaled_features)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(song_name):
    if song_name not in df['track_name'].values:
        return ["Song not found"]

    index = df[df['track_name'] == song_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in scores[1:6]:
        recommended.append(df.iloc[i[0]]['track_name'])

    return recommended

# -------------------------------
# UI Selection
# -------------------------------
song_list = df['track_name'].values
selected_song = st.selectbox("Select a song", song_list)

# -------------------------------
# Button
# -------------------------------
if st.button("Recommend"):
    recommendations = recommend(selected_song)

    st.subheader("🎧 Recommended Songs:")
    for song in recommendations:
        st.write("👉", song)
