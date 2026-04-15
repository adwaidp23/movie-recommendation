# Movie Recommendation System

A content-based movie recommendation system using Netflix data. Built with Flask (API) and Streamlit (web app).

## Features

- Recommends movies based on title, cast, genre, and description similarity
- Uses TF-IDF vectorization and cosine similarity
- Supports fuzzy matching for movie titles
- Available as a REST API (Flask) or interactive web app (Streamlit)

## Dataset

Uses the Netflix dataset (`netflix_data.csv`) with movie/TV show information.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adwaidp23/movie-recommendation.git
   cd movie-recommendation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Flask API

Run the Flask app:
```bash
python app.py
```

The API will be available at `http://localhost:5000`.

- GET `/`: Serves the index.html template
- POST `/recommend`: Accepts JSON `{"query": "movie title"}` and returns recommendations

Example request:
```bash
curl -X POST http://localhost:5000/recommend -H "Content-Type: application/json" -d '{"query": "Inception"}'
```

### Streamlit App

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Open the provided URL in your browser and enter a movie title to get recommendations.

## Deployment

### Flask
Use Gunicorn for production:
```bash
gunicorn -w 4 app:app
```

### Streamlit
Deploy to [Streamlit Cloud](https://share.streamlit.io) by connecting this GitHub repository.

## Security

- Set `SECRET_KEY` environment variable for Flask sessions
- Ensure the dataset path is correct
- For production, disable debug mode

## License

[Add license if applicable]