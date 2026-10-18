# NextFlix 🎬

A movie discovery app with content-based and collaborative recommendations, built with Python, Streamlit, pandas, and scikit-learn.

[Launch the live demo](https://nextflix-4ibeoqz4yz6y4mqzing3da.streamlit.app/)

## Features

- Search the bundled MovieLens catalog and filter by exact genre.
- Find similar movies using cosine similarity over TF-IDF genre vectors.
- Explore recommendations for anonymous sample viewers using user-based collaborative filtering. Movies already rated by the selected viewer are excluded.
- Browse weekly trending movies, posters, details, and YouTube trailers through TMDB.
- Chat with the Groq-powered movie assistant, ask follow-up questions, and open movie cards matched through TMDB. Movie search remains available when Groq is unavailable.
- Continue browsing and requesting local title/genre recommendations when TMDB is unavailable.

## Run locally

Use Python 3.12 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For posters, trending, actor search, and trailers, create `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_tmdb_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

Alternatively set the `TMDB_API_KEY` and `GROQ_API_KEY` environment variables. Never commit credentials. Without a key, local catalog features remain available with placeholder artwork.

## Deployment

The Streamlit Community Cloud app should point to `krishnaverma09/NextFlix`, branch `main`, entrypoint `app.py`, with Python 3.12 or newer. Set `TMDB_API_KEY` and `GROQ_API_KEY` in the app's secrets settings. Pushes to the connected branch trigger an update; sleeping apps may require a wake-up from their public page.

The runtime reads the bundled CSV files and builds cached sparse models. It does not load the legacy pickle models or require downloading large model files. API results are cached for 15 minutes, and requests have bounded timeouts.

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests cover recommendation correctness, excluding watched movies, genre filtering, detail/back navigation, assistant queries, similar-viewer recommendations, and API outages. GitHub Actions runs these checks on pushes and pull requests.

## Data and limitations

The bundled dataset is the [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/) from GroupLens; see its usage terms. Collaborative recommendations use anonymous sample viewers, not a logged-in user's watch history. Content similarity uses genres, so it can favor movies with identical genre combinations. The local catalog is historical; weekly trending comes from TMDB.

Movie metadata and artwork are provided by [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB. Trailer availability depends on TMDB and YouTube.

The AI assistant uses Groq’s `openai/gpt-oss-120b` model. It sends the latest query and up to eight recent messages to Groq, keeps conversation history in the current session, and offers a Clear conversation button. AI recommendations can be inaccurate; movie cards are resolved against TMDB or the local catalog.
