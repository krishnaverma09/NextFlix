import streamlit as st
from movie_assistant import groq_response
import os
from pathlib import Path
from recommender import (load_catalog, content_model, content_recommendations,
                         collaborative_model, collaborative_recommendations)
import pandas as pd
import requests
import re

# -----------------------------
# Load models
# -----------------------------
st.set_page_config(page_title="NextFlix · Movie discovery", page_icon="🎬", layout="wide")

@st.cache_resource
def load_models():
    catalog = load_catalog()
    return catalog, content_model(catalog)

@st.cache_resource
def load_collaborative():
    return collaborative_model()

movies, content_matrix = load_models()

def api_key(name="TMDB_API_KEY"):
    key = os.environ.get(name, "")
    if not key:
        try:
            key = st.secrets.get(name, "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass
    return key

TMDB_KEY = api_key()
GROQ_KEY = api_key("GROQ_API_KEY")
PLACEHOLDER = str(Path(__file__).resolve().parent / "assets" / "poster.svg")

GENRE_KEYWORDS = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "mystery": 9648,
    "romance": 10749,
    "science fiction": 878,
    "sci fi": 878,
    "thriller": 53,
    "war": 10752,
}

MOOD_KEYWORDS = {
    "feel good": 35,
    "happy": 35,
    "fun": 35,
    "emotional": 18,
    "sad": 18,
    "romantic": 10749,
    "scary": 27,
    "tense": 53,
    "mind bending": 878,
}


@st.cache_data(ttl=900, max_entries=1024, show_spinner=False)
def tmdb_get(path, params=None):

    if not TMDB_KEY:
        return {}

    query_params = params.copy() if params else {}
    query_params["api_key"] = TMDB_KEY
    query_params.setdefault("language", "en-US")

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/{path}",
            params=query_params,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return {}


def poster_from_path(poster_path):

    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path

    return None

@st.cache_data(ttl=900, max_entries=1024)
# -----------------------------
# Poster
# -----------------------------
def fetch_poster(tmdb):

    if pd.isna(tmdb):
        return PLACEHOLDER
    data = tmdb_get(f"movie/{int(tmdb)}")
    poster = poster_from_path(data.get("poster_path"))

    if poster:
        return poster

    return PLACEHOLDER


# -----------------------------
# Movie details
# -----------------------------
def movie_details(tmdb):

    data = tmdb_get(f"movie/{int(tmdb)}") if not pd.isna(tmdb) else {}

    rating=data.get("vote_average","N/A")
    release=data.get("release_date","")
    overview=data.get("overview","")

    return rating,release,overview


# -----------------------------
# Trailer
# -----------------------------
def trailer(tmdb):

    data = tmdb_get(f"movie/{int(tmdb)}/videos") if not pd.isna(tmdb) else {}

    for v in data.get("results",[]):

        if v.get("type")=="Trailer" and v.get("site")=="YouTube" and v.get("key"):

            return f"https://www.youtube.com/watch?v={v['key']}"

    return None


# -----------------------------
# Trending
# -----------------------------
def trending():

    data = tmdb_get("trending/movie/week")

    movies_list=[]

    for m in data.get("results",[])[:6]:

        poster = poster_from_path(m.get("poster_path"))

        if not poster:
            continue

        movies_list.append({
            "title":m.get("title","Unknown"),
            "poster":poster,
            "tmdb":m.get("id")
        })

    return movies_list


# -----------------------------
# Recommendation
# -----------------------------
def recommend(movie):

    return catalog_cards(content_recommendations(movies, content_matrix, movie))


def catalog_cards(rows):
    return [{"title": row["title"], "tmdb": row["tmdbId"],
             "poster": fetch_poster(row["tmdbId"])} for _, row in rows.iterrows()]


# -----------------------------
# Movie assistant
# -----------------------------
def normalize_title(value):

    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def extract_actor_name(query):

    patterns = [
        r"suggest\s+me\s+some\s+([a-zA-Z .'-]+?)\s+(?:movies|films)",
        r"(?:movies|films)\s+(?:of|by|with|starring|featuring)\s+([a-zA-Z .'-]+)",
        r"(?:starring|featuring|with)\s+([a-zA-Z .'-]+)",
    ]

    for pattern in patterns:

        match = re.search(pattern, query, flags=re.IGNORECASE)

        if not match:
            continue

        candidate = re.sub(r"\b(?:movies|films)\b", "", match.group(1), flags=re.IGNORECASE)
        candidate = candidate.strip(" .")

        if not candidate:
            continue

        norm_candidate = normalize_title(candidate)

        if norm_candidate in GENRE_KEYWORDS or norm_candidate in MOOD_KEYWORDS:
            continue

        return candidate

    return None


def detect_genre_id(query):

    lowered = normalize_title(query)

    for keyword, genre_id in sorted(GENRE_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):

        if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
            return genre_id

    for keyword, genre_id in sorted(MOOD_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):

        if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
            return genre_id

    return None


def normalize_tmdb_movies(raw_movies, limit=8):

    results = []
    seen_ids = set()

    for item in raw_movies:

        tmdb_id = item.get("id")
        title = item.get("title") or item.get("name")

        if not tmdb_id or not title or tmdb_id in seen_ids:
            continue

        results.append({
            "title": title,
            "tmdb": tmdb_id,
            "poster": poster_from_path(item.get("poster_path"))
        })

        seen_ids.add(tmdb_id)

        if len(results) == limit:
            break

    return results


def fetch_movies_for_actor(actor_name, limit=8):

    people = tmdb_get("search/person", {
        "query": actor_name,
        "include_adult": "false"
    }).get("results", [])

    if not people:
        return []

    person_id = people[0].get("id")

    if not person_id:
        return []

    credits = tmdb_get(f"person/{person_id}/movie_credits")
    cast = sorted(
        credits.get("cast", []),
        key=lambda x: (x.get("popularity", 0), x.get("vote_count", 0)),
        reverse=True
    )

    return normalize_tmdb_movies(cast, limit)


def fetch_movies_for_genre(genre_id, limit=8):

    data = tmdb_get("discover/movie", {
        "with_genres": genre_id,
        "sort_by": "popularity.desc",
        "vote_count.gte": 80,
        "include_adult": "false",
        "page": 1
    })

    return normalize_tmdb_movies(data.get("results", []), limit)


def fetch_movies_for_query(query, limit=8):

    data = tmdb_get("search/movie", {
        "query": query,
        "include_adult": "false",
        "page": 1
    })

    return normalize_tmdb_movies(data.get("results", []), limit)


def assistant_movie_recommendations(query, limit=5):

    actor_name = extract_actor_name(query)

    if actor_name:

        actor_movies = fetch_movies_for_actor(actor_name, limit + 3)

        if actor_movies:
            return actor_movies[:limit], f"Here are some popular {actor_name} movies:"

    genre_id = detect_genre_id(query)

    if genre_id:

        genre_movies = fetch_movies_for_genre(genre_id, limit + 3)

        if genre_movies:
            return genre_movies[:limit], "Here are some movie recommendations:"

    searched_movies = fetch_movies_for_query(query, limit + 3)

    if searched_movies:
        return searched_movies[:limit], f"Here are some movie recommendations for \"{query}\":"

    popular_movies = trending()

    if popular_movies:
        return popular_movies[:limit], "Here are some popular movies you can try:"

    genre_names = {878: "Sci-Fi", 10751: "Children"}
    if genre_id:
        name = genre_names.get(genre_id, next((k.title() for k, v in GENRE_KEYWORDS.items() if v == genre_id), ""))
        rows = movies[movies["genres"].str.split("|").apply(lambda values: name in values)]
    else:
        rows = movies[movies["title"].str.contains(query, case=False, regex=False)]
    if not rows.empty:
        return catalog_cards(rows.head(limit)), "Here are matches from the local movie catalog:"
    return [], "No matches found. Try a movie title or a genre such as comedy or science fiction."


def build_clickable_assistant_movies(recommended_movies, limit=5):

    clickable = []

    for item in recommended_movies:

        poster = item.get("poster") or fetch_poster(item["tmdb"])

        clickable.append({
            "title": item["title"],
            "tmdb": item["tmdb"],
            "poster": poster
        })

        if len(clickable) == limit:
            break

    return clickable


def build_assistant_text_response(header, recommended_movies):

    if not recommended_movies:
        return "I could not find recommendations for that request right now."

    lines = [header, ""]

    for item in recommended_movies:
        lines.append(f"- {item['title']}")

    return "\n".join(lines)


def render_movie_banner_grid(movies_list, key_prefix, max_columns):

    if not movies_list:
        return

    show_count = min(max_columns, len(movies_list))
    cols = st.columns(show_count)

    for i, movie in enumerate(movies_list[:show_count]):

        with cols[i]:

            st.image(movie.get("poster") or PLACEHOLDER, width="stretch")

            if st.button(movie["title"], key=f"{key_prefix}{i}"):

                st.session_state.movie_page = {
                    "title": movie["title"],
                    "tmdb": movie["tmdb"]
                }

                st.rerun()


# -----------------------------
# Session state
# -----------------------------
if "movie_page" not in st.session_state:
    st.session_state.movie_page=None

if "recommendations" not in st.session_state:
    st.session_state.recommendations=None

if "assistant_query" not in st.session_state:
    st.session_state.assistant_query=None

if "assistant_text_response" not in st.session_state:
    st.session_state.assistant_text_response=None

if "assistant_clickable_movies" not in st.session_state:
    st.session_state.assistant_clickable_movies=[]

if "assistant_fallback_note" not in st.session_state:
    st.session_state.assistant_fallback_note=None


# -----------------------------
# Movie detail page
# -----------------------------
def show_movie(movie):

    poster = fetch_poster(movie["tmdb"])
    rating, release, overview = movie_details(movie["tmdb"])
    vid = trailer(movie["tmdb"])

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(poster)

    with col2:
        st.title(movie["title"])
        st.write(f"⭐ Rating: {rating}")
        st.write(f"📅 Release Date: {release}")
        st.write("### Overview")
        st.write(overview)

    if not overview:
        st.info("Movie details are temporarily unavailable. Please try again later.")

    if vid:
        st.write("### 🎬 Trailer")
        st.video(vid)

    else:
        st.caption("No trailer available for this movie.")

    if st.button("⬅ Back to Home"):
        st.session_state.movie_page = None
        st.rerun()


# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>

/* Keep styling theme-aware so Streamlit handles both Light and Dark modes */
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

img{
border-radius:8px;
transition: transform 0.3s ease;
}

img:hover{
transform: scale(1.03);
}

</style>
""",unsafe_allow_html=True)


# -----------------------------
# If movie page
# -----------------------------
if st.session_state.movie_page:

    show_movie(st.session_state.movie_page)
    st.stop()


# -----------------------------
# Hero
# -----------------------------
st.title("🎬 NextFlix")

st.write("Discover your next favorite movie")
if not TMDB_KEY:
    st.info("Browsing the local catalog. Add a TMDB API key to enable posters, trending movies, actor search, and trailers.")


# -----------------------------
# Genre filter
# -----------------------------
genres=set()

for g in movies["genres"]:
    for x in g.split("|"):
        genres.add(x)

genre=st.selectbox("Filter by Genre",["All"]+sorted(list(genres)))

if genre!="All":
    filtered_movies=movies[movies["genres"].str.split("|").apply(lambda values: genre in values)]
else:
    filtered_movies=movies


# -----------------------------
# Trending row
# -----------------------------
st.subheader("🔥 Trending")

trend=trending()

if trend:
    render_movie_banner_grid(trend,"t",6)
else:
    st.caption("Trending movies are temporarily unavailable. Explore the catalog below.")


st.write("---")




# -----------------------------
# Search
# -----------------------------
movie=st.selectbox("Search Movie",filtered_movies["title"].values)

if st.button("Recommend", type="primary"):
    st.session_state.recommendations=recommend(movie)


# -----------------------------
# Show Recommended Movies
# -----------------------------
if st.session_state.recommendations:

    results=st.session_state.recommendations

    render_movie_banner_grid(results,"rec",5)


st.write("---")


# -----------------------------
# Movie assistant
# -----------------------------
with st.expander("Recommendations from similar viewers"):
    st.caption("Explore recommendations for an anonymous MovieLens sample viewer, based on ratings from similar viewers.")
    viewer = st.selectbox("Sample viewer", list(range(1, 611)))
    if st.button("Find viewer recommendations"):
        st.session_state["viewer_recommendations"] = catalog_cards(
            collaborative_recommendations(movies, load_collaborative(), viewer))
    if st.session_state.get("viewer_recommendations"):
        render_movie_banner_grid(st.session_state["viewer_recommendations"], "viewer", 5)

st.subheader("🤖 Movie Assistant")
st.caption("Ask for recommendations, compare movies, or follow up with your preferences.")
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if st.button("Clear conversation"):
    st.session_state.conversation = []
    st.session_state.assistant_clickable_movies = []
    st.session_state.assistant_fallback_note = None
if not GROQ_KEY:
    st.caption("Movie search mode · Configure Groq to enable AI conversation.")

query=st.chat_input("Ask for movie suggestions")

if query:

    with st.spinner("Finding movies for you..."):
        ai_result = groq_response(query, GROQ_KEY, st.session_state.conversation)
        recommended_movies = []
        if ai_result:
            response_text = ai_result["reply"]
            seen = set()
            for title in ai_result["titles"]:
                matches = fetch_movies_for_query(title, limit=1)
                if not matches:
                    rows = movies[movies["title"].str.contains(title, case=False, regex=False)]
                    matches = catalog_cards(rows.head(1))
                for item in matches:
                    if item["title"] not in seen:
                        recommended_movies.append(item)
                        seen.add(item["title"])
            st.session_state.assistant_fallback_note = None
        else:
            recommended_movies, header = assistant_movie_recommendations(query, limit=5)
            response_text = build_assistant_text_response(header, recommended_movies)
            st.session_state.assistant_fallback_note = (
                "Groq is temporarily unavailable. Showing movie search results instead."
                if GROQ_KEY else "Showing movie search results. AI conversation needs a Groq API key.")
        st.session_state.assistant_clickable_movies = build_clickable_assistant_movies(recommended_movies)
        st.session_state.conversation.extend([
            {"role": "user", "content": query[:3000]},
            {"role": "assistant", "content": response_text}])
        st.session_state.conversation = st.session_state.conversation[-10:]
        # Retain the latest response for integrations and regression checks.
        st.session_state.assistant_text_response = response_text

for message in st.session_state.conversation:
    st.chat_message(message["role"]).write(message["content"])

if st.session_state.assistant_fallback_note:
    st.info(st.session_state.assistant_fallback_note)

if st.session_state.assistant_clickable_movies:

    st.write("### 🎥 Explore movies")

    render_movie_banner_grid(
        st.session_state.assistant_clickable_movies,
        "ai",
        len(st.session_state.assistant_clickable_movies)
    )
