import streamlit as st
import pickle
import pandas as pd
import requests
from groq import Groq

# -----------------------------
# Load models
# -----------------------------
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

TMDB_KEY = st.secrets["TMDB_API_KEY"]
GROQ_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_KEY)

@st.cache_data
# -----------------------------
# Poster
# -----------------------------
def fetch_poster(tmdb):

    url=f"https://api.themoviedb.org/3/movie/{tmdb}?api_key={TMDB_KEY}"
    data=requests.get(url).json()

    if "poster_path" in data and data["poster_path"]:
        return "https://image.tmdb.org/t/p/w500"+data["poster_path"]

    return "https://via.placeholder.com/500x750?text=No+Poster"


# -----------------------------
# Movie details
# -----------------------------
def movie_details(tmdb):

    url=f"https://api.themoviedb.org/3/movie/{tmdb}?api_key={TMDB_KEY}"
    data=requests.get(url).json()

    rating=data.get("vote_average","N/A")
    release=data.get("release_date","")
    overview=data.get("overview","")

    return rating,release,overview


# -----------------------------
# Trailer
# -----------------------------
def trailer(tmdb):

    url=f"https://api.themoviedb.org/3/movie/{tmdb}/videos?api_key={TMDB_KEY}"
    data=requests.get(url).json()

    for v in data.get("results",[]):

        if v["type"]=="Trailer" and v["site"]=="YouTube":

            return f"https://www.youtube.com/watch?v={v['key']}"

    return None


# -----------------------------
# Trending
# -----------------------------
def trending():

    url=f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_KEY}"
    data=requests.get(url).json()

    movies_list=[]

    for m in data["results"][:6]:

        movies_list.append({
            "title":m["title"],
            "poster":"https://image.tmdb.org/t/p/w500"+m["poster_path"],
            "tmdb":m["id"]
        })

    return movies_list


# -----------------------------
# Recommendation
# -----------------------------
def recommend(movie):

    idx=movies[movies["title"]==movie].index[0]
    distances=similarity[idx]

    movie_list=sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x:x[1]
    )[1:6]

    results=[]

    for i in movie_list:

        row=movies.iloc[i[0]]

        results.append({
            "title":row["title"],
            "tmdb":row["tmdbId"],
            "poster":fetch_poster(row["tmdbId"])
        })

    return results


# -----------------------------
# AI assistant
# -----------------------------
def ai_movie_assistant(query):

    sample=movies["title"].sample(100).tolist()

    prompt=f"""
User request: {query}

Available movies:
{sample}

Recommend 3 movies.
"""

    response=client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {"role":"system","content":"You are a movie expert"},
            {"role":"user","content":prompt}
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "movie_page" not in st.session_state:
    st.session_state.movie_page=None

if "recommendations" not in st.session_state:
    st.session_state.recommendations=None


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

    if vid:
        st.write("### 🎬 Trailer")
        st.video(vid)

    if st.button("⬅ Back to Home"):
        st.session_state.movie_page = None
        st.rerun()


# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>

body{
background:#0e1117;
color:white;
}

.stApp{
background:#0e1117;
}

img{
border-radius:8px;
transition: transform 0.3s ease;
}

img:hover{
transform: scale(1.15);
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


# -----------------------------
# Genre filter
# -----------------------------
genres=set()

for g in movies["genres"]:
    for x in g.split():
        genres.add(x)

genre=st.selectbox("Filter by Genre",["All"]+sorted(list(genres)))

if genre!="All":
    filtered_movies=movies[movies["genres"].str.contains(genre)]
else:
    filtered_movies=movies


# -----------------------------
# Trending row
# -----------------------------
st.subheader("🔥 Trending")

trend=trending()

cols=st.columns(6)

for i,m in enumerate(trend):

    with cols[i]:

        st.image(m["poster"])

        if st.button(m["title"],key=f"t{i}"):

            st.session_state.movie_page=m
            st.rerun()


st.write("---")




# -----------------------------
# Search
# -----------------------------
movie=st.selectbox("Search Movie",filtered_movies["title"].values)

if st.button("Recommend"):
    st.session_state.recommendations=recommend(movie)


# -----------------------------
# Show Recommended Movies
# -----------------------------
if st.session_state.recommendations:

    results=st.session_state.recommendations

    cols=st.columns(5)

    for i,m in enumerate(results):

        with cols[i]:

            st.image(m["poster"],use_container_width=True)

            if st.button(m["title"],key=f"rec{i}"):

                st.session_state.movie_page={
                    "title":m["title"],
                    "tmdb":m["tmdb"]
                }

                st.rerun()


st.write("---")


# -----------------------------
# AI chatbot
# -----------------------------
st.subheader("🤖 Movie Assistant")

query=st.chat_input("Ask for movie suggestions")

if query:

    st.chat_message("user").write(query)

    response=ai_movie_assistant(query)

    st.chat_message("assistant").write(response)
