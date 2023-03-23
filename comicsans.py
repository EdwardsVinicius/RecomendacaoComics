import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

comics_data = pd.read_json('marvel_movies.json')
comics_df = pd.DataFrame(comics_data)

comics_df['genres'] = comics_df['genres'].apply(lambda x: ', '.join(x))

movies_data = pd.read_json('comics.json')
movies_df = pd.DataFrame(movies_data)

movies_df['genres'] = movies_df['genres'].apply(lambda x: ', '.join(x))

tfidf = TfidfVectorizer()

comics_matrix = tfidf.fit_transform(comics_df['genres'])

cosine_similarities = cosine_similarity(comics_matrix, tfidf.transform(movies_df['genres']))

def recommend_movies(comic_name):
    try:
        comic_index = comics_df.loc[comics_df['name'] == comic_name].index[0]

        sim_scores = list(enumerate(cosine_similarities[comic_index]))

        sim_scores = [x for x in sim_scores if comics_df.loc[x[0], 'name'] != comic_name]

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        movie_indices = [i[0] for i in sim_scores[0:5]]

        return movies_df['name'].iloc[movie_indices]
    except:
        return 'Comic not found'
    

@app.get("/")
async def recommend(data):
    return recommend_movies(data)

# para iniciar o servidor uvicorn
# uvicorn comicsans:app --reload 