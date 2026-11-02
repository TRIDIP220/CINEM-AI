from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from dotenv import load_dotenv
from groq import Groq

from backend.schemas import ExplainRequest, RecommendRequest, SentimentRequest
from backend.services.recommender import MovieRecommender
from backend.services.sentiment import SentimentAnalyzer


# Load environment variables from .env file in project root
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT / ".env")

app = FastAPI(
    title="AI Movie Sentiment and Recommendation API",
    description="Transformer-based sentiment analysis plus TMDB content-based movie recommendations.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment = SentimentAnalyzer()
recommender = MovieRecommender()

# Initialize Groq client if key is set
api_key = os.getenv("GROQ_API_KEY")
groq_client = None
if api_key and "your_groq_api" not in api_key and api_key.strip():
    try:
        groq_client = Groq(api_key=api_key)
        print("Groq Client initialized successfully!")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Movie AI API is running. Visit /docs for interactive API documentation."}


@app.get("/api/health")
@app.get("/api/status")
def health() -> dict[str, object]:
    # Check if groq is connected
    groq_active = False
    if groq_client:
        groq_active = True
    else:
        # Check if environment variable changed
        load_dotenv(dotenv_path=ROOT / ".env")
        new_key = os.getenv("GROQ_API_KEY")
        if new_key and "your_groq_api" not in new_key and new_key.strip():
            groq_active = True

    return {
        "status": "online",
        "device": str(sentiment.device),
        "sentiment_models_loaded": sentiment.loaded,
        "available_sentiment_models": ["distilbert_transformer"] if sentiment.loaded else ["lexicon_fallback"],
        "movie_count": len(recommender.df),
        "recommendation_engine": "semantic_transformer_collab_hybrid",
        "groq_connected": groq_active,
    }


@app.get("/api/movies")
def movies() -> dict[str, list[str]]:
    return {"movies": recommender.titles()}


@app.post("/api/sentiment")
def analyze_sentiment(payload: SentimentRequest) -> dict[str, object]:
    prediction = sentiment.predict(payload.text)
    return {
        "prediction": prediction,
        "attributions": sentiment.attributions(payload.text),
    }


@app.post("/api/sentiment/aspects")
def analyze_aspects(payload: SentimentRequest) -> dict[str, object]:
    return {"aspects": sentiment.predict_aspects(payload.text)}


@app.post("/api/recommend")
def recommend(payload: RecommendRequest) -> dict[str, object]:
    review_sentiment = None
    if payload.review:
        review_sentiment = sentiment.predict(payload.review)["label"]
    try:
        recommendations = recommender.recommend(payload.title, payload.top_n, review_sentiment)
        selected_movie = recommender.get_movie(payload.title)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie title not found: {payload.title}") from None
    return {
        "selected_movie": selected_movie,
        "review_sentiment": review_sentiment,
        "recommendations": recommendations,
    }


@app.post("/api/explain")
def explain(payload: ExplainRequest) -> dict[str, str]:
    user_sentiment = None
    if payload.user_review:
        user_sentiment = sentiment.predict(payload.user_review)["label"]
    try:
        explanation = recommender.explanation(payload.selected_title, payload.recommended_title, user_sentiment)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Movie title not found: {exc}") from None
    return {"explanation": explanation}


@app.post("/api/chat")
def chat(payload: dict) -> dict[str, object]:
    message = payload.get("message", "")
    
    # 1. Retrieve top 5 candidate movies semantically using SentenceTransformers cosine similarity
    try:
        query_emb = recommender.embedder.encode(message)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        
        sims = np.dot(recommender.embeddings, query_emb)
        top_indices = np.argsort(sims)[::-1][:5]
        
        recs = []
        for idx in top_indices:
            row = recommender.df.iloc[idx].copy()
            similarity = float(sims[idx])
            row["score"] = similarity
            row["semantic_similarity"] = similarity
            row["collaborative_similarity"] = 0.0
            row["genre_similarity"] = 0.0
            recs.append(recommender._movie_payload(row))
    except Exception as e:
        return {
            "reply": f"Sorry, I had trouble retrieving movies from the database. Error: {str(e)}",
            "movies": []
        }

    # 2. Check if Groq client needs initialization
    global groq_client
    if not groq_client:
        load_dotenv(dotenv_path=ROOT / ".env")
        new_api_key = os.getenv("GROQ_API_KEY")
        if new_api_key and "your_groq_api" not in new_api_key and new_api_key.strip():
            try:
                groq_client = Groq(api_key=new_api_key)
            except Exception:
                pass

    # 3. If Groq is active, use Retrieval-Augmented Generation (RAG)
    if groq_client:
        try:
            # Build Context String
            context = ""
            for idx, r in enumerate(recs, 1):
                genres_str = ", ".join(r["genres"])
                context += f"Movie {idx}: {r['title']} ({r['year']})\n"
                context += f"- Genres: {genres_str}\n"
                context += f"- Director: {r['director']}\n"
                context += f"- Audience Rating: {r['vote_average']}/10\n"
                context += f"- Overview: {r['overview']}\n\n"

            prompt = f"""You are a helpful, professional, and friendly movie recommendation assistant named CinemAI.
Analyze the user's description and preferences, read the context of retrieved movies from the database below, and recommend the best options.

USER REQUEST:
"{message}"

RETRIEVED MOVIES CONTEXT:
{context}

INSTRUCTIONS:
- Directly answer the user's request.
- Based on the context, choose 2 to 3 movies that are the absolute best match for the user's request.
- Reference details from the context (e.g. director, plot elements, or rating) in a highly conversational way to explain why you recommended them.
- Keep your response engaging, friendly, and concise (under 250 words).
- Do not mention or make up any movies that are not listed in the retrieved context.
"""
            completion = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are CinemAI, a conversational movie RAG assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            reply = completion.choices[0].message.content
            return {
                "reply": reply,
                "movies": recs[:3] # Return top 3 matching movie objects for display cards
            }
        except Exception as e:
            print(f"Groq API Error: {e}")

    # 4. Fallback to offline formatting if Groq is unavailable
    reply = "Here are a few movies matching your interest based on local semantic search:\n\n"
    for idx, r in enumerate(recs[:3], 1):
        reply += f"{idx}. **{r['title']}** ({r['year']}) — {r['overview'][:150]}...\n"
    
    return {
        "reply": reply,
        "movies": recs[:3]
    }


@app.get("/api/stats")
def stats() -> dict[str, object]:
    return recommender.stats()
