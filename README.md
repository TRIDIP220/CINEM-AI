# 🎬 CinemAI — Explainable Movie Recommender & Sentiment Analyst

**CinemAI** is an advanced movie discovery and analysis platform that integrates deep learning sentiment classifiers, semantic search, and generative conversational AI. Built with a decoupled **FastAPI** backend and an interactive **Streamlit** frontend, it provides explainable and conversational recommendations.

### Key Features
* **Hybrid Recommendation Engine**: Leverages **Sentence Transformers** (`all-MiniLM-L6-v2`) to perform semantic vector searches on movie plots, combined with **Collaborative Filtering** Jaccard similarities mapped from over **1.5 million MovieLens ratings**.
* **Transformer Sentiment & ABSA**: Features a fine-tuned **DistilBERT** classifier for movie reviews, coupled with **Aspect-Based Sentiment Analysis (ABSA)** that separately evaluates and graphs audience sentiment for **Acting**, **Plot/Story**, and **Visuals/CGI**.
* **Conversational RAG Chat Agent**: Uses a **Retrieval-Augmented Generation (RAG)** pipeline. The agent retrieves the top semantically matching films from the database, constructs a context-aware prompt, and uses the **Groq API (Llama 3)** to generate engaging, conversational recommendations (with an offline fallback).
* **Interactive Analytics**: Displays visual insights on genre distributions, rating spreads, and popularity scatter plots.

### Technology Stack
* **Frontend**: Streamlit, HTML5, Custom CSS
* **Backend**: FastAPI, Uvicorn, Pydantic, Python-dotenv
* **ML & AI**: PyTorch, Hugging Face Transformers, Sentence-Transformers, Groq SDK
* **Data Processing**: Pandas, NumPy, Scikit-Learn
