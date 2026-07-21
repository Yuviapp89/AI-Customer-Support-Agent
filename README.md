AI Customer Support Agent
An intelligent, multi‑agent customer support system powered by Retrieval‑Augmented Generation (RAG), LangGraph, OpenAI, and Pinecone.
This chatbot retrieves product information from a structured jewellery knowledge base and generates accurate, context‑aware responses through a Streamlit interface.

📛 Badges
https://img.shields.io/badge/Python-3.10-blue.svg
https://img.shields.io/badge/Streamlit-App-red.svg
https://img.shields.io/badge/LangChain-RAG-green.svg
https://img.shields.io/badge/Pinecone-VectorDB-purple.svg
https://img.shields.io/badge/OpenAI-GPT--4o--Mini-black.svg

📘 Project Overview
This project implements an AI‑powered Customer Support Agent using RAG.
The system retrieves relevant information from Pinecone and generates grounded responses using OpenAI GPT‑4o Mini.
It supports multi‑turn conversations, intent classification, query rewriting, validation, and product recommendations.

🎯 Objectives
Build an intelligent customer support chatbot

Retrieve accurate product information using Pinecone

Answer complex customer queries using RAG

Recommend alternative products when items are unavailable

Maintain conversation context

Deploy the application using Streamlit

🧰 Technology Stack
Component	Technology
Programming Language	Python 3.10
Frameworks	LangChain, LangGraph, Streamlit
LLM	OpenAI GPT‑4o Mini
Embeddings	OpenAI text-embedding-3-small
Vector DB	Pinecone Serverless
Development	Google Colab
Knowledge Base	Jewellery Catalog, Warranty, Shipping, Returns, Care Guide, FAQ


⚙️ Installation & Setup
1. Clone the Repository
Code
git clone https://github.com/Yuviapp89/AI-Customer-Support-Agent.git
cd AI-Customer-Support-Agent
2. Install Dependencies
Code
pip install -r requirements.txt
3. Add Your API Keys
Use Streamlit Secrets when deploying:

Code
[general]
OPENAI_API_KEY = "your_key"
PINECONE_API_KEY = "your_key"
PINECONE_INDEX_NAME = "your_index"
LANGSMITH_API_KEY = "your_key"
LANGSMITH_TRACING = true
LANGSMITH_PROJECT = "your_project"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
For local development, you may also use .env.

🚀 Running the Application
Local
Code
streamlit run app.py
Cloud (Streamlit Sharing)
Upload the repo → Add secrets → Deploy.

🔄 Project Workflow
Install required libraries

Load API keys

Connect to Pinecone

Load product knowledge base

Perform semantic chunking

Generate embeddings

Store vectors in Pinecone

Retrieve relevant documents

Generate AI responses

Launch Streamlit chatbot

📊 Architecture Diagram
Code
                      Jewellery PDF Documents
  (Product Catalog, Warranty, Shipping, Returns, Care Guide, FAQ)
                                  │
                                  ▼
                       Document Loading & Parsing
                                  │
                                  ▼
               Semantic Chunking + Metadata Enhancement
                                  │
                                  ▼
                 OpenAI Embeddings (text-embedding-3-small)
                                  │
                                  ▼
                  Pinecone Vector Database (Serverless)
                                  │
                                  ▼
                          Customer Question
                                  │
                                  ▼
                         Initialize Agent
                                  │
                                  ▼
                    Intent Classification Agent
                                  │
                                  ▼
                       Query Rewriter Agent
                                  │
                                  ▼
                       Document Retrieval Agent
                                  │
                                  ▼
                         Response Agent (LLM)
                                  │
                                  ▼
                       Response Validation Agent
                                  │
                                  ▼
                      Recommendation Agent
                                  │
                                  ▼
                       Final Customer Response
                                  │
                                  ▼
                       Streamlit Chat Interface
✨ Features Implemented
Multi‑agent workflow using LangGraph

Retrieval‑Augmented Generation (RAG)

Semantic chunking with overlap

Metadata‑enhanced retrieval

OpenAI embeddings for similarity search

Intent Classification Agent

Query Rewriter Agent

Document Retrieval Agent

Response Generation Agent (grounded in retrieved docs)

Response Validation Agent (reduces hallucinations)

Recommendation Agent for alternative products

Multi‑turn conversation support

Source‑aware responses

Fallback responses for unsupported queries

Modular agent architecture

Streamlit chat interface

Cloud‑based implementation using Google Colab, OpenAI, Pinecone

📂 Repository Structure
Code
AI-Customer-Support-Agent/
│
├── data/                 # PDF knowledge base
├── embeddings/           # Vector storage (optional local)
├── src/
│   ├── agents/           # Multi-agent logic
│   ├── rag/              # RAG pipeline
│   ├── utils/            # Helper functions
│   └── config.py         # API keys & settings
│
├── app.py                # Streamlit interface
├── requirements.txt
└── README.md
