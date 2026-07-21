AI Customer Support Agent
**Project Overview**
This project implements an AI-powered Customer Support Agent using Retrieval-Augmented Generation (RAG).

The chatbot answers customer queries by retrieving information from a product knowledge base stored in Pinecone and generating responses using OpenAI models.

**Objectives**

1.	Build an intelligent customer support chatbot
2.	Retrieve accurate product information using Pinecone
3.	Answer complex customer queries using RAG
4.	Recommend alternative products when requested items are unavailable
5.	Maintain conversation context
6.	Deploy the application using Streamlit


**Technology Stack**
1.	Programming Language: Python 3.10
2.	Frameworks: LangChain, LangGraph, Streamlit
3.	LLM: OpenAI GPT-4o Mini
4.	Embedding Model: OpenAI text-embedding-3-small
5.	Vector Database: Pinecone Serverless
6.	Development Environment: Google Colab
7.	Knowledge Base: Jewellery Product Catalog, Warranty Policy, Shipping Policy, Return Policy, Jewellery Care Guide, FAQ
8.	Project Workflow
9.	Install required libraries
10.	Load API keys
11.	Connect to Pinecone
12.	Load product knowledge base
13.	Retrieve relevant documents
14.	Generate AI responses
15.	Launch the Streamlit chatbot


Flow Diagram
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
                       
Features Implemented

1.	Multi-Agent Customer Support workflow using LangGraph.
2.	Retrieval-Augmented Generation (RAG) using a Pinecone vector database.
3.	Semantic document chunking with overlap to preserve context.
4.	Metadata enhancement for improved document retrieval accuracy.
5.	OpenAI embeddings for semantic similarity search.
6.	Intent Classification Agent to identify customer query categories.
7.	Query Rewriter Agent to transform follow-up questions into standalone queries for better retrieval.
8.	Intelligent document retrieval with metadata-based filtering and ranking.
9.	Response Generation Agent using retrieved knowledge only.
10.	Response Validation Agent to reduce hallucinations and ensure answers are grounded in retrieved documents.
11.	Recommendation Agent to suggest alternative products when an item is unavailable or when recommendations are requested.
12.	Conversation history support for multi-turn customer interactions.
13.	Source-aware responses generated only from the uploaded knowledge base.
14.	Automatic handling of unsupported or unavailable information with appropriate fallback responses.
15.	Modular agent architecture that separates retrieval, reasoning, validation, and recommendation tasks.
16.	Interactive Streamlit chat interface for real-time customer support.
17.	Cloud-based implementation using Google Colab, OpenAI, and Pinecone.
