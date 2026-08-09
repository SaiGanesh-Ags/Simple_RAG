Simple RAG Streamlit App

A Retrieval-Augmented Generation (RAG) application that allows users toupload multiple research papers in PDF format, index their content in aFAISS vector database, and ask questions using an OpenAI-poweredchatbot.

Overview

This project helps users retrieve information from uploaded researchpapers without manually searching through large PDF documents. Theapplication loads PDFs, splits their content into chunks, createsembeddings, stores them in FAISS, retrieves relevant chunks using MMR,and generates answers using an OpenAI chat model. It also displays thesource documents and page numbers used for the answer.

Features

Upload multiple research paper PDFs through a Streamlit interface

Extract and split PDF content into searchable chunks

Generate OpenAI embeddings and store them in a FAISS vector database

Retrieve relevant context using Maximum Marginal Relevance (MMR)

Generate answers using OpenAI GPT-4o-mini

Prevent unsupported answers by restricting responses to retrieveddocument context

Display source document names and page numbers used for each answer

Save and reload the FAISS vector index locally

Tech Stack

Python • Streamlit • LangChain • OpenAI • FAISS • PyMuPDF

Setup

git clone <REPO-URL>
cd <REPO-NAME>
pip install -r requirements.txt

The project uses the following compatible package versions:

langchain==0.3.27
langchain-community==0.3.27
langchain-core==0.3.86
langchain-openai==0.3.28

Run

streamlit run rag_streamlit_assignment.py

Project Structure

.
├── rag_streamlit_assignment.py    → Main Streamlit application
├── faiss_index/                   → Locally saved FAISS vector index
├── requirements.txt               → Python dependencies
└── README.md                      → Project documentation

Usage

Start the Streamlit application.

Enter your OpenAI API key in the sidebar.

Upload one or more research paper PDFs.

Click Submit and Process.

The application loads the PDFs and preserves the original uploadedfilenames as source metadata.

The documents are split into chunks using a recursive character textsplitter.

OpenAI embeddings are generated for the chunks.

The chunks are stored in a local FAISS vector database.

Enter a question about the uploaded research papers.

Click Ask.

The application retrieves relevant chunks using MMR and sends theretrieved context to GPT-4o-mini.

The generated answer is displayed along with the source documentsand page numbers used.

Retrieval Configuration

The application uses MMR retrieval with:

k = 6
fetch_k = 30
lambda_mult = 0.7

The document chunks use:

chunk_size = 1500
chunk_overlap = 300

License

MIT License
