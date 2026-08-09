# RAG Streamlit App (Improved with Original PDF Source Attribution)
# Compatible with:
# langchain==0.3.27
# langchain-community==0.3.27
# langchain-core==0.3.86
# langchain-openai==0.3.28

import os
import tempfile
from collections import defaultdict

import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

st.set_page_config(page_title="Research Paper RAG", layout="wide")
st.title("📄 Research Paper RAG Chatbot")

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.session_state["api_key"] = api_key

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Submit and Process"):

        if "api_key" not in st.session_state:
            st.error("Enter OpenAI API Key")
            st.stop()

        if not uploaded_files:
            st.error("Upload at least one PDF")
            st.stop()

        documents = []

        with st.spinner("Loading PDFs..."):

            for pdf in uploaded_files:

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf.getvalue())
                    temp_path = tmp.name

                loader = PyMuPDFLoader(temp_path)
                docs = loader.load()

                # Replace temporary filename with original uploaded filename
                for doc in docs:
                    doc.metadata["source"] = pdf.name

                documents.extend(docs)

                os.remove(temp_path)

        with st.spinner("Splitting Documents..."):

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300
            )

            chunks = splitter.split_documents(documents)

        with st.spinner("Creating Vector Database..."):

            embeddings = OpenAIEmbeddings(
                api_key=st.session_state["api_key"]
            )

            db = FAISS.from_documents(
                chunks,
                embeddings
            )

            db.save_local("faiss_index")

        st.success(f"Indexed {len(chunks)} chunks successfully.")

with st.form("question_form"): #it waits untill user submit the button

    user_question = st.text_input(
        "Ask a question about the uploaded research papers"
    )

    ask_button = st.form_submit_button("Ask")

if ask_button and user_question:

    if "api_key" not in st.session_state:
        st.error("Enter OpenAI API Key")
        st.stop()

    if not os.path.exists("faiss_index"):
        st.error("Please upload and process PDFs first.")
        st.stop()

    embeddings = OpenAIEmbeddings(
        api_key=st.session_state["api_key"]
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 30,
            "lambda_mult": 0.7
        }
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=st.session_state["api_key"]
    )

    prompt = ChatPromptTemplate.from_template(
"""
You are an expert research assistant.

Answer ONLY using the retrieved context.

If information is spread across multiple chunks, combine it.

Do not make up information.

If the answer cannot be found, reply exactly:

Answer is not available in the uploaded documents.

At the end of the answer, briefly mention which documents were used.

Context:
{context}

Question:
{input}

Answer:
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )#it is the context 

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )#it is sending to lllm with retrived docs 

    with st.spinner("Searching..."):
        response = retrieval_chain.invoke({"input": user_question})

    st.subheader("Answer")
    st.write(response["answer"])

    st.divider()

    st.subheader("📚 Source Attribution")

    sources = defaultdict(list)

    for doc in response["context"]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        if page not in sources[source]:
            sources[source].append(page)

    for source, pages in sources.items():
        pages.sort()
        st.markdown(f"**{source}** (Pages: {', '.join(map(str, pages))})")
        #map converts page numbers to string , and join can only join strings so using map 

    st.divider()
