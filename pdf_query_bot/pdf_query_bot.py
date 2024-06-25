import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import cassio

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

st.title("PDF Query Application")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None,
    )

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    astra_vector_store.add_texts(texts)

    st.write(f"Inserted {len(texts)} text chunks.")

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    query_text = st.text_input("Enter your question:")
    if st.button("Submit Query"):
        if query_text:
            st.write(f"QUESTION: \"{query_text}\"")
            answer = astra_vector_index.query(query_text, llm=llm).strip()
            st.write(f"ANSWER: \"{answer}\"")

            # st.write("FIRST DOCUMENTS BY RELEVANCE:")
            # for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
            #     st.write(f"[{score:.4f}] \"{doc.page_content[:84]} ...\"")
        else:
            st.write("Please enter a question.")
