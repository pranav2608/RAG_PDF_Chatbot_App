import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

## Required API KEY 
os.environ['GROQ_KEY'] = os.getenv("GROQ_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_KEY")

# Intialising LLM and creating Prompt Template
llm = ChatGroq(groq_api_key=groq_api_key,model_name='gemma2-9b-it')
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on provided context only
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Create the vector embeddings & Storing it in session state of the app (browser)
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") # Document Ingestion (as an enhancement , you can think of taking pdf inputs from user using streamlit app)
        st.session_state.docs = st.session_state.loader.load() #Document load complete
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A Application")
user_query = st.text_input("Enter your query from the provided documents")
if st.button("Start: Create Embedding"):
    create_vector_embedding()
    st.write("Vector Database is Ready")

if user_query:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    rag_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = rag_chain.invoke({"input": user_query})
    print(f'Response time : {time.process_time() - start}')

    st.write(response['answer'])

    # Using Streamlit expander to show the context used while answering the user query
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------------')

