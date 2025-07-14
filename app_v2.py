import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

## Required API KEY 
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Setup Streamlit app
st.title("Converastional RAG with PDF Uploads and Chat History")
st.write("Upload PDf and chat with content")

# Input GROQ Api Key (Taking inputs from User)
api_key = st.text_input("Enter your GROQ API Key")
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name='gemma2-9b-it')

    session_id = st.text_input("Session ID", value='default_session')

    if "store" not in st.session_state:
        st.session_state.store = {}
    
    # Take the pdf file from the user -> store it locally somewhere -> load and convert those files in document
    uploaded_files = st.file_uploader("Choose your PDF file to upload",type="pdf",accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            tempPdf = f'./upload.pdf'
            with open(tempPdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            
            loader = PyPDFLoader(tempPdf)
            docs = loader.load()
            documents.extend(docs)

        # Split the documents created and create its embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()

        # Prompts to create history aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in chat history, "
            "formulate the standalone question which can be understood"
            "without the chat history, DO NOT ANSWER the question"
            "just reformulate as is if needed and otherwise return it as is"
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # Answer question prompt
        system_prompt = (
            "You are an assistant for question answering task"
            "use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer , say that you"
            "don't know. Use three sentences maximum to keep the"
            "answer consise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
                
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question: ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config = {
                    "configurable":{"session_id": session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant: ", response['answer'] )
            st.write("Chat History: ", session_history.messages)
else:
    st.warning('Please enter the GROQ Api key')





