import os
import streamlit as st
from langchain import hub
from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from html_templates import css, bot_template, user_template
from langchain.schema import Document

load_dotenv(".env")

langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

myembeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_pdf_text(docs):
    documents = []
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"source": pdf, "page": page_num + 1}))
    return documents


def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=10)
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_vectorstore(chunks):
    vectorstore = FAISS.from_documents(chunks, myembeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_conversation_chain(vectorstore):
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        | StrOutputParser()
    )

    return rag_chain


def handle_question(question, rag_chain):
    response = rag_chain.invoke(question)
    st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Internproject", page_icon=":gear:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("LLM-RAG system :space_invader:")
    question = st.text_input("Ask question from your document:")
    if question:
        if st.session_state.conversation:
            handle_question(question, st.session_state.conversation)
    
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(docs)
                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
