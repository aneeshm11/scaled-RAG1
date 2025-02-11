# pip install transformers sentence-transformers langchain langchain-community langchain-openai langchain-google-genai langchain-groq python-dotenv chromadb  PyPDF2 -q
import os 
import re
import torch
import numpy as np 
from groq import Groq
from langchain import hub
from PyPDF2 import PdfReader
from operator import itemgetter
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.load import dumps, loads
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from html_templates import css, bot_template, user_template



env_file_path = '/kaggle/input/interndata/.env'
load_dotenv(env_file_path)

# login(os.getenv("HF_API_KEY"))
# model_id = "meta-llama/Meta-Llama-3.1-70B"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="second-project"

client = Groq()
llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature=0)


model_name = "bert-base-uncased"  
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name)

def chat(question):
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    answer = chat_completion.choices[0].message.content
    
    return answer 

def embeddings(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return np.array(cls_embedding)

def similar(query_result, document_result):
    return cosine_similarity(query_result, document_result)[0][0]

def pick(text):
    match = re.search(r'\{(.*?)\}', text, re.DOTALL)
    if match:
        extract_text = match.group(1).strip()
    else:
        extract_text = ""
    return extract_text

def clean_doc(docs): # takes in pdf path , loads , cleans and returns document 
    
    cleandocs = []
    pdf_reader = PdfReader(docs)
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()

        question= """
        
        I need you to do the following actions:
        Clean the following text by removing any components related to date, time, person's name, PDF source, PDF number , page number and PDF name, name of persons wherever mentioned,
        pdf exported date, author's name, deviation number, document number, reference number. These should be strictly removed and nothing else.
        Remove excessive whitespaces to trim down the text.
        Don't remove any numbers that follow any chemical names , for example if there is something like Mycophenolate Sodium BS21001952, dont remove this type of number.
        Return the processed text enclosed within curly braces {{ }}
        Perform these actions on the following text:
                    
        """ 
        
        formated_q =  f"{question} {text} "
    
        clean_text = chat(formated_q)

        match = re.search(r'\{(.*?)\}', clean_text, re.DOTALL)
        if match:
            extracted_clean_text = match.group(1).strip()
        else:
            extracted_clean_text = ""

        if extracted_clean_text:
            cleandocs.append(Document(page_content=extracted_clean_text, metadata={"source": docs, "page": page_num + 1}))

        loaderp= PyPDFLoader(docs)
        originaldocs = loaderp.load()

    return cleandocs , originaldocs

cleandocs , originaldocs = clean_doc("/kaggle/input/interndata/3.pdf")
cleandocs

def hyde(docs , mode): # takes in loaded docs , not pdf, returns plain text

    prompt = """
    I want you to generate a very detailed summary of the following text which captures the main content it is trying to deliver,
    especially any specific goal, problem, issue, or key information, any solutions.

    Return the summarized text enclosed within curly braces {{ }}.

    Perform these actions on the following text:
    """

    fulltext=""

    if mode=="doclevel":
        for x in docs:
            fulltext+= x.page_content
    
    elif mode=="chunklevel" or mode=="textlevel":
        fulltext = docs      # iteration will be handled in the create_vectorstore func

    formated_q =  f"{prompt} {fulltext} "

    clean_text = chat(formated_q)
    summary    = pick(clean_text)

    return summary

def create_datastore(pdf_paths, chunk_size, chunk_overlap , mode):
    result = []

    for pdf_path in pdf_paths:

        clean_docs,_ = clean_doc(pdf_path)  # clean_doc returns a loaded+cleaned doc of a pdf without noise

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(clean_docs)

        myembeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(documents=splits, embedding=myembeddings)

        hyde_summaries = None 
        
        if mode=="doclevel":
            doc_hyde_summaries   = hyde(clean_docs , mode)
            hyde_summaries       = doc_hyde_summaries

        elif mode=="chunklevel":
            chunk_hyde_summaries = [hyde(chunk.page_content , mode) for chunk in splits]
            hyde_summaries       = chunk_hyde_summaries
            
        result.append({
            'pdf_path': pdf_path,
            'vectorstore': vectorstore,
            'hyde_summaries': hyde_summaries
        })

    return result 

def search(question, pdf_paths, chunk_size, chunk_overlap, mode, numdocs, numchunks=None):
    datastore = create_datastore(pdf_paths, chunk_size, chunk_overlap, mode)
    q_hyde = hyde(question, mode="textlevel")
    q_hyde_embs = embedding(q_hyde)

    if mode == "doclevel":
        final_vectorstores = []
        
        for data in datastore:
            doc_hyde = data['hyde_summaries']
            score = similar(q_hyde_embs, embedding(doc_hyde))
            
            final_vectorstores.append({
                'pdf_path': data['pdf_path'],
                'vectorstore': data['vectorstore'],
                'score': score
            })
        
        sorted_docs = sorted(final_vectorstores, key=lambda x: x['score'], reverse=True)
        top_docs = sorted_docs[:numdocs]
        
        return top_docs
    
    elif mode == "chunklevel":
        doc_scores = []

        for data in datastore:
            chunk_hyde_summaries = data['hyde_summaries']
            chunk_scores = []
            
            for i, chunk_hyde in enumerate(chunk_hyde_summaries):
                chunk_hyde_embs = embedding(chunk_hyde)
                score = similar(q_hyde_embs, chunk_hyde_embs)
                chunk_scores.append(score)
                
            avg_score = sum(chunk_scores) / len(chunk_scores)
            doc_scores.append({
                'pdf_path': data['pdf_path'],
                'vectorstore': data['vectorstore'],
                'score': avg_score, 
            })

        sorted_docs = sorted(doc_scores, key=lambda x: x['score'], reverse=True)
        top_docs = sorted_docs[:numdocs]
        
        return top_docs

def ask(question, pdf_paths, chunk_size, chunk_overlap, mode, numdocs, top_k,  numchunks=None  ):
    
    vs = []
    f  = []

    datastore = search(question, pdf_paths, chunk_size, chunk_overlap, mode, numdocs, numchunks=None)
    for ds in datastore:
        vs.append(ds["vectorstore"])
        f.append(ds["pdf_path"])

    responses = ""
    for v in vs:
        
        retriever = v.as_retriever(search_kwargs={"k": top_k })  # top_k 
        rel_docs  = retriever.get_relevant_documents(question)

        formated_q =  f"""
        Find the soltuion and insights related to the question below using the following context and information given 
        Question: {question} 
        Context : {rel_docs}
        Return the answer to the text enclosed within curly braces {{ }}.
        """
    
        temp_response = chat(formated_q)
        extracted_text = pick(temp_response)
        responses+=  extracted_text

    final_responses = responses 
    final_formated_q =  f"""
        Find the soltuion and insights related to the question below using the following context and information given
        Question: {question} 
        Context : {final_responses}
        Return the summarized text enclosed within curly braces {{ }}.
        """
    
    final_answer         = chat(final_formated_q)
    final_extracted_text = pick(final_answer)

    return final_extracted_text

def handle_question(question):
    # response = 
    st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)


def get_pdf_text(docs):
    documents = []
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"source": pdf, "page": page_num + 1}))
    return documents


pdf_paths = [
    "/kaggle/input/interndata/3.pdf",
    "/kaggle/input/interndata/4.pdf",    
]

top_k_full_docs_compared = 2  # not related to mode=doclevel or chunklevel, both methods keep top_k_full_docs_compared, but here scoring sees these top_k summaries of fulldoc for doclevel and all chunks for chunklevel
top_k_retrieval = 3           # not related to mode=doclevel or chunklevel, both methods keep top_k_full_docs_compared, but here scoring sees these top_k summaries of fulldoc for doclevel and all chunks for chunklevel

chunk_size    = 400
chunk_overlap = 10
mode="doclevel"

numdocs = top_k_full_docs_compared
top_k= top_k_retrieval


question= "why did the typo error occur in the Lipstatin seed inoculum development BMR "

out = create_datastore(pdf_paths , chunk_size ,chunk_overlap )
# out = search(question, pdf_paths, chunk_size, chunk_overlap, mode, numdocs)
# out = ask(question, pdf_paths, chunk_size, chunk_overlap, mode, numdocs, top_k,  numchunks=None  )


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
