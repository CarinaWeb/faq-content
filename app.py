import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse

LOCAL_vector_store = None

def loadFileFromURL(text_file_url): #param: https://raw.githubusercontent.com/CarinaWeb/faq-content/main/about-carina.txt
    output_file = "about-carina.txt"
    resp = requests.get(text_file_url)
    with open(output_file, "w",  encoding='utf-8') as file:
      file.write(resp.text)

    # load text doc from URL w/ TextLoader
    loader = TextLoader('./'+output_file)
    txt_file_as_loaded_docs = loader.load()
    return txt_file_as_loaded_docs

def splitDoc(loaded_docs):
    # split docs into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_docs = splitter.split_documents(loaded_docs)
    return chunked_docs

def makeEmbeddings(chunked_docs):
    # Create embeddings and store them in a Weaviate vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = Weaviate.from_documents(chunked_docs, embedder)
    return vector_store

def askQs(vector_store, chain, q):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(q)
    resp = chain.run(input_documents=similar_docs, question=q)
    return resp

def loadLLM():
    llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


app = Flask(__name__)

@app.route("/", methods=['GET'])
def force_vector_store():
  global LOCAL_vector_store
  LOCAL_ldocs = loadFileFromURL('https://raw.githubusercontent.com/CarinaWeb/faq-content/main/about-carina.txt')
  LOCAL_cdocs = splitDoc(LOCAL_ldocs) #chunked
  LOCAL_vector_store = makeEmbeddings(LOCAL_cdocs)
  return "<h1>Twilio SMS + Langchain + Weaviate healthcheck</h1>"

@app.route('/sms', methods=['GET', 'POST'])
def sms():
    resp = MessagingResponse()
    inb_msg = request.form['Body'].lower().strip() #get inbound text body
    print(inb_msg)
    chain = loadLLM()
    global LOCAL_vector_store
    if LOCAL_vector_store == None:
      print("vector store is none")
      LOCAL_ldocs = loadFileFromURL('https://raw.githubusercontent.com/CarinaWeb/faq-content/main/about-carina.txt')
      LOCAL_cdocs = splitDoc(LOCAL_ldocs) #chunked
      LOCAL_vector_store = makeEmbeddings(LOCAL_cdocs)
    LOCAL_resp = askQs(LOCAL_vector_store, chain, inb_msg)
    print(LOCAL_resp)
    resp.message(LOCAL_resp)
    return str(resp)

if __name__ == "__main__":

  app.run(host='0.0.0.0', port=81)
