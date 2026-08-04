#Import the required Libraries
import os
import numpy as np
from openai import OpenAI
import faiss
from dotenv import load_dotenv
import PyPDF2
import json
import streamlit as s
import nltk
nltk.download('punkt')

gradient_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #4A90E2 0%, #9013FE 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
}
</style>
"""

s.markdown(gradient_bg, unsafe_allow_html=True)

s.title("COPAC - Policy & Claims Co-Pilot")
s.subheader("AI Assistant that gives accurate policy answers and pre checks claim before submission")
query = s.text_input("Ask something:")

#Function to read data from the input PDF file and save it to text
def load_pdf_to_text(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

pdf_text = load_pdf_to_text("Policy & Claims Copilot – Knowledge Base.pdf")
#print(pdf_text[:500])

#Create OPEN-AI Client
load_dotenv(dotenv_path=r'C:\Users\User\Agentic AI\.env')
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key = openai_api_key,
                       project="proj_AiRCDdpGwAmt6xebs4kjwD8f" )


#Define fucntion to call a open-api embeddings model to perform word embeddings
def create_embeddings(text):
    response = openai_client.embeddings.create(
        model = "text-embedding-3-small",
        input = text
    )
    return np.array(response.data[0].embedding, dtype="float32")

#Function to chunk the text usint NLTK punkt, here we are creating chucks of 500 characters
def chunk_by_sent(text, max_chars=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())

    return chunks


#Chunk text nefore proceeding to Embedding in order to give better context to RAG
chunks = chunk_by_sent(pdf_text)
#print(chunks[:3])

#Create word embedding and save them in faiss index
# dimension of embedding
d = 1536
# create index
index = faiss.IndexFlatL2(d)
# metadata store (FAISS cannot store text)
metadata = []

#create embedding and save in the Faiss Index
#Create word embeddings and save index & text in faiss index & metadata
for i, t in enumerate(chunks):
    vec = create_embeddings(t)
    index.add(vec.reshape(1, -1))# FAISS expects shape (1, d)
    metadata.append({"id": i, "text": t})


#Write data to faiss index
faiss.write_index(index, "policy_index.faiss")
#To save the text into the metadata.json file
with open("metadata.json", "w") as f:
    json.dump(metadata, f)


#Load the index & metadata into memory to be available for LLM/RAG
index = faiss.read_index("policy_index.faiss")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

#Function to embed the user query, retrieve the information from the faiss & share it as context to the LLM and run the LLM 
def ask_policy_copilot(query):
    #query = "How do i raise a claim?"
    query_vector = create_embeddings(query).reshape(1, -1)

    k = 2  # number of results
    distances, indices = index.search(query_vector, k)
    retrieved_data = []

    for i in indices[0]:
        retrieved_data.append(metadata[i]["text"])

    #Saving the FAISS response as Context to the LLM
    retrieved_Response = "\n\n".join(retrieved_data)

    prompt = f"""
    You're are COPAC:Policy Claim Co-pilot. Use ONLY the context below to answer the question.

    Context:
    {retrieved_Response}

    Question:
    {query}

    Answer:
    """
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content


if query:
    #query = input("Am COPAC: Policy & Claims Co-pilot. Please enter your query")
    copilt_response =ask_policy_copilot(query)
    #print(f"\n Your Query: {query}\n Copilot_response:\n {copilt_response}")
    s.write("### COPAC Response")
    s.write(copilt_response)
    #if query.lower() == "quit":
