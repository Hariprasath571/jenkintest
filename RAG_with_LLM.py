# Import required libraries
# Here we used sentence_transformers and Facebook AI Similarity Search
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import streamlit as st
import os

# Step 1: Load the sentence transformer model for embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 

# Initialize an empty list for documents
documents = []

# Step 5: Define a function to read uploaded documents
def read_documents(uploaded_files):
    document_texts = []
    for uploaded_file in uploaded_files:
        # Read the uploaded file as a string
        text = uploaded_file.read().decode("utf-8")
        document_texts.append(text)
    return document_texts

# Step 6: Define a function to generate response using LLaMA model
def generate_response(prompt):
    llm = Ollama(model="llama3.2")  # Adjust the model name as necessary
    response = llm(prompt)  # Call the model with the prompt directly
    return response

# Step 7: Define a function to query FAISS
def query_faiss(query, k=1):
    # Generate embedding for the query
    query_embedding = embedder.encode([query], convert_to_tensor=False).astype('float32')
    
    # Search for the most similar documents
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the top-k documents
    retrieved_documents = [documents[idx] for idx in indices[0]]
    return retrieved_documents

# Streamlit application
def main():
    st.title("Question Answering with LLaMA and FAISS")

    # Step 2: File uploader in Streamlit
    uploaded_files = st.file_uploader("Upload text documents", type=["txt"], accept_multiple_files=True)

    if uploaded_files:
        # Read the uploaded documents
        global documents
        documents = read_documents(uploaded_files)

        # Step 3: Generate embeddings for the documents
        if documents:
            document_embeddings = embedder.encode(documents, convert_to_tensor=False)
            document_embeddings = np.array(document_embeddings).astype('float32')

            # Step 4: Initialize FAISS index and add embeddings
            global index
            index = faiss.IndexFlatL2(document_embeddings.shape[1])  # L2 distance
            index.add(document_embeddings)

        # Input box in Streamlit for user question
        question = st.text_input("Ask a question:")

        if question:
            # Step 8: Query FAISS to retrieve relevant documents
            retrieved_docs = query_faiss(question)
            # st.write("Retrieved Documents:")
            for doc in retrieved_docs:
                print(doc)
                # st.write("-", doc)
            
            # Step 9: Prepare the prompt for LLaMA
            context = "\n".join(retrieved_docs)
            prompt = f"Context: {context}\nQuestion: {question}"
            
            # Step 10: Generate response using LLaMA
            answer = generate_response(prompt)  
            
            # Display the answer in Streamlit
            st.subheader("Generated Response:")
            st.write(answer)

if __name__ == "__main__":
    main()
