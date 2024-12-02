##################### it is working with display pdf file ############
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import nltk
from nltk import sent_tokenize

# nltk.download('punkt')
history = []
chunks = []


# Create the Documents folder if it doesn't exist
documents_folder = os.path.join(os.getcwd(), 'Documents')
os.makedirs(documents_folder, exist_ok=True)

# Load the sentence transformer model for embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize session state for storing current PDF and page
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
    st.session_state.current_page = None


# Step 1: Function to handle PDF reading and return pages as chunks
def read_pdf(file):
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"content": text, "page": i + 1})
    return pages

# Step 2: Function to handle text, CSV, Excel, and PDF file reading
def read_documents(uploaded_files):
    document_texts = []
    doc_types = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Save the file locally
        file_path = os.path.join(documents_folder, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Read files based on type
        if file_extension == 'txt':
            with open(file_path, 'r', encoding="utf-8") as f:
                text = f.read()
            doc_types.append((text, uploaded_file.name, None))  # No page for .txt
        elif file_extension == 'pdf':
            pages = read_pdf(file_path)
            for page in pages:
                document_texts.append(page['content'])
                doc_types.append((page['content'], uploaded_file.name, page['page']))
        elif file_extension == 'csv':
            df = pd.read_csv(file_path)
            text = df.to_string()
            doc_types.append((text, uploaded_file.name, None))
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_path)
            text = df.to_string()
            doc_types.append((text, uploaded_file.name, None))

    return doc_types

# Step 3: Split documents into smaller chunks (sentences)
def split_documents_into_chunks(documents):
    chunks = []
    for document in documents:
        sentences = sent_tokenize(document[0])  # Document text
        for sentence in sentences:
            chunks.append({"sentence": sentence, "file_name": document[1], "page": document[2]})
    return chunks

# Step 4: Define a function to query FAISS
def query_faiss(query, k=4):
    query_embedding = embedder.encode([query], convert_to_tensor=False).astype('float32')
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [document_chunks[idx] for idx in indices[0]]
    return retrieved_chunks

# Step 5: Define a function to generate response using LLaMA
def generate_response(prompt):
    llm = Ollama(model="llama3.2")  # Adjust the model name as necessary
    response = llm(prompt)
    return response

# Step 6: Function to convert PDF to base64 for rendering
def convert_pdf_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def display_pdf(file_path, page_number): 
    st.sidebar.subheader(f"Displaying: {os.path.basename(file_path)} - Page {page_number}")
    print(file_path, page_number)
    pdf_display = f'<iframe src="data:application/pdf;base64,{convert_pdf_to_base64(file_path)}#page={page_number}" width="500" height="500" type="application/pdf"></iframe>'
    print("display pdf details",file_path, page_number)
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)


def update_history(question, response, retrieved_chunks):
    # Append the question, response, and retrieved chunks to the history
    history.append({
        "question": question,
        "response": response,
        "chunks": retrieved_chunks
    })
  

# Step 9: Function to display the history
def display_history():

    st.subheader("Interaction History")
    for entry in history:
        st.write(f"**Question:** {entry['question']}")
        st.write(f"**Response:** {entry['response']}")
        st.write("**Chunks Retrieved:**")
        for chunk in entry['chunks']:
            st.write(f" - {chunk['sentence']} (Page {chunk['page']})")

# Streamlit application
def main():
    st.title("Question Answering with LLaMA and FAISS")
    with st.sidebar:
        st.subheader("PDF Viewer")  # Show the sidebar
    
    # Initialize session state variables if not already set
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = None

    # File uploader
    uploaded_files = st.file_uploader("Upload text, PDF, CSV, or Excel files", type=["txt", "pdf", "csv", "xlsx"], accept_multiple_files=True)

    if uploaded_files:
        documents = read_documents(uploaded_files)
        global document_chunks
        document_chunks = split_documents_into_chunks(documents)

        # Generate embeddings for the chunks
        document_embeddings = embedder.encode([chunk['sentence'] for chunk in document_chunks], convert_to_tensor=False)
        document_embeddings = np.array(document_embeddings).astype('float32')

        # Initialize FAISS index and add embeddings
        global index
        index = faiss.IndexFlatL2(document_embeddings.shape[1])
        index.add(document_embeddings)

    # Input question
    question = st.text_input("Ask a question:")

    if question:
        retrieved_chunks = query_faiss(question)

        # Prepare the prompt for LLaMA 
        context = "\n".join([chunk['sentence'] for chunk in retrieved_chunks])
        prompt = f"Context: {context}\nQuestion: {question}"

        # Generate response using LLaMA
        answer = generate_response(prompt)
        update_history(question, answer, retrieved_chunks)

        # Display the generated response
        st.subheader("Generated Response:")
        st.write(answer)

        # Display retrieved chunks with clickable links
        st.subheader("Retrieved Chunks:")
        for chunk in retrieved_chunks:
            st.write(f"**Sentence:** {chunk['sentence']}")
            if chunk['page']:
                # Button to view the PDF at the specific page
                if st.button(f"View PDF (Page {chunk['page']})", key=f"{chunk['file_name']}_{chunk['page']}"):
                    pdf_path = os.path.join(documents_folder, chunk['file_name'])
                    # Update session state with the current PDF and page
                    st.session_state.current_pdf = pdf_path
                    st.session_state.current_page = chunk['page']
                    # Rerun the app to refresh the sidebar

                    st.rerun()

    # Check if a PDF and page are currently set in session state to display
    if st.session_state.current_pdf and st.session_state.current_page:
        # Ensure sidebar shows the latest PDF page
        display_pdf(st.session_state.current_pdf, st.session_state.current_page)  # Always display the current PDF page
    display_history()
    # print("chunks are", chunks)

if __name__ == "__main__":
    main()


