import streamlit as st
import os
from llama_index import GPTVectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from PyPDF2 import PdfReader

# Define the embedding function using LlamaIndex (OpenAI embedding for demonstration)
def embed_text(text):
    """
    Embed the input text using LlamaIndex embeddings.

    Args:
        text (str): The input text to be embedded.

    Returns:
        list: A list of embeddings for the input text.
    """
    embedding_model = OpenAIEmbedding()  # Using OpenAI embedding as a placeholder
    embeddings = embedding_model.embed(text)
    return embeddings

# Function to read and parse PDF file
def read_pdf(file):
    """
    Reads a PDF file and extracts the text content.

    Args:
        file (UploadedFile): The uploaded PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit app layout
def main():
    st.title("Simple RAG with LlamaIndex and Streamlit")
    st.write("Upload a file, extract text, and generate embeddings.")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = read_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type.")
            return

        st.subheader("Extracted Text")
        st.text_area("File Notes", text, height=300)

        # Generate embeddings
        embeddings = embed_text(text)
        
        st.subheader("Generated Embeddings")
        st.write(embeddings)

        # Simulate vector database entry (print embeddings and metadata)
        st.subheader("Metadata Preview")
        st.json({
            "text_snippet": text[:100],
            "embedding": embeddings[:5],  # Show only first 5 elements for brevity
            "length": len(embeddings)
        })

if __name__ == "__main__":
    main()