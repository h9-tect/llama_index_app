import streamlit as st
from settings import initialize_settings, load_documents_from_file, setup_node_parser, setup_query_engines
from parsers import MarkdownElementNodeParser

st.title("Llama Index Query Engine")

# Input for Llama Index API key
api_key = st.text_input("Enter your Llama Index API key:", type="password")

# File uploader for PDF documents
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if api_key and uploaded_file:
    st.success("API key and file uploaded successfully!")
    
    initialize_settings(api_key)
    
    # Load documents from uploaded file
    documents = load_documents_from_file(api_key, uploaded_file)
    st.write("Document Loaded: ", documents[0].text[:1000])

    node_parser = MarkdownElementNodeParser(llm=None, num_workers=8)
    nodes, base_nodes, objects, raw_index, index_with_obj = setup_node_parser(node_parser, documents)

    raw_query_engine, index_with_obj_query_engine, recursive_query_engine = setup_query_engines(nodes, base_nodes, objects, raw_index, index_with_obj)

    st.write("Query Engines Set Up Successfully!")
    
    # Example queries
    query = st.text_input("Enter your query:")
    if query:
        response_1 = raw_query_engine.query(query)
        st.write("Raw Query Engine Response:", response_1)
        
        response_2 = index_with_obj_query_engine.query(query)
        st.write("Index with Tables Query Engine Response:", response_2)
        
        response_3 = recursive_query_engine.query(query)
        st.write("Recursive Retriever with Tables Query Engine Response:", response_3)
