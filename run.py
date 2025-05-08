import streamlit as st
from main import load_resources, process_query

st.set_page_config(
    page_title="Shakespeare RAG System",
    page_icon="📚",
)

# Load resources
st.title("Shakespeare RAG System")
st.write("Enter a query to retrieve relevant passages and generate an answer.")

@st.cache_resource
def load_cached_resources():
    return load_resources()

index, metadata, embedder = load_cached_resources()

# Input query
query = st.text_input("Enter your query:", placeholder="e.g., Which sonnet has the line 'Shall I compare thee to a summer's day'?")

if query:
    st.write("Processing your query...")
    results, answer = process_query(query, index, metadata, embedder)

    # Display results
    st.subheader("Generated Answer:")
    st.write(answer)
    
    st.subheader("Top Retrieved Passages:")
    for idx, result in enumerate(results):
        st.write(f"**Passage #{idx + 1}:**")
        st.write(f"**Title:** {result['name']}")
        st.write(f"**Contents:** {result['contents']}")
        st.write("---")
