import streamlit as st
import sys
from pathlib import Path

# Add src to path to import retrieval module
sys.path.append(str(Path(__file__).parent / "src"))

from retrieval import FaissIndex


@st.cache_resource
def load_retriever():
    """Load the FAISS retriever (cached to avoid reloading)"""
    embedding_path = "data/train/embeddings/embeddings.npy"
    documents_path = "data/train/embeddings/documents_with_embedding.json"

    with st.spinner("Loading retrieval system..."):
        retriever = FaissIndex(embedding_path, documents_path)

    return retriever


def format_metadata(metadata):
    """Format metadata for display"""
    formatted = []
    if "topic" in metadata:
        formatted.append(f"**Topic:** {metadata['topic']}")
    if "section" in metadata:
        formatted.append(f"**Section:** {metadata['section']}")
    if "source_url" in metadata:
        formatted.append(f"**Source:** [{metadata['source_url']}]({metadata['source_url']})")
    if "word_count" in metadata:
        formatted.append(f"**Words:** {metadata['word_count']}")

    return " | ".join(formatted)


def get_score_color(score):
    """Get color based on similarity score"""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "orange"
    else:
        return "red"


def main():
    st.set_page_config(
        page_title="DocuMind - Semantic Search",
        page_icon="D",
        layout="wide"
    )

    # Header
    st.title("DocuMind - Semantic Document Search")
    st.markdown("Search through your documents using semantic similarity")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=5,
            help="How many similar documents to retrieve"
        )

        st.markdown("---")
        st.header("System Info")

        try:
            retriever = load_retriever()
            st.metric("Total Documents", len(retriever.documents))
            st.metric("Embedding Dimension", retriever.embeddings.shape[1])
            st.success("System Ready")
        except Exception as e:
            st.error(f"Error loading system: {e}")
            return

    # Main content
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Enter your search query:",
            height=100,
            placeholder="e.g., What is deep learning?",
            help="Enter a question or keywords to search"
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Search functionality
    if search_button and query.strip():
        with st.spinner("Searching..."):
            try:
                results = retriever.query(query, top_k=top_k)

                st.markdown("---")
                st.subheader(f"Top {len(results)} Results")
                

                for result in results:
                    rank = result['rank']
                    score = result['score']
                    distance = result['distance']
                    doc = result['document']

                    # Create expander for each result
                    with st.expander(
                        f"**#{rank}** | Score: {score:.3f} | {doc.get('metadata', {}).get('topic', 'Unknown Topic')}",
                        expanded=(rank == 1)
                    ):
                        # Score indicator
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Similarity", f"{score:.3f}")
                        with col_b:
                            st.metric("Distance", f"{distance:.3f}")

                        # Document text
                        st.markdown("#### Content")
                        st.write(doc['text'])

                        # Metadata
                        if 'metadata' in doc:
                            st.markdown("#### Metadata")
                            st.markdown(format_metadata(doc['metadata']))

                # Summary stats
                st.markdown("---")
                avg_score = sum(r['score'] for r in results) / len(results)
                st.info(f"Average similarity score: **{avg_score:.3f}**")

            except Exception as e:
                st.error(f"Error during search: {e}")

    elif search_button and not query.strip():
        st.warning("Please enter a search query")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with Streamlit | Powered by FAISS & Sentence Transformers
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
