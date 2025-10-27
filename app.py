import html
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
    """Return metadata badges as HTML for display inside result cards."""
    if not metadata:
        return ""

    badge_map = [
        ("topic", "Topic"),
        ("section", "Section"),
        ("source_url", "Source"),
        ("word_count", "Words"),
    ]

    badges = []
    for key, label in badge_map:
        value = metadata.get(key)
        if value:
            safe_value = html.escape(str(value))
            badges.append(
                f"<span class='meta-badge'>{label}: {safe_value}</span>"
            )

    return "".join(badges)


def get_score_class(score):
    """Return CSS class name based on similarity score."""
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def inject_custom_styles():
    """Inject custom CSS to refresh the UI."""
    st.markdown(
        """
        <style>
            :root {
                --documind-primary: #7b8dff;
                --documind-dark: #f7f8ff;
                --documind-gray: rgba(197, 201, 255, 0.72);
                --documind-surface: rgba(27, 29, 49, 0.9);
                --documind-panel: rgba(22, 24, 42, 0.92);
                --documind-border: rgba(123, 136, 255, 0.32);
                --documind-highlight: rgba(123, 136, 255, 0.18);
                --documind-card-shadow: 0px 18px 52px rgba(0, 0, 0, 0.4);
                --documind-body: #0e101f;
            }

            [data-testid="stAppViewContainer"] {
                background: radial-gradient(120% 120% at 50% 0%, rgba(61, 66, 136, 0.25) 0%, rgba(14, 16, 31, 0.96) 42%, rgba(14, 16, 31, 1) 100%);
                color: var(--documind-dark);
            }

            [data-testid="stSidebar"] > div:first-child {
                background: rgba(13, 15, 28, 0.96);
                border-right: 1px solid rgba(69, 74, 131, 0.4);
                color: rgba(232, 235, 255, 0.86);
            }

            .hero {
                padding: 2.5rem 0 1.5rem;
                margin-bottom: 1rem;
                border-bottom: 1px solid var(--documind-border);
            }

            .hero__title {
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--documind-dark);
                margin-bottom: 0.5rem;
            }

            .hero__subtitle {
                font-size: 1.05rem;
                color: var(--documind-gray);
                max-width: 680px;
            }

            .sample-chip {
                display: flex;
                width: 100%;
            }

            .sample-chip .stButton {
                width: 100%;
            }

            .sample-chip button {
                background: rgba(123, 136, 255, 0.14) !important;
                color: var(--documind-primary);
                border-radius: 16px;
                border: 1px solid rgba(123, 136, 255, 0.35);
                width: 100%;
                font-size: 0.88rem;
                font-weight: 600;
                padding: 0.75rem 1rem;
                box-shadow: 0px 18px 36px rgba(5, 7, 26, 0.35);
                transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.2s ease;
            }

            .sample-chip button:hover {
                background: rgba(123, 136, 255, 0.24) !important;
                border-color: rgba(123, 136, 255, 0.55);
                box-shadow: 0px 24px 42px rgba(7, 10, 32, 0.5);
                transform: translateY(-1px);
            }

            .sample-chip button:focus {
                outline: 2px solid rgba(123, 136, 255, 0.45);
                outline-offset: 2px;
            }

            form[data-testid="stForm"] {
                background: var(--documind-panel);
                padding: 1.9rem;
                border-radius: 22px;
                border: 1px solid var(--documind-border);
                box-shadow: 0px 22px 48px rgba(4, 5, 18, 0.55);
                margin-bottom: 1.5rem;
            }

            form[data-testid="stForm"] label {
                font-weight: 600;
                color: var(--documind-dark);
            }

            form[data-testid="stForm"] textarea {
                background: rgba(12, 14, 26, 0.92);
                border-radius: 18px;
                border: 1px solid rgba(123, 136, 255, 0.32);
                padding: 1.1rem;
                color: rgba(236, 238, 255, 0.92);
                font-size: 1rem;
                box-shadow: inset 0 2px 10px rgba(6, 7, 18, 0.7);
            }

            form[data-testid="stForm"] textarea:focus {
                border-color: rgba(123, 136, 255, 0.65);
                box-shadow: 0px 0px 0px 4px rgba(123, 136, 255, 0.25);
            }

            form[data-testid="stForm"] textarea::placeholder {
                color: rgba(195, 199, 255, 0.58);
            }

            .main-surface {
                background: rgba(20, 22, 40, 0.9);
                border-radius: 28px;
                border: 1px solid rgba(95, 110, 192, 0.42);
                box-shadow: 0px 30px 70px rgba(3, 4, 16, 0.65);
                padding: 3rem 3.4rem;
                margin-top: -3.5rem;
                color: rgba(232, 235, 255, 0.92);
            }

            .main-surface * {
                color: inherit;
            }

            .main-surface a {
                color: var(--documind-primary);
            }

            @media (max-width: 1100px) {
                .main-surface {
                    padding: 2rem 1.5rem;
                    border-radius: 20px;
                }
            }

            .result-card {
                background: rgba(17, 19, 35, 0.92);
                border-radius: 18px;
                padding: 1.5rem;
                border: 1px solid rgba(123, 136, 255, 0.28);
                box-shadow: var(--documind-card-shadow);
                margin-bottom: 1rem;
            }

            .result-card__top {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .result-title {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .result-rank {
                font-weight: 600;
                color: var(--documind-primary);
                font-size: 1.1rem;
            }

            .result-topic {
                font-weight: 600;
                color: var(--documind-dark);
                font-size: 1.1rem;
            }

            .score-badge {
                font-weight: 600;
                padding: 0.35rem 0.85rem;
                border-radius: 999px;
                color: #ffffff;
                font-size: 0.85rem;
                letter-spacing: 0.02em;
            }

            .score-badge.high { background: #00b074; }
            .score-badge.medium { background: #f5a623; }
            .score-badge.low { background: #ef4f4f; }

            .result-card__content {
                color: rgba(230, 233, 255, 0.9);
                line-height: 1.55;
                margin-bottom: 1rem;
                font-size: 0.96rem;
                white-space: pre-wrap;
            }

            .result-card__meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }

            .meta-badge {
                background: rgba(123, 136, 255, 0.16);
                color: rgba(208, 212, 255, 0.92);
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.8rem;
                border: 1px solid rgba(123, 136, 255, 0.28);
            }

            .meta-badge.muted {
                color: rgba(197, 201, 255, 0.6);
                background: rgba(120, 126, 173, 0.18);
                border-color: rgba(120, 126, 173, 0.2);
            }

            div[data-testid="stMetricValue"] {
                color: var(--documind-dark);
                font-weight: 700;
            }

            div[data-testid="stMetricLabel"] {
                color: var(--documind-gray);
                font-weight: 500;
            }

            .callout {
                background: rgba(0, 176, 116, 0.16);
                border: 1px solid rgba(0, 176, 116, 0.32);
                color: rgba(185, 255, 224, 0.92);
                padding: 0.9rem 1.2rem;
                border-radius: 14px;
                margin-top: 1.5rem;
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(result):
    """Render a single retrieval result card with styling."""
    document = result.get("document", {})
    metadata = document.get("metadata") or {}

    rank = result.get("rank")
    topic = metadata.get("topic") or "Result"
    distance = result.get("distance")
    score = result.get("score", 0.0)

    score_class = get_score_class(score)
    score_display = f"{score:.3f}"

    ranked_label = f"#{rank}" if rank is not None else "#—"

    text_content = document.get("text") or ""
    safe_text = html.escape(text_content)
    formatted_text = safe_text.replace("\n", "<br>")

    metadata_html = format_metadata(metadata)
    if distance is not None:
        metadata_html += (
            f"<span class='meta-badge muted'>Distance: {distance:.3f}</span>"
        )

    card_html = f"""
        <div class="result-card">
            <div class="result-card__top">
                <div class="result-title">
                    <span class="result-rank">{ranked_label}</span>
                    <span class="result-topic">{html.escape(topic)}</span>
                </div>
                <span class="score-badge {score_class}">{score_display}</span>
            </div>
            <div class="result-card__content">{formatted_text}</div>
            <div class="result-card__meta">{metadata_html}</div>
        </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="DocuMind - Semantic Search",
        page_icon="D",
        layout="wide"
    )

    inject_custom_styles()

    try:
        retriever = load_retriever()
    except Exception as exc:
        st.error(f"Erreur lors du chargement du système : {exc}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Paramètres")
        top_k = st.slider(
            "Nombre de résultats",
            min_value=1,
            max_value=12,
            value=5,
            help="Nombre de passages sémantiquement proches à retourner."
        )

        st.markdown("---")
        st.header("Statut")
        st.metric("Documents indexés", len(retriever.documents))
        st.metric("Dimension des embeddings", retriever.embeddings.shape[1])
        st.metric("Vector Store", "FAISS")
        st.caption("Le retriever est mis en cache pour une expérience fluide.")

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    st.markdown("<div class='main-surface'>", unsafe_allow_html=True)

    # Hero section
    st.markdown(
        """
        <section class="hero">
            <div class="hero__title">DocuMind Semantic Explorer</div>
            <div class="hero__subtitle">
                Interrogez votre base documentaire grâce à la recherche sémantique.
                DocuMind retrouve les passages les plus pertinents et vous montre ce qui compte.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Quick stats
    stats_cols = st.columns(3)
    stats_cols[0].metric("Documents indexés", len(retriever.documents))
    stats_cols[1].metric("Dimension des vecteurs", retriever.embeddings.shape[1])
    stats_cols[2].metric("Moteur de similarité", "FAISS · cosine")

    st.markdown("")

    sample_queries = [
        "Quelles sont les étapes pour préparer les données ?",
        "Comment fonctionne le pipeline de récupération FAISS ?",
        "Quels paramètres utiliser pour le fine-tuning LoRA ?",
    ]

    st.caption("Essayez une requête pré-remplie pour démarrer :")
    for row_start in range(0, len(sample_queries), 3):
        chip_columns = st.columns(3)
        for offset, col in enumerate(chip_columns):
            sample_index = row_start + offset
            if sample_index >= len(sample_queries):
                col.empty()
                continue

            sample = sample_queries[sample_index]
            with col:
                st.markdown("<div class='sample-chip'>", unsafe_allow_html=True)
                if st.button(sample, key=f"sample_{sample_index}", use_container_width=True):
                    st.session_state.query_text = sample
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    with st.form("semantic_search"):
        st.text_area(
            "Posez votre question",
            key="query_text",
            height=160,
            placeholder="Exemple : Quels sont les avantages de l'indexation FAISS ?",
            help="Rédigez une question complète ou quelques mots clés.",
        )

        form_cols = st.columns([3, 1])
        with form_cols[0]:
            st.caption("Appuyez sur ⌘/Ctrl + Entrée pour lancer la recherche.")
        with form_cols[1]:
            submitted = st.form_submit_button(
                "Lancer la recherche",
                type="primary",
                use_container_width=True,
            )

    query = st.session_state.query_text.strip()

    if submitted:
        if not query:
            st.warning("Merci de saisir une question avant de lancer la recherche.")
        else:
            with st.spinner("Recherche sémantique en cours..."):
                try:
                    results = retriever.query(query, top_k=top_k)
                except Exception as exc:
                    st.error(f"Erreur pendant la recherche : {exc}")
                    st.stop()

            if not results:
                st.info("Aucun passage pertinent trouvé. Essayez d'élargir votre requête.")
            else:
                st.markdown(f"### Résultats ({len(results)})")
                for result in results:
                    render_result_card(result)

                average_score = sum(r["score"] for r in results) / len(results)
                st.markdown(
                    f"<div class='callout'>Score moyen de similarité : <strong>{average_score:.3f}</strong></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("")
    st.caption("Construit avec Streamlit · Propulsé par FAISS & Sentence Transformers")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
