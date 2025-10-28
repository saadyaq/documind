# 🧠 DocuMind - Intelligent RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red)](https://streamlit.io/)

> Système RAG (Retrieval-Augmented Generation) avec recherche sémantique FAISS et interface Streamlit

**🚀 Statut: MVP déployé - Semantic Search opérationnel**

---

## 📋 Table des matières

- [À propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Demo](#demo)
- [Installation](#installation)
- [Docker](#docker)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Roadmap](#roadmap)

---

## 🎯 À propos

DocuMind est un système de recherche sémantique intelligent basé sur l'architecture RAG. Le système utilise:

- **Recherche sémantique** via embeddings et FAISS pour trouver les documents pertinents
- **Interface moderne** avec Streamlit pour une expérience utilisateur fluide
- **Score de similarité** pour évaluer la pertinence des résultats
- **123 documents indexés** sur le deep learning et l'IA

### Problème résolu

Trouver rapidement des informations précises dans une large base documentaire nécessite une recherche sémantique intelligente. DocuMind indexe vos documents et retourne les passages les plus pertinents en quelques millisecondes.

### Cas d'usage

- Recherche dans documentation technique
- Base de connaissances interne
- Support FAQ intelligent
- Assistant de recherche académique

---

## ✨ Fonctionnalités

### ✅ Opérationnel

- [x] Génération d'embeddings sémantiques (Google EmbeddingGemma-300M)
- [x] Indexation vectorielle rapide (FAISS)
- [x] Recherche par similarité cosinus
- [x] Interface web Streamlit moderne
- [x] 123 documents indexés (deep learning)
- [x] Déploiement Docker
- [x] Métriques en temps réel
- [x] Queries pré-remplies

### 🚧 En développement

- [ ] Fine-tuning LLM avec LoRA
- [ ] Génération de réponses avec citations
- [ ] Pipeline RAG complet (retrieval + generation)
- [ ] Multi-turn conversations
- [ ] API REST

### 🎯 Prochainement

- [ ] Re-ranking des documents
- [ ] Support multi-langues
- [ ] Export des résultats
- [ ] Historique des recherches

---

## 🎥 Demo

**Interface de recherche sémantique:**

L'interface permet de:
- Poser des questions en langage naturel
- Voir les documents les plus pertinents avec scores
- Ajuster le nombre de résultats (1-12)
- Utiliser des queries pré-remplies pour démarrer

**Métriques affichées:**
- 123 documents indexés
- Dimension des vecteurs: 768
- Moteur: FAISS avec similarité cosinus

---

## 🚀 Installation

### Prérequis

- Python 3.11+
- 8 GB RAM minimum
- 2 GB espace disque
- Docker (optionnel)

### Installation locale

```bash
# Clone le repository
git clone https://github.com/username/documind.git
cd documind

# Crée environnement virtuel
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Installe dépendances
pip install -r requirements.txt

# Lance l'application
streamlit run app.py
```

Ouvre ton navigateur sur `http://localhost:8501`

### Configuration

Crée un fichier `.env` (optionnel):

```bash
HF_TOKEN=your_huggingface_token
EMBEDDING_MODEL=google/embeddinggemma-300m
TOP_K_DOCUMENTS=5
```

**Note:** Le token HuggingFace est requis pour le modèle `google/embeddinggemma-300m` (modèle gated). Obtiens-le sur https://huggingface.co/settings/tokens

---

## 🐳 Docker

### Build et Run

```bash
# Build l'image
docker build -t documind .

# Run avec token HuggingFace
docker run -d -p 8501:8501 -e HF_TOKEN=your_token --name documind-app documind

# Vérifie les logs
docker logs -f documind-app
```

Accède à l'app sur `http://localhost:8501`

### Commandes utiles

```bash
# Arrêter
docker stop documind-app

# Redémarrer
docker start documind-app

# Supprimer
docker rm -f documind-app

# Rebuild complet
docker build --no-cache -t documind .
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DOCUMIND SEARCH SYSTEM                    │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Query      │
│  utilisateur │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  EMBEDDING                                                │
│                                                           │
│  1. Encode la query                                      │
│     └─> Google EmbeddingGemma-300M                       │
│     └─> Vector 768 dimensions                            │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  RETRIEVAL                                                │
│                                                           │
│  2. Recherche FAISS                                      │
│     └─> Similarité cosinus                               │
│     └─> Top-K documents (K=5 par défaut)                │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  RESULTS                                                  │
│                                                           │
│  3. Affichage résultats                                  │
│     ├─ Rank                                              │
│     ├─ Topic                                             │
│     ├─ Score de similarité                               │
│     ├─ Distance                                          │
│     └─ Contenu du document                               │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Structure du projet

```
documind/
├── app.py                          # Interface Streamlit
├── Dockerfile                      # Container Docker
├── requirements.txt                # Dépendances Python
├── .gitignore
├── README.md
│
├── data/
│   └── train/
│       ├── documents.json          # Documents sources
│       └── embeddings/
│           ├── embeddings.npy      # Vecteurs (123 x 768)
│           ├── documents_with_embedding.json
│           └── faiss_index/
│               └── faiss_index.bin # Index FAISS
│
├── src/
│   ├── create_embeddings.py        # Génération embeddings
│   ├── retrieval.py                # Classe FaissIndex
│   ├── generate.py                 # Génération Q&A (Claude)
│   └── test_rag_system.py          # Tests RAG
│
├── notebooks/
│   └── fine_tuning.ipynb           # Fine-tuning LoRA
│
└── tests/
    ├── test_create_embeddings.py
    ├── test_retrieval.py
    └── test_data_preparation.py
```

---

## 🔧 Technologies

### ML/AI Stack

| Technologie | Usage | Version |
|------------|-------|---------|
| **Sentence Transformers** | Embeddings sémantiques | 2.3.1 |
| **FAISS** | Recherche vectorielle | 1.8.0+ |
| **Transformers** | Framework Hugging Face | 4.45.0+ |
| **PyTorch** | Deep learning backend | 2.0.0+ |

### Infrastructure

| Technologie | Usage |
|------------|-------|
| **Streamlit** | Interface web |
| **Docker** | Containerisation |
| **NumPy** | Calcul vectoriel |
| **Python-dotenv** | Variables d'environnement |

### Modèles

- **Embeddings:** `google/embeddinggemma-300m` (768 dimensions)
- **Futurs LLM:** `TinyLlama-1.1B-Chat` avec LoRA

---

## 📊 Performances

**Système actuel:**

| Métrique | Valeur |
|----------|--------|
| Documents indexés | 123 |
| Dimension vecteurs | 768 |
| Temps recherche | < 100ms |
| Précision retrieval | Score visible par résultat |

**Objectifs futurs (avec generation):**

| Métrique | Cible |
|----------|-------|
| Précision réponses | > 85% |
| Recall@3 retrieval | > 90% |
| Temps réponse total | < 2s |
| Citations correctes | 100% |

---

## 🗓️ Roadmap

### ✅ Phase 1: Semantic Search (TERMINÉ)
- [x] Setup projet
- [x] Génération embeddings
- [x] Index FAISS
- [x] Interface Streamlit
- [x] Docker deployment
- [x] 123 documents indexés

### 🚧 Phase 2: Generation (EN COURS)
- [ ] Dataset Q&A (Claude API)
- [ ] Fine-tuning LoRA TinyLlama
- [ ] Pipeline RAG complet
- [ ] Citations automatiques
- [ ] Évaluation metrics

### 🎯 Phase 3: Production (À VENIR)
- [ ] Déploiement cloud (Streamlit Cloud / HF Spaces)
- [ ] API REST FastAPI
- [ ] Monitoring et logs
- [ ] Tests end-to-end
- [ ] Documentation complète

### 💡 Phase 4: Améliorations
- [ ] Multi-modal RAG (texte + images)
- [ ] Re-ranking avancé
- [ ] GraphRAG
- [ ] Support multi-langues
- [ ] Base vectorielle cloud (Pinecone)

---

## 🧪 Développement

### Ajouter des documents

```bash
# Place tes documents dans data/raw/
# Formats supportés: .txt, .json, .csv

# Génère les embeddings
python src/create_embeddings.py
```

### Regénérer l'index FAISS

```bash
python src/retrieval.py
```

### Tester le système

```bash
# Tests unitaires
pytest tests/

# Test retrieval spécifique
pytest tests/test_retrieval.py -v
```

---

## 🤝 Contribution

Les contributions sont bienvenues!

1. Fork le projet
2. Crée ta branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvre une Pull Request

---

## 📝 License

MIT License - voir [LICENSE](LICENSE)

---

## 👤 Auteur

**Votre Nom**

- GitHub: [@username](https://github.com/username)
- LinkedIn: [profil](https://linkedin.com/in/username)
- Portfolio: [site.com](https://site.com)

---

## 🙏 Remerciements

- [Hugging Face](https://huggingface.co/) pour l'écosystème ML
- [FAISS](https://faiss.ai/) par Meta Research
- [Streamlit](https://streamlit.io/) pour l'interface rapide
- Communauté open-source ML

---

## 📚 Ressources

### Papers
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

### Guides
- [FAISS Documentation](https://faiss.ai/index.html)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

**⭐ Si ce projet t'aide, donne-lui une étoile!**

---

*Dernière mise à jour: Octobre 2025*
*Version: 0.2.0 (Semantic Search MVP)*