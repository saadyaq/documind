# 🧠 DocuMind - Intelligent RAG Chatbot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)

> Retrieval-Augmented Generation (RAG) chatbot avec fine-tuning LLM et recherche sémantique

**🚧 Projet en développement actif - Semaine du [Date]**

---

## 📋 Table des matières

- [À propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Développement](#développement)
- [Technologies](#technologies)
- [Roadmap](#roadmap)
- [Auteur](#auteur)

---

## 🎯 À propos

DocuMind est un système de question-réponse intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation). Le système combine:

- **Retrieval sémantique** via embeddings et FAISS pour trouver les documents pertinents
- **Génération de réponses** avec un LLM fine-tuné via LoRA
- **Citations automatiques** des sources pour garantir la traçabilité
- **Interface web intuitive** avec Gradio

### Problème résolu

[Domaine spécifique] nécessite souvent de rechercher dans de nombreux documents pour répondre à des questions. DocuMind automatise ce processus en fournissant des réponses précises, contextualisées et sourcées en quelques secondes.

### Cas d'usage

- Réponse automatique aux FAQ
- Assistance à la documentation technique
- Support client intelligent
- Recherche sémantique dans bases de connaissances

---

## ✨ Fonctionnalités

### 🎯 Core Features

- [x] Génération d'embeddings sémantiques (Sentence Transformers)
- [x] Indexation vectorielle rapide (FAISS)
- [x] Recherche par similarité cosinus
- [ ] Fine-tuning LLM avec LoRA
- [ ] Pipeline RAG end-to-end
- [ ] Interface chat Gradio
- [ ] Citations automatiques des sources

### 🚀 Features Avancées (Prévues)

- [ ] Multi-turn conversations avec contexte
- [ ] Re-ranking des documents récupérés
- [ ] Détection d'hallucinations
- [ ] Support multi-langues
- [ ] API REST pour intégration

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     DOCUMIND RAG SYSTEM                     │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Question   │
│  utilisateur │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  RETRIEVAL PHASE                                          │
│                                                           │
│  1. Embedding de la question                             │
│     └─> Sentence Transformer (all-MiniLM-L6-v2)         │
│                                                           │
│  2. Recherche similarité FAISS                           │
│     └─> Top-K documents pertinents (K=3)                 │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  GENERATION PHASE                                         │
│                                                           │
│  3. Construction du prompt                               │
│     └─> Context: [documents] + Question                  │
│                                                           │
│  4. Génération réponse                                   │
│     └─> LLM fine-tuné (Phi-3-mini + LoRA)               │
│                                                           │
│  5. Post-processing                                      │
│     └─> Ajout citations + vérifications                  │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  RÉPONSE FINALE                                          │
│  ├─ Réponse générée                                      │
│  ├─ Sources citées                                       │
│  └─ Score de confiance                                   │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- 8 GB RAM minimum
- 10 GB espace disque disponible
- [UV](https://astral.sh/uv) (recommandé) ou pip

### Installation rapide avec UV
```bash
# 1. Clone le repository
git clone https://github.com/[username]/documind-rag-chatbot.git
cd documind-rag-chatbot

# 2. Installe UV (si pas déjà installé)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Crée l'environnement virtuel
uv venv

# 4. Active l'environnement
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 5. Installe les dépendances
uv pip install -r requirements.txt

# 6. Vérifie l'installation
python scripts/test_installation.py
```

### Installation alternative avec pip
```bash
# Clone et navigue
git clone https://github.com/[username]/documind-rag-chatbot.git
cd documind-rag-chatbot

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Installe dépendances
pip install -r requirements.txt
```

### Configuration
```bash
# Copie le template de configuration
cp .env.example .env

# Édite .env avec tes tokens
nano .env
```

**.env:**
```bash
# Hugging Face Token (optionnel pour API)
HF_TOKEN=your_token_here

# Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
TOP_K_DOCUMENTS=3
```

---

## 💻 Utilisation

### Démarrage rapide (quand terminé)
```bash
# Active l'environnement
source .venv/bin/activate

# Lance l'interface Gradio
python app.py
```

Ouvre ton navigateur sur `http://localhost:7860`

### Utilisation en ligne de commande
```python
from src.rag_pipeline import RAGPipeline

# Initialise le pipeline
rag = RAGPipeline()

# Pose une question
question = "Comment fonctionne X?"
answer, sources = rag.answer(question)

print(f"Réponse: {answer}")
print(f"Sources: {sources}")
```

### API (à venir)
```bash
# Démarre le serveur API
python api/server.py

# Requête exemple
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment faire Y?"}'
```

---

## 🛠️ Développement

### Structure du projet
```
documind-rag-chatbot/
├── data/
│   ├── raw/                    # Documents sources bruts
│   ├── processed/              # Embeddings et index FAISS
│   └── training/               # Dataset fine-tuning Q&A
├── models/
│   ├── embeddings/             # Modèles embeddings
│   └── fine_tuned/             # LLM fine-tuné (LoRA adapters)
├── src/
│   ├── embeddings.py           # Génération embeddings
│   ├── retrieval.py            # Système FAISS retrieval
│   ├── generation.py           # Génération LLM
│   ├── rag_pipeline.py         # Pipeline RAG complet
│   └── evaluation.py           # Métriques et évaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embeddings_test.ipynb
│   ├── 03_fine_tuning.ipynb
│   └── 04_evaluation.ipynb
├── tests/
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_pipeline.py
├── scripts/
│   └── test_installation.py
├── app.py                      # Interface Gradio
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Workflow de développement
```bash
# 1. Crée une branche pour ta feature
git checkout -b feature/nom-feature

# 2. Code et teste
python -m pytest tests/

# 3. Vérifie le style (optionnel)
black src/
flake8 src/

# 4. Commit et push
git add .
git commit -m "feat: description"
git push origin feature/nom-feature
```

### Ajouter des documents
```bash
# Place tes documents dans data/raw/
# Format supporté: .txt, .json, .csv

# Génère les embeddings
python src/embeddings.py --input data/raw/ --output data/processed/
```

### Fine-tuning du modèle
```bash
# Voir notebook: notebooks/03_fine_tuning.ipynb
# Ou utilise Google Colab avec GPU gratuit
```

---

## 🔧 Technologies

### Core ML/AI
- **[Transformers](https://huggingface.co/docs/transformers)** - Framework Hugging Face
- **[Sentence Transformers](https://www.sbert.net/)** - Embeddings sémantiques
- **[FAISS](https://faiss.ai/)** - Recherche vectorielle rapide
- **[PEFT](https://huggingface.co/docs/peft)** - Fine-tuning LoRA
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

### Interface & Deployment
- **[Gradio](https://gradio.app/)** - Interface web interactive
- **[Hugging Face Spaces](https://huggingface.co/spaces)** - Hébergement gratuit

### Modèles utilisés
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (80 MB)
- **LLM:** `microsoft/Phi-3-mini-4k-instruct` (7.6 GB)
- **Fine-tuning:** LoRA (rank=8, alpha=16)

---

## 📊 Performances (à venir)

| Métrique | Valeur |
|----------|--------|
| Précision réponses | TBD% |
| Recall@3 retrieval | TBD% |
| Temps réponse moyen | TBD s |
| Taux hallucination | < TBD% |
| Citations correctes | TBD% |

---

## 🗓️ Roadmap

### Semaine 1 (Actuel)
- [x] Setup projet et structure
- [x] Installation dépendances
- [ ] Système embeddings + FAISS
- [ ] Fine-tuning LLM avec LoRA
- [ ] Pipeline RAG complet
- [ ] Interface Gradio
- [ ] Déploiement Hugging Face Spaces

### Améliorations futures
- [ ] Multi-modal RAG (texte + images)
- [ ] GraphRAG pour données structurées
- [ ] Support multi-langues
- [ ] Re-ranking avancé
- [ ] Intégration bases de données vectorielles (Pinecone, Weaviate)
- [ ] Monitoring et analytics production

---

## 📈 Résultats (à compléter)

### Exemples de questions/réponses

**Question:** [Exemple question]

**Réponse:** [Réponse générée]

**Sources:** 
- Document 1 (score: 0.89)
- Document 2 (score: 0.85)

---

## 🤝 Contribution

Les contributions sont bienvenues! Pour contribuer:

1. Fork le projet
2. Crée ta branche (`git checkout -b feature/AmazingFeature`)
3. Commit tes changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvre une Pull Request

---

## 📝 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 👤 Auteur

**[Ton Nom]**

- GitHub: [@username](https://github.com/username)
- LinkedIn: [Ton profil](https://linkedin.com/in/username)
- Portfolio: [ton-site.com](https://ton-site.com)

---

## 🙏 Remerciements

- [Hugging Face](https://huggingface.co/) pour l'écosystème transformers
- [Sentence Transformers](https://www.sbert.net/) pour les embeddings
- [FAISS](https://faiss.ai/) par Facebook Research
- Communauté ML pour les ressources et tutoriels

---

## 📚 Ressources & Références

### Papers
- [RAG: Retrieval-Augmented Generation (2020)](https://arxiv.org/abs/2005.11401)
- [LoRA: Low-Rank Adaptation (2021)](https://arxiv.org/abs/2106.09685)
- [Dense Passage Retrieval (2020)](https://arxiv.org/abs/2004.04906)

### Tutoriels
- [Hugging Face RAG Course](https://huggingface.co/learn/cookbook)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/qa)

---

**⭐ Si ce projet t'aide, n'hésite pas à lui donner une étoile!**

---

*Dernière mise à jour: 21/10/2025*
*Version: 0.1.0 (Development)*