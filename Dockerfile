FROM python:3.11-slim

WORKDIR /documind

# Copie les fichiers de code
COPY requirements.txt .
COPY app.py .
COPY src/ ./src/

# IMPORTANT: Copie les données (même si dans .gitignore)
COPY data/ ./data/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
