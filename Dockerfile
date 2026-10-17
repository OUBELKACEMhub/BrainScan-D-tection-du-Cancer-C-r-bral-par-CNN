# Utiliser une image officielle Python légère
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Variables d'environnement globales (évite .pyc sur volume monté, logs non bufferisés)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Installer les dépendances système requises (notamment pour OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances en premier pour optimiser le cache Docker
COPY requirements.txt .

# Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du code source du projet
# NOTE: en développement, docker-compose.yml monte .:/app en bind mount
# et écrase ce COPY à l'exécution — ce COPY reste utile pour un build
# de production sans volume. Les modifications en dev sont donc
# automatiquement synchronisées via le volume.
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande par défaut (surchargée par docker-compose.yml pour activer le hot reload)
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.runOnSave=true", "--server.fileWatcherType=poll"]

