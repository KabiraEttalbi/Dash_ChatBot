Voici une version améliorée et corrigée de ton `README.md`, avec une meilleure structure, une orthographe soignée, et une formulation plus fluide tout en conservant l’ensemble des fonctionnalités et instructions :

---

# 🔬 arXiv Scientific Papers Chatbot

Ce projet propose une interface chatbot intelligente pour explorer des articles scientifiques issus de **arXiv** grâce à une recherche sémantique avancée. Il comprend un pipeline de données robuste pour l’extraction, le nettoyage et l’indexation des données, ainsi que deux applications web interactives développées avec **Streamlit** et **Dash**.

---

## ✨ Fonctionnalités

* **🔍 Recherche Sémantique**
  Utilise les modèles de **Sentence Transformers** pour générer des embeddings sémantiques et **FAISS** pour une recherche rapide et précise sur les titres et résumés des articles arXiv.

* **🤖 Réponses Générées par l'IA**
  Fournit des réponses complètes de type ChatGPT basées sur les résultats de recherche les plus pertinents : synthèse des résultats, méthodes, applications et tendances actuelles.

* **⚙️ Pipeline de Données Automatisé**

  * **Extraction** : Récupère les métadonnées des articles via l'API arXiv.
  * **Nettoyage** : Normalisation avancée des textes, extraction de mots-clés, validation de la qualité.
  * **Indexation** : Génération d'embeddings vectoriels et création d’un index **FAISS** pour les recherches sémantiques.

* **🖥️ Interfaces Utilisateur Interactives**

  * **Streamlit (`app/chatbot.py`)** : Interface simple et rapide pour discuter avec l’assistant et consulter les résultats.
  * **Dash (`app/dash_app.py`)** : Tableau de bord complet incluant chat, exploration des articles, visualisations interactives et statistiques globales.

* **📈 Statistiques Complètes**
  Donne un aperçu des tendances de publication, des catégories dominantes, des auteurs les plus actifs et des mots-clés fréquents.

---

## 🚀 Mise en Route

### 🔧 Prérequis

* **Python 3.8+** : [Installer Python](https://www.python.org/downloads/)
* **pip** : Le gestionnaire de paquets Python (généralement installé avec Python)

---

### 📦 Installation

1. **Cloner le dépôt ou extraire le fichier ZIP :**

```bash
git clone https://github.com/KabiraEttalbi/Dash_ChatBot  
cd arxiv-chatbot
```

2. **Créer et activer un environnement virtuel :**

```bash
python -m venv venv

# Sur macOS / Linux :
source venv/bin/activate

# Sur Windows :
venv\Scripts\activate
```

3. **Installer les dépendances du projet :**

```bash
pip install streamlit dash pandas scikit-learn faiss-cpu sentence-transformers plotly requests python-dotenv
```

*💡 Astuce : Utilisez `faiss-gpu` si vous avez une carte NVIDIA compatible CUDA pour des performances accrues.*

---

## 🛠️ Pipeline de Préparation des Données

Le chatbot repose sur un ensemble de données arXiv prétraitées et indexées. Le pipeline se compose de **trois scripts à exécuter dans l’ordre**. 

### 1. 📥 Extraction des Données

```bash
python data/scripts/arxiv_extractor.py 
```

* Récupère les métadonnées des articles depuis l’API arXiv.
* Sauvegarde dans : `data/data_source/raw_data.json` et `raw_data.csv`.
* ⚠️ Ce processus peut prendre plusieurs heures (limites d’API : 1 requête / 3 secondes).

---

### 2. 🧹 Nettoyage et Traitement

```bash
python data/scripts/data_cleaner.py 
```

* Supprime les caractères spéciaux, commandes LaTeX, espaces inutiles.
* Extrait et standardise les mots-clés, auteurs, catégories.
* Filtre les articles incomplets ou de faible qualité.
* Sauvegarde les résultats dans : `data/processed/clean_data.json` et `cleaning_statistics.json`.

---

### 3. 📦 Indexation Sémantique

```bash
python data/scripts/semantic_indexer.py 
```

* Génère des embeddings avec le modèle `all-MiniLM-L6-v2`.
* Construit un index **FAISS** pour la recherche rapide.
* Sauvegarde dans : `data/search_index/faiss_index.bin`, `metadata.json`, `papers_data.json`.

---

## 💻 Lancer les Applications Web

### 1. Application Streamlit (chatbot simple)

```bash
streamlit run app/chatbot.py
```

➡️ Ouvre automatiquement dans votre navigateur : `http://localhost:8501`

---

### 2. Application Dash (tableau de bord complet)

```bash
python app/dash_app.py
```

➡️ Accès via : `http://127.0.0.1:8050`

---

## ⚙️ Exécution Avancée (Chemins Manuels)

### Exemple : Extraction personnalisée

```bash
python data/scripts/arxiv_extractor.py --query "quantum computing" --max_results 500 --output data/data_source/custom.json json
```

### Nettoyage personnalisé

```bash
python data/scripts/data_cleaner.py --input data/data_source/custom.json --output data/processed/custom_clean.json 
```

### Indexation personnalisée

```bash
python data/scripts/semantic_indexer.py --input data/processed/custom_clean.json --output data/search_index/custom_index --model all-MiniLM-L6-v2 
```

---

## 💬 Utilisation

* **Interface de Chat** : Posez des questions de recherche. L’IA répond avec une synthèse enrichie et liste les articles associés.
* **Recherches Rapides** : Boutons intégrés pour explorer des sujets populaires.
* **Navigation (Dash)** : Accédez aux différentes vues : Assistant, Articles, Visualisations, Statistiques.

---

## 🧰 Technologies Utilisées

| Composant             | Utilisation                 |
| --------------------- | --------------------------- |
| Python                | Langage principal           |
| Streamlit, Dash       | Interfaces interactives     |
| Sentence Transformers | Embeddings sémantiques      |
| FAISS                 | Recherche vectorielle       |
| Pandas, Scikit-learn  | Traitement de données       |
| Plotly                | Visualisations              |
| Requests, `xml.etree` | Requêtes API et parsing XML |
| python-dotenv         | Variables d’environnement   |

---

## 🤝 Contribuer

Les contributions sont les bienvenues !
Pour proposer une amélioration :

1. Forkez le dépôt
2. Créez une branche (`git checkout -b feature/MaFonctionnalite`)
3. Apportez vos modifications
4. Faites un commit (`git commit -m "Ajout de la fonctionnalité"`)
5. Poussez la branche (`git push origin feature/MaFonctionnalite`)
6. Ouvrez une **Pull Request**

---

## 📄 Licence

Ce projet est open-source, distribué sous la licence [MIT](LICENSE).

---
