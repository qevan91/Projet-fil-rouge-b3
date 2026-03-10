# WeatherForYnov - Projet Fil Rouge B3

## Présentation du Projet
**WeatherForYnov** est un outil d'analyse et de prévision météorologique conçu pour accompagner les collectivités locales dans leur gestion quotidienne, exemple :
Prévoir des points de collectes d'eau et faire de la prévention si il fait trop chaud
Faire des activités extérieurs cas de beau temps.
Ect...

L'objectif est de permettre aux **communes** de mieux s'organiser face aux **aléas climatiques** (prévention des inondations, gestion du déneigement, optimisation de l'arrosage des espaces verts) grâce à l'exploitation intelligente des données météo.

## Road Map du Projet

### Phase 1 : Cadrage & Choix du Sujet
* **Sélection du projet :** Débat autour du sujet le plus complet.
* **Décision :** Sujet "Prévisions Météorologiques" car les données sont accessibles et permettent un gain de temps immédiat sur la phase de collecte.

### Phase 2 : Ciblage Utilisateur
* **Public cible :** Les communes.
* **Objectif :** Aide à la décision et organisation logistique en cas d'intempéries ou de changements climatiques soudains.

### Phase 3 : Planification
* Élaboration de la RoadMap technique (Ce que l'on fait, pourquoi et comment).

### Phase 4 : Data Engineering
* **Source :** Récupération des données via l'API **Open-Météo**.
* **Nettoyage :** Traitement des données brutes, gestion des valeurs manquantes.

### Phase 5 : Analyse Exploratoire
* **Visualisation :** Compréhension de la structure des données (température, humidité, nuage, précipitations).

### Phase 6 : Feature Engineering
* Réflexion sur la constitution du modèle.
* Analyse des tendances saisonnières et des corrélations entre les variables météo.

### Phase 7 : Modélisation & Machine Learning
* Mise en place du modèle prédictif.
* Phases de tests et premières prédictions.

### Phase 8 : Déploiement & Restitution
* Création d'un dashboard de visualisation.
* Rédaction du rapport final et présentation des résultats.

## Stack Technique
* **Langage :** Python, SQL 
* **Librairies :** Pandas, NumPy, Matplotlib/Seaborn, Scikit-Learn, sqlite3, request_cache, retry_request
* **API :** Open-Meteo API

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Comment lancer le projet ?

1.  **Cloner le repository :**
    ```bash
    git clone https://github.com/Wuelass/Projet-fil-rouge-b3
    ```
2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Lancer l'analyse :**
    ```bash
    python main.ipynb
    ```


