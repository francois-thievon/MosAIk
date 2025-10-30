# MOSAIK — Comparaison de la qualité de modélisation de l’incertitude d’un auto-encodeur vs. un modèle de classification

## 🎯 Objectif du projet

Ce projet de recherche vise à **vérifier l’hypothèse de perte d’information concernant la modélisation de l’incertitude** lorsque l’on contraint un modèle à prédire une étiquette.  
Autrement dit, l’étude cherche à montrer que **l’apprentissage supervisé multi-classes conduit à une dégradation de la qualité de la modélisation de l’incertitude**, par rapport à un auto-encodeur non contraint par une tâche de classification.

Le travail s’inscrit dans le cadre du **projet MOSAIK (Modélisation et Quantification d’Incertitude)**, mené au sein de l’équipe **LORIA – équipe MosAIk**, et s’appuie sur les méthodes récentes de **quantification d’incertitude**.

---

## 🧠 Contexte scientifique

Un modèle de classification probabiliste classique ne capture qu’un seul type d’incertitude : la **probabilité d’appartenance à une classe**.  
Or, plusieurs natures d’incertitude coexistent :

- **Incertitude aléatoire (épistémique)** : liée à la difficulté intrinsèque du problème (ex. lancer de pièce).
- **Incertitude informationnelle (modélisation)** : liée à un manque de données ou à des observations rares (ex. apparition d’un O.V.N.I dans un problème Avion/Oiseau).

Plusieurs approches existent pour modéliser ces différentes incertitudes :
- **Méthodes bayésiennes** (ex. Dropout bayésien, BNN)
- **Méthodes ensemblistes** (ensembles de modèles, variance prédictive)
- **Minimisation de risque de second ordre**
- **Méthodes par densité locale**

Le projet explore cette problématique à travers trois approches complémentaires : démonstration, preuve empirique, et preuve formelle.

---

## 🗂️ Structure du dépôt

```bash
MOSAIK/
├── demonstration/              # Application sur données réelles (MNIST, CIFAR-10)
│   ├── notebooks/              # Notebooks de visualisation et d’analyse
│   ├── models/                 # Réseaux LeNet, auto-encodeur, etc.
│   ├── results/                # Figures et métriques générées
│   └── README.md               # Description de cette partie
│
├── preuve_empirique/           # Expérimentations sur données simulées
│   ├── simulations/            # Génération manuelle de jeux de données
│   ├── analyses/               # Études statistiques et convergence (loi des grands nombres)
│   ├── plots/                  # Visualisations des résultats empiriques
│   └── README.md
│
├── preuve_formelle/            # Partie théorique et démonstrations mathématiques
│   ├── pdf/                    # Documents LaTeX / notes de preuve
│   └── README.md
│
├── utils/                      # Fonctions communes : visualisation, métriques, incertitude
├── requirements.txt            # Dépendances Python
├── LICENSE
└── README.md                   # Ce fichier
```
---

## 🔬 Approches complémentaires

### 1. **Démonstration (Application sur données réelles)**

Cette première approche illustre la problématique sur des jeux de données standards :
- **MNIST** (chiffres manuscrits) pour l’expérimentation initiale
- **CIFAR-10** (images couleur) pour des tests plus complexes

L’objectif est de **visualiser les différences de comportement entre un modèle de classification et un auto-encodeur** en termes d’incertitude :
- Distribution des probabilités de sortie
- Mesure de confiance (entropy, mutual information, variance prédictive)
- Représentation des embeddings latents

---

### 2. **Preuve empirique (Validation statistique sur données simulées)**

Cette partie vise à **confirmer empiriquement** les observations faites sur les données réelles :
- Génération manuelle de jeux de données simples (2D, classes séparables ou ambiguës)
- Application répétée de modèles pour observer la **stabilité et la convergence** des mesures d’incertitude
- Étude de la **loi des grands nombres** appliquée aux estimations de confiance

L’objectif est de montrer que la perte d’information se vérifie de manière robuste sur de nombreux tirages indépendants.

---

### 3. **Preuve formelle (Démonstration mathématique)**

Enfin, une analyse théorique vient **établir formellement** les bases de l’hypothèse :
- Définition mathématique de l’incertitude (entropie, divergence de Kullback-Leibler, etc.)
- Analyse de la sur-spécification du modèle de classification
- Mise en évidence de la perte d’information par projection sur l’espace des classes

Cette section ne contient **aucun code**, uniquement des démonstrations et développements analytiques.

---

## 🧰 Technologies principales

- **Python 3.10+**
- **PyTorch / TensorFlow** pour les modèles
- **NumPy / SciPy / Pandas** pour le traitement des données
- **Matplotlib / Seaborn / Plotly** pour la visualisation
- **Jupyter Notebook** pour la documentation interactive

---

## 📈 Plan du projet

1. Entraîner un modèle **LeNet** sur MNIST.  
2. Transformer le réseau en **auto-encodeur**.  
3. Implémenter différentes **méthodes de quantification d’incertitude**.  
4. Définir un **critère de comparaison** robuste de la qualité de la modélisation d’incertitude.  
5. Évaluer et comparer les modèles.  
6. (Optionnel) Étendre l’étude à **CIFAR-10** et des architectures plus profondes.  

---

## 👥 Encadrement

Projet de recherche mené au sein de l’équipe **MosAIk – LORIA**, dans le cadre d’un **projet de fin d’études** en data science et intelligence artificielle.

---

## 📜 Licence

Ce projet est distribué sous licence MIT — voir le fichier [LICENSE](./LICENSE) pour plus d’informations.

---

