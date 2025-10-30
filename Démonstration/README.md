# Démonstration : Application sur données réelles

## 🎯 Objectif

Cette partie a pour but de **mettre en évidence la perte d’information liée à la modélisation de l’incertitude** lorsque le modèle est contraint par une tâche de classification supervisée.

À travers des **jeux de données réels** comme **MNIST** ou **CIFAR-10**, nous comparons :
- un **réseau de classification** (ex. LeNet) entraîné sur des étiquettes explicites,  
- un **auto-encodeur** visant uniquement à reconstruire l’entrée sans supervision.

## 🧠 Intention

L’objectif est de **visualiser concrètement les différences d’incertitude** :
- Comportement des distributions de probabilité de sortie,  
- Qualité des représentations latentes,  
- Sensibilité aux exemples ambigus ou hors distribution.

Ces expériences servent de **démonstration intuitive** de l’hypothèse, avant toute validation statistique ou formelle.

## 🧰 Utilité

Cette partie fournit :
- Une **preuve qualitative** par observation visuelle,  
- Une base de **notebooks illustratifs** pour comprendre le phénomène,  
- Un point d’entrée pour tester et visualiser les effets de la contrainte de classification.

---
