# Démonstration : Application sur données réelles

## 🎯 Objectif

Cette partie a pour but de **mettre en évidence la perte d'information liée à la modélisation de l'incertitude** lorsque le modèle est contraint par une tâche de classification supervisée.

À travers des **jeux de données réels** comme **MNIST** ou **CIFAR-10**, nous comparons :
- un **réseau de classification** (ex. LeNet) entraîné sur des étiquettes explicites,  
- un **auto-encodeur** visant uniquement à reconstruire l'entrée sans supervision.

## 🧠 Intention

L'objectif est de **visualiser concrètement les différences d'incertitude** :
- Comportement des distributions de probabilité de sortie,  
- Qualité des représentations latentes,  
- Sensibilité aux exemples ambigus ou hors distribution.

Ces expériences servent de **démonstration intuitive** de l'hypothèse, avant toute validation statistique ou formelle.

## 🧰 Utilité

Cette partie fournit :
- Une **preuve qualitative** par observation visuelle,  
- Une base de **notebooks illustratifs** pour comprendre le phénomène,  
- Un point d'entrée pour tester et visualiser les effets de la contrainte de classification.

---

# 📓 Notebook: `demo_mnist.ipynb`

## 📋 Vue d'ensemble

Le notebook `demo_mnist.ipynb` démontre une **approche complète de quantification d'incertitude** sur le dataset **MNIST**, en comparant deux architectures :
1. **Deep Ensemble de MLPs** (réseaux de classification supervisée)
2. **Deep Ensemble d'Autoencodeurs** (réseaux de reconstruction non-supervisée)

---

## 📚 Organisation générale du notebook

Le notebook suit cette structure :
Cellules 1-6 → Chargement et visualisation MNIST
Cellules 7-10 → Baseline MLP (784→256→128→10)
Cellules 11-13 → Single Autoencoder (784→256→128→10→128→256→784)
Cellules 14-15 → Chargement Fashion MNIST (données OOD)
Cellules 16-22 → Deep Ensemble MLP (K=10) + Uncertainties
Cellules 23-28 → Deep Ensemble Autoencodeurs (K=10) + Uncertainties
Cellule 29 → Comparaison MLP vs Autoencodeurs

---

## 🔍 Détails par section

### Section 1-4 : Données et Baseline

**MNIST Loading** - Chargement du dataset complet (60K train, 10K test) via fichiers IDX binaires

**MLP Baseline** - Réseau simple 784→256→128→10 pour référence de performance (~97% accuracy)

**Single Autoencoder** - Autoencodeur de démonstration avec bottleneck 10D pour vérifier la reconstruction

**Fashion MNIST** - 1,000 images de Fashion MNIST comme données hors-distribution (OOD test set)

---

### Section 5 : Deep Ensemble MLP - Quantification d'incertitude

#### Architecture
- **K=10 modèles MLP** entraînés avec bootstrap sampling
- Chaque modèle reçoit un échantillon aléatoire (60K avec remise) → ~38K uniques
- Seeds différentes pour diversité

#### Calcul d'incertitude

**Aleatoric Uncertainty** (incertitude des données)

$$u_a = H(\bar{p}(x)) = -\sum_{i=0}^{9} \bar{p}_i(x) \log(\bar{p}_i(x))$$

où $\bar{p}(x) = \frac{1}{K}\sum_{k=1}^{K} p_k(x)$

**Epistemic Uncertainty** (incertitude du modèle via Mutual Information)

$$u_e = MI = H(\bar{p}(x)) - \frac{1}{K}\sum_{k=1}^{K} H(p_k(x))$$

#### Détection OOD

**Seuil** : 75e percentile du ratio $u_e / u_a$ sur MNIST

**Décision** : 
- Si ratio > threshold → Classé comme **OOD (anomaly)**
- Sinon → Classé comme **ID (normal)**

**Résultats attendus** :
- **FPR** (MNIST) : ~25% (faux positifs acceptables)
- **TPR** (Fashion MNIST) : ~90%+ (excellente détection)

#### Visualisations

1. **KDE Plot Aleatoric** : Distributions MNIST vs Fashion MNIST (quasi-identiques = attendu)
2. **KDE Plot Epistemic** : Distributions bien séparées (MNIST << Fashion MNIST)
3. **2D Scatter** : Aleatoric vs Epistemic avec limite décisionnelle (diagonal)

---

### Section 6 : Deep Ensemble Autoencodeurs - Quantification d'incertitude

#### Différence clé

Les **MLPs** prédisent directement $p_k(x)$ ∈ ℝ^10 via softmax du dernier layer.

Les **Autoencodeurs** utilisent la **couche bottleneck** (10 neurones ReLU) comme base de prédiction :
- Forward pass jusqu'à bottleneck : $z_k(x)$ ∈ ℝ^10
- Conversion en probabilités : $p_k(x) = \text{softmax}(z_k(x))$
- Justification : Représentation **non-supervisée**, pas biaisée par les labels

#### Architecture
784 → [256 ReLU] → [128 ReLU] → [10 ReLU bottleneck] → [128 ReLU] → [256 ReLU] → [784 Sigmoid]
Encoder Decoder


#### Entraînement

- **K=10 autoencodeurs** avec bootstrap sampling (identique aux MLPs)
- Loss : **MSE reconstruction**
- Bottleneck extraction après entraînement

#### Calcul d'incertitude (identique aux MLPs)

$$u_a = H(\bar{p}(x))$$

$$u_e = MI = H(\bar{p}(x)) - \frac{1}{K}\sum_{k=1}^{K} H(p_k(x))$$

#### OOD Detection (identique aux MLPs)

Même stratégie ratio-based, seuil indépendant calculé sur MNIST

---

### Section 7 : Comparaison MLP vs Autoencodeurs

**Métriques comparées :**

| Aspect | MLP | Autoencodeur |
|--------|-----|--------------|
| Architecture | Supervisée | Non-supervisée |
| Prédictions | Softmax direct | Bottleneck softmax |
| FPR (MNIST) | ~25% | ? |
| TPR (Fashion) | ~90%+ | ? |
| Représentation | 128D → 10D softmax | 10D ReLU bottleneck |

**Questions explorées :**
- Quelle approche a meilleure séparation ID/OOD?
- L'absence de labels aide-t-elle ou nuit-elle?
- Les représentations autoencodeur sont-elles plus robustes?

---

## 🚀 Utilisation

```bash
cd demonstration/
jupyter notebook demo_mnist.ipynb
```
temps estimé : 1h