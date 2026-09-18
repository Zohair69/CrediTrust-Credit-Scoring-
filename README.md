# 💳 CrediTrust Finance : Moteur de Crédit Scoring

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Statut](https://img.shields.io/badge/Statut-Projet%20de%20formation-lightgrey)

Projet réalisé dans le cadre de la formation **Data Analyst, Simplon Lyon**, brief *Machine Learning & Classification* (Activité 3), pour le client fictif **CrediTrust Finance**.

**🚀 Démo en ligne :** [creditrust-scoring.streamlit.app](https://creditrust-scoring.streamlit.app)
**📑 Présentation :** [presentation.pdf](./presentation.pdf)

---

## 🎯 Contexte et objectif

CrediTrust Finance souhaite automatiser une partie de son processus d'octroi de crédit. L'objectif du projet est de construire un modèle de classification capable de prédire si une demande de prêt doit être **accordée (Y)** ou **refusée (N)**, à partir du profil du demandeur (revenus, historique de crédit, situation familiale, zone géographique...).

**Priorité métier explicite du brief :** réduire les faux négatifs. Un mauvais payeur non détecté coûte plus cher à la banque qu'un bon payeur refusé à tort — le **rappel sur la classe N** est donc le critère de choix retenu, avant l'accuracy globale.

## 👥 Équipe et répartition du travail

Projet réalisé en trinôme sur un notebook partagé en temps réel :

| | Rôle |
|---|---|
| **Laurie** | Valeurs manquantes catégorielles (remplacement par `"Inconnu"`) + modèle de Régression Logistique |
| **David** | Valeurs manquantes numériques (médiane calculée sur le train uniquement) + modèle d'Arbre de Décision |
| **Zohair** | Encodage des variables catégorielles + alignement train/test + modèles Random Forest (standard et balanced) |

## 📁 Structure du dépôt

```
CrediTrust-Credit-Scoring-/
├── notebook_3.ipynb          # Notebook d'analyse complet (EDA, prétraitement, modélisation)
├── dashboard_creditrust.py   # Dashboard Streamlit interactif
├── requirements.txt          # Dépendances Python
├── loan_data.csv             # Jeu de données
├── presentation.pdf          # Support de présentation (restitution orale)
└── README.md
```

## 🧠 Modèles comparés

Quatre modèles ont été entraînés et évalués sur un jeu de test (20%, stratifié) :

| Modèle | Accuracy | Rappel classe N |
|---|---|---|
| Régression Logistique | 86.2% | 57.9% |
| Arbre de Décision | 74.8% | 60.5% |
| Random Forest | 83.7% | 63.2% |
| **Random Forest (balanced)** | 83.7% | **68.4%** |

### 🏆 Modèle retenu : Random Forest (`class_weight='balanced'`)

La Régression Logistique affiche la meilleure accuracy globale, mais rate près de 4 mauvais payeurs sur 10 (rappel classe N = 57.9%). Le Random Forest balanced conserve une bonne accuracy (83.7%) tout en détectant le mieux les mauvais payeurs (68.4%) — un compromis assumé, directement aligné sur la priorité métier du brief.

## 📊 Dashboard Streamlit

Le dashboard reprend toutes les étapes du projet, avec des graphiques interactifs (Plotly) :

- **Contexte** — objectif métier, répartition de l'équipe
- **Exploration des données** — distributions, corrélations, croisements catégoriels
- **Prétraitement** — récapitulatif du pipeline (imputations, encodage, standardisation)
- **Comparaison des modèles** — tableau et graphique des 4 modèles
- **Importance des variables** — variables les plus déterminantes du modèle final + matrice de confusion
- **Simulateur de prêt** — formulaire interactif → décision instantanée du modèle final
- **Recommandations & limites**

## ⚙️ Lancer le projet en local

```bash
git clone https://github.com/Zohair69/CrediTrust-Credit-Scoring-.git
cd CrediTrust-Credit-Scoring-
pip install -r requirements.txt
streamlit run dashboard_creditrust.py
```

Le fichier `loan_data.csv` doit être présent à la racine du dépôt (déjà inclus).

## 🛠️ Stack technique

- **Python** (pandas, numpy, scikit-learn)
- **Streamlit** + **Plotly** pour le dashboard interactif
- **CoCalc** pour le travail collaboratif sur le notebook

## ⚠️ Limites connues

- Jeu de données de taille modeste
- Déséquilibre des classes (~69% Y / 31% N) partiellement corrigé, pas éliminé
- Pas de validation externe (autre banque, autre période) à ce stade
- Le modèle reflète des données historiques et peut reproduire des biais déjà présents dans les décisions passées

---

*Projet académique à Simplon Lyon, formation Data Analyst.*
