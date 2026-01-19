Link to github [project](https://github.com/FAYCEL75/supervised-machine-learning)

# Prédiction du Taux de Conversion – Machine Learning Supervisé

## Abonnement à la newsletter *DataScienceWeekly*

---

## Présentation du projet

Ce projet traite un **cas d’usage business réel** :
**prédire si un visiteur d’un site web va s’abonner à une newsletter**.

L’objectif est double :

* **Business** : identifier les visiteurs à fort potentiel de conversion afin d’optimiser les actions marketing.
* **Data** : construire un modèle de machine learning supervisé **robuste, interprétable et optimisé pour le F1-score** dans un contexte de fort déséquilibre de classes.

Projet réalisé dans le cadre du **bootcamp Jedha – Supervised Machine Learning**

---

## Description des données

Deux fichiers sont fournis :

* `conversion_data_train.csv`
  → données étiquetées utilisées pour l’entraînement et la validation
* `conversion_data_test.csv`
  → données sans labels utilisées pour la prédiction finale

### Variables explicatives

* `country` — pays du visiteur
* `age` — âge
* `new_user` — nouvel utilisateur (1) ou utilisateur existant (0)
* `source` — canal d’acquisition (SEO, Ads, Direct)
* `total_pages_visited` — nombre de pages consultées lors de la session

### Variable cible

* `converted` — abonnement à la newsletter (0 / 1)

Le dataset est **fortement déséquilibré** (~3 % de conversions), ce qui oriente le choix de la métrique et des stratégies de modélisation.

---

## Méthodologie

### 1. Analyse exploratoire (EDA)

* Analyse globale du dataset (shape, types, valeurs manquantes)
* Étude du déséquilibre de la variable cible
* Distributions (histogrammes, boxplots)
* Segmentation convertis / non-convertis
* Corrélations numériques
* Taux de conversion par pays et source de trafic

**Insight clé** :
`total_pages_visited` est le facteur le plus déterminant de la conversion, proxy direct de l’engagement utilisateur.

---

### 2. Préprocessing (Pipeline scikit-learn)

Pipeline entièrement reproductible :

* **Variables numériques**

  * imputation par la médiane
  * standardisation (`StandardScaler`)
* **Variables catégorielles**

  * imputation par la valeur la plus fréquente
  * One-Hot Encoding (`handle_unknown="ignore"`)
* **Split stratifié** train / validation

Aucune fuite de données, même pipeline appliqué au jeu de test.

---

### 3. Modèles testés

* Régression logistique (baseline)
* Régression logistique régularisée (GridSearchCV)
* Random Forest
* Gradient Boosting

#### Gestion du déséquilibre

* test de `class_weight="balanced"`
* **optimisation du seuil de décision** (solution retenue)
* métrique d’évaluation : **F1-score uniquement**

---

## Modèle final retenu

**Régression Logistique régularisée + seuil de décision optimisé**

### Justification

* **F1-score élevé et stable (~0.77–0.78)**
* Modèle **interprétable** (coefficients, PDP)
* Entraînement rapide, déploiement simple
* Excellent compromis performance / lisibilité métier

Le seuil de décision est optimisé (~0.45) afin de maximiser le F1-score, plutôt que d’utiliser le seuil par défaut de 0.5.

---

## Interprétation du modèle

* Importance des variables via les coefficients (log-odds)
* Partial Dependence Plots sur les variables clés

### Principaux facteurs de conversion

1. **Nombre de pages visitées** (engagement)
2. **Utilisateurs existants** > nouveaux utilisateurs
3. **Pays** (segmentation géographique)
4. **Source de trafic** (qualité du canal d’acquisition)

Le modèle est **directement exploitable par les équipes marketing**.

---

## Recommandations business

* Encourager l’engagement (navigation interne, recommandations de contenu)
* Déclencher le CTA newsletter après un certain niveau d’engagement
* Adapter les campagnes par pays et par source
* Mettre en place une stratégie spécifique pour les nouveaux utilisateurs
* Prioriser les visiteurs à forte probabilité de conversion

---

## Livrable final

Le projet génère le fichier suivant :

```
submission_conversion_rate.csv
```

* 1 colonne : `converted`
* Valeurs binaires (0 / 1)
* Format conforme pour évaluation / mise en production

Distribution des prédictions :

* ~97 % non-convertis
* ~3 % convertis
  → cohérent avec les données d’entraînement (pas de sur-prédiction).

---

## Stack technique

* Python
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* Jupyter Notebook
* Git / GitHub

---

## Limites & perspectives

**Limites**

* Peu de variables comportementales (durée de session, device, campagne…)
* Pas de dimension temporelle
* Seuil optimisé sur validation (léger biais optimiste possible)

**Améliorations possibles**

* Enrichissement des features
* Modèles boosting avancés (XGBoost, LightGBM)
* Déploiement API (FastAPI)
* Monitoring et ré-entraînement continu

---

## 👤 Auteur

**Faycel Faddaoui**
