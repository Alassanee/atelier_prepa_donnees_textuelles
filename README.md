# Atelier – Préparation de données textuelles

Préparation d'un corpus d'avis clients en français (`smart_reviews_raw.csv`) pour une classification de sentiment (`positif`, `neutre`, `negatif`) : exploration, nettoyage, tokenisation, normalisation, découpage train/test, vectorisation, puis un bonus (embeddings, validation croisée, pipeline de prédiction).

Tout le travail est dans un seul notebook : [`notebooks/atelier_prepa_donnees_textuelles.ipynb`](notebooks/atelier_prepa_donnees_textuelles.ipynb).

## Structure du projet

```
atelier_prepa_donnees_textuelles/
├── data/
│   ├── smart_reviews_raw.csv        # données brutes (1200 avis, 8 colonnes)
│   ├── smart_reviews_cleaned.csv    # données nettoyées (généré, partie 4.8)
│   ├── tfidf_matrix.npz             # matrice TF-IDF d'entraînement (générée, partie 6.5)
│   ├── tfidf_matrix_test.npz        # matrice TF-IDF de test (générée, partie 6.5)
│   └── modele_sentiment.joblib      # pipeline de prédiction (généré, partie 7.4)
├── notebooks/
│   └── atelier_prepa_donnees_textuelles.ipynb
└── README.md
```

Les fichiers « générés » sont créés en exécutant le notebook.

## Les données

Colonnes du fichier brut : `id_avis`, `date`, `source`, `produit`, `texte`, `sentiment`, `note`, `langue`.

| Étape | Avis restants |
|---|---|
| Fichier brut | 1200 |
| Après suppression des textes manquants (2.1) | 1195 |
| Après suppression des doublons de texte (2.2) | 318 |
| Textes finaux distincts (après nettoyage complet) | 103 |

Le jeu de données est donc **très petit** une fois dédoublonné. Points relevés à l'exploration : 5 textes manquants et 1 texte vide, beaucoup de doublons, URLs, mentions, hashtags, emojis, ponctuation répétée, textes en majuscules, un avis aberrant de 635 caractères (une phrase répétée 12 fois) et une valeur `WEB` en majuscules dans `source`. Sur les données brutes, les classes sont déséquilibrées (65 % de positifs).

## Contenu du notebook

| Partie | Contenu |
|---|---|
| 1. Exploration | dimensions, types, valeurs manquantes, longueurs des textes, doublons, équilibre des classes, métriques adaptées (macro-F1), caractères spéciaux, répartition par source et par produit |
| 2. Nettoyage | suppression des NaN, des doublons, des URLs, des mentions ; hashtags conservés sans `#` ; espaces ; ponctuation répétée réduite ; fonction `nettoyer_texte` et colonne `texte_clean` |
| 3. Tokenisation | limites de `split()`, tokenisation NLTK (nombre de tokens par texte, moyenne, min, max) |
| 4. Normalisation | minuscules, accents conservés, stop words (suppression, avec conservation des négations, mots d'intensité et de `mais`), lemmatisation, colonne `texte_final`, sauvegarde du CSV nettoyé |
| 5. Découpage | 80 % / 20 %, `random_state=42`, stratifié, doublons de `texte_final` retirés pour éviter toute fuite entre train et test |
| 6. Vectorisation | Bag of Words, TF-IDF, TF-IDF avec bigrammes, comparaison, sauvegarde de la matrice, fuite de données |
| 7. Bonus | embeddings Hugging Face, comparaison avec TF-IDF, validation croisée, pipeline de prédiction |

## Choix méthodologiques

- **Stop words :** liste française de NLTK, sauf `pas`, `ne`, `mais` (et `jamais`, `sans`, `plus`, `rien`, `très`, `trop`, `peu`, `beaucoup`), qui portent le sentiment.
- **Ponctuation :** `!` et `?` sont conservés (émotion), le reste est retiré.
- **Lemmatisation** plutôt que stemming : NLTK n'a pas de lemmatiseur français, on utilise `simplemma`.
- **Vectoriseurs :** `token_pattern=r'\S+'` pour garder `!`, `?` et les emojis. Le vocabulaire est appris sur le train uniquement (`fit` sur le train, `transform` sur le test).
- **Embeddings :** `cmarkea/distilcamembert-base` (Hugging Face, 273 Mo), moyenne des vecteurs des mots.

## Résultats

Évaluation par validation croisée à 5 plis (macro-F1) sur les 103 avis :

| Représentation | macro-F1 |
|---|---|
| TF-IDF + régression logistique | 0,96 ± 0,03 |
| Embeddings + régression logistique | 0,92 ± 0,05 |

TF-IDF est légèrement meilleur ici : avis courts et répétitifs, les mots-clés suffisent. Le pipeline final (7.4) atteint 0,96 de macro-F1 sur le test.

## Installation

Python 3.13. Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn nltk simplemma certifi joblib ipykernel
pip install torch "transformers<5" sentencepiece protobuf   # partie 7 (embeddings)
python -m ipykernel install --user --name atelier-nlp --display-name "Python (atelier-nlp)"
```

Ouvrir le notebook et choisir le kernel **Python (atelier-nlp)**, puis **Run All**. Les ressources NLTK (`punkt_tab`, `stopwords`) se téléchargent depuis le notebook.

Si NLTK affiche `CERTIFICATE_VERIFY_FAILED` (macOS), la cellule de la partie 3.2 corrige le problème avec `certifi`.

`transformers` est limité à la version 4 : la version 5 ne charge pas le tokenizer de `distilcamembert-base`.

## Limites

- **Peu de données :** 103 avis distincts et un test de 21 avis, donc des scores qui varient beaucoup d'un découpage à l'autre. La validation croisée limite ce défaut sans le supprimer.
- **Corpus très répétitif :** des avis quasi identiques, ce qui rend la tâche facile et les scores élevés. Ils ne préjugent pas des performances sur des avis réels plus variés.
- **Recharger le modèle :** pour utiliser `modele_sentiment.joblib`, les fonctions de prétraitement du notebook doivent être définies dans le code qui le charge.

## Suite prévue

Un système multi-agents avec une interface web, qui prendrait un dataset en entrée et réaliserait automatiquement le nettoyage. Les étapes de ce notebook (partie 2 et fonction `nettoyer_texte`, partie 4, pipeline de la partie 7) en sont la base de départ.
