# Computer Vision Sudoku Solver

Sudoku solver piloté par vision par ordinateur pour `https://sudoku.com/`.

Objectif : capturer la grille, détecter les cellules visuellement, reconnaître les chiffres,
résoudre le Sudoku, puis remplir la grille automatiquement.

## Scope et contraintes du sujet

- Perception de la grille **uniquement par vision** (pas de DOM pour détecter grille/chiffres).
- Interaction navigateur autorisée pour les actions (clics/saisie).
- Fine-tuning d'un détecteur d'objets attendu pour la détection grille/cellules (YOLO recommandé).
- Livrables : code propre, Docker fonctionnel, README reproductible, poids de modèles.

## Structure du projet

```text
.
├── api/
│   └── app.py
├── src/
│   ├── automation/
│   │   └── interface.py
│   ├── detection/
│   │   └── interface.py
│   ├── ocr/
│   │   └── interface.py
│   ├── solver/
│   │   └── backtracking.py
│   └── pipeline.py
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lancer l'API locale :

```bash
make api
```

Tester le endpoint :

```bash
curl -X POST "http://localhost:8080/predict" -F "file=@/path/to/image.png"
```

Endpoint debug (image annotée en base64) :

```bash
curl -X POST "http://localhost:8080/predict_debug" -F "file=@/path/to/image.png"
```

Sauvegarder l'image debug en local (simple) :

```bash
make debug-image IMAGE="/path/to/image.png" OUT="debug.png" OCR_MODE="advanced"
```

## Commandes utiles

- `make api` : lance l'API Flask locale sur le port `8080`.
- `make train` : entraîne YOLO (alias de `make train-yolo`).
- `make infer IMAGE=...` : placeholder pour inférence locale.
- `make demo` : exécute un run pipeline local (placeholder).
- `make debug-image IMAGE=... OUT=...` : appelle `/predict_debug` et sauvegarde le PNG annoté.
- `make benchmark-ocr MANIFEST=... BENCH_OUT=...` : comparaison chiffrée OCR `basic` vs `advanced`.
- `make benchmark-targets BENCH_OUT=... TARGET_REPORT_OUT=...` : vérifie les métriques vs objectifs fiabilité/latence.
- `make generate-manifest MANIFEST_INPUT=... MANIFEST_OUT=...` : autogénère un manifest depuis un dossier d'images.
- `make filter-manifest MANIFEST_OUT=... FILTERED_MANIFEST_OUT=...` : filtre les samples OCR trop bruités avant dataset CNN.
- `make build-ocr-dataset MANIFEST=...` : construit un dataset OCR CNN depuis captures labellisées.
- `make train-ocr-cnn OCR_DATA_DIR=...` : entraîne un OCR CNN (classes `0..9`).
- `make train-yolo YOLO_DATA=...` : fine-tuning YOLO pour détection grille/cellules.
- `make live-solve` : calibration écran + capture + résolution + saisie live (OCR Tesseract).
- `make capture-dataset` : boucle auto `attente -> screenshot -> New Game -> Easy`.
- `make docker-build` : build de l'image Docker.
- `make docker-run` : run de l'API en conteneur.

### Champs OCR dans la réponse

La réponse `POST /predict` inclut :

- `ocr_confidence_mean`
- `ocr_confidence_min`
- `ocr_confidence_max`
- `ocr_confidence_std`
- `num_uncertain_clues`
- `num_confident_clues`
- `clue_quality_score`
- `solve_eligible`
- `uncertain_cells`

Ces scores permettent d'activer un mode "safe" côté automation (ne pas exécuter les actions si la confiance est trop faible).

Le paramètre query `ocr_mode` permet de comparer les approches :

- `ocr_mode=advanced` (par défaut)
- `ocr_mode=basic`
- `ocr_mode=hybrid`
- `ocr_mode=cnn` (nécessite `models/ocr_cnn.pt`)
- `ocr_mode=hybrid_cnn`
- `ocr_mode=tesseract` (nécessite binaire Tesseract installé sur la machine)
- `ocr_mode=hybrid_tesseract`

Exemple :

```bash
curl -X POST "http://localhost:8080/predict?ocr_mode=basic" -F "file=@/path/to/image.png"
```

## Docker

Construire et exécuter l'API vision :

```bash
make docker-build
make docker-run
```

Puis tester :

```bash
curl -X POST "http://localhost:8080/predict" -F "file=@/path/to/image.png"
```

## Roadmap implémentation

1. **Détection** : remplacer le stub par un modèle YOLO fine-tuné (grille + cellules).
2. **OCR** : comparer au moins 2 approches (OCR pré-entraîné vs template matching/CNN).
3. **Solveur** : backtracking + heuristique MRV (base déjà fournie).
4. **Automation** : clic/saisie sur sudoku.com depuis coordonnées image.
5. **Évaluation** : métriques comparatives chiffrées (latence, accuracy OCR, succès complet).

## Aller au bout du sujet (checklist exécutable)

1. **Construire le dataset détection** (annotations grille/cellules) et créer `data/sudoku-detector.yaml`.
2. **Fine-tuner YOLO** :

```bash
make train-yolo YOLO_DATA="data/sudoku-detector.yaml" YOLO_DEVICE="0"
```

3. **Créer un benchmark OCR labellisé** à partir de `docs/benchmark-manifest.example.json`.
4. **Mesurer objectivement `basic` vs `advanced`** :

```bash
make benchmark-ocr MANIFEST="docs/benchmark-manifest.example.json" BENCH_OUT="artifacts/ocr_benchmark.json"
```

5. **Compléter les slides** avec métriques et analyses.
6. **Finaliser l'automatisation web live** (clic/saisie) en gardant la perception 100% vision.

## OCR CNN (grilles difficiles)

1. Préparer un manifest de captures avec la grille initiale (`docs/benchmark-manifest.example.json`).
2. Construire le dataset de cellules:

```bash
make build-ocr-dataset MANIFEST="docs/benchmark-manifest.example.json"
```

3. Entraîner le modèle:

```bash
make train-ocr-cnn OCR_DATA_DIR="data/ocr_cnn" OCR_CNN_EPOCHS=15
```

4. Utiliser le modèle en live:

```bash
python3 -m scripts.live_solve --min-clues 17
```

Tu peux aussi passer directement la zone de grille si tu veux éviter la calibration interactive:

```bash
python3 -m scripts.live_solve --x 318 --y 166 --size 568 --min-clues 17
```

## Collecte automatique d'images (dataset brut)

Pour générer rapidement beaucoup de captures:

```bash
make capture-dataset CAPTURE_OUT="data/raw_screenshots" CAPTURE_INTERVAL=3 CAPTURE_PREFIX="sudoku"
```

Le script te demande:

1. zone de screenshot (coin haut-gauche puis bas-droit),
2. position du bouton `New Game`,
3. position du bouton `Easy`.

Puis boucle:

- attend `CAPTURE_INTERVAL` secondes,
- prend un screenshot,
- clique `New Game`,
- clique `Easy`,
- recommence jusqu'à `Ctrl+C`.

## Manifest auto-généré depuis les screenshots

Pour générer un manifest prêt à être consommé (prérempli OCR):

```bash
make generate-manifest MANIFEST_INPUT="data/raw_screenshots" MANIFEST_OUT="data/manifest_autogen.json" OCR_MODE="hybrid"
```

Notes:

- Le champ `grid` est prérempli automatiquement par OCR.
- Tu peux l'utiliser directement pour `build-ocr-dataset`, mais une relecture/correction manuelle est recommandée pour un dataset propre.

Filtrer ensuite les cas trop bruités (hard/master) avant entraînement:

```bash
make filter-manifest MANIFEST_OUT="data/manifest_autogen.json" FILTERED_MANIFEST_OUT="data/manifest_filtered.json"
```

Voir `docs/subject-checklist.md` pour l'avancement complet.

## Démo live navigateur (sans DOM)

Pré-requis macOS:

- Donner les permissions **Accessibility** et **Screen Recording** au terminal/IDE.
- Ouvrir la page Sudoku et afficher la grille en entier.

Lancer:

```bash
make live-solve
```

Le script:

1. demande une calibration (coin haut-gauche puis bas-droit),
2. capture la grille,
3. applique le pipeline CV,
4. remplit automatiquement les cases vides si les seuils de sécurité sont satisfaits.

Version simple recommandée:

```bash
python3 -m scripts.live_solve --min-clues 17
```

## Évaluation et métriques conseillées

Comparer les approches dans les mêmes conditions :

- Temps moyen `capture -> grille remplie`.
- Taux de cellules correctement détectées.
- Accuracy OCR par cellule.
- Taux de grilles entièrement résolues automatiquement.
- Distribution des erreurs (erreurs de détection, OCR, interaction).

## Notes importantes

- Le Docker est surtout pertinent pour la partie modèle/API vision.
- La partie interaction browser peut rester en local host si nécessaire pour la démo live.