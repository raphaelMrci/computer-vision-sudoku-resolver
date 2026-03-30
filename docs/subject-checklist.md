# Checklist final du sujet

## Pipeline attendu

- [x] Analyse visuelle de la grille (sans DOM).
- [x] Reconnaissance des chiffres.
- [x] Solveur Sudoku (backtracking + contraintes).
- [x] Interaction navigateur en live (clic/saisie).

## Modèles et approches

- [ ] Détection objets fine-tunée (YOLO/D-FINE) pour grille/cellules.
- [x] Approche OCR template matching/OpenCV.
- [x] Option OCR alternative pour comparaison (`ocr_mode=basic` vs `ocr_mode=advanced`).

## Comparaison chiffrée (obligatoire)

- [x] Script de benchmark OCR (`scripts/benchmark_ocr.py`).
- [ ] Manifest benchmark rempli avec plusieurs captures labellisées.
- [ ] Tableau final comparatif (latence, accuracy, recall/précision).

## Livrables repo

- [x] README de lancement.
- [x] Dockerfile fonctionnel.
- [ ] Poids entraînés à versionner / fournir (`models/*.pt`).
- [x] Code structuré.
- [ ] Notebook(s) entraînement/évaluation (optionnel mais recommandé).

## Slides

- [ ] Contexte + contraintes.
- [ ] Données + preprocessing + annotation.
- [ ] Approche A → résultats → analyse.
- [ ] Approche B → résultats → analyse.
- [ ] Comparaison objective et chiffrée.
- [ ] Stack + défis + solutions.
- [ ] Démo live.
- [ ] Limites + améliorations.
