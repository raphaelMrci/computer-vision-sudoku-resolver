# Trame slides (PDF)

## 1. Contexte

- Problématique : interaction visuelle avec une UI sans DOM.
- Contraintes du sujet : perception 100% vision, interaction web ensuite.
- Objectif : résoudre une grille Sudoku depuis `sudoku.com`.

## 2. Données et annotation

- Sources des captures (ex: niveaux easy/medium/hard).
- Préprocessing (crop, resize, normalisation).
- Outil d'annotation pour détection grille/cellules (YOLO format).

## 3. Approche A - OCR basic

- Description rapide du pipeline.
- Résultats benchmark (`ocr_mode=basic`).
- Limites observées.

## 4. Approche B - OCR advanced

- Heuristiques supplémentaires + résolution relaxée.
- Résultats benchmark (`ocr_mode=advanced`).
- Gains mesurés vs Approche A (latence/accuracy).

## 5. Détection par modèle (YOLO)

- Setup fine-tuning (`scripts/train_yolo.py`).
- Métriques de détection (mAP, recall) à rapporter.
- Exemple qualitatif de prédictions.

## 6. Démo live

- API (`/predict`, `/predict_debug`), résolution en direct.
- Interaction navigateur (clic/saisie) et précautions.

## 7. Conclusion

- Points forts, limites connues, pistes d'amélioration.
- Ouverture : OCR CNN, suivi temporel, robustesse multi-thèmes.
