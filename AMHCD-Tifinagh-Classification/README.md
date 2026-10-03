# AMHCD Tifinagh Character Recognition (MLP From Scratch)

Implementation d'un reseau de neurones multicouches (MLP) entierement code from scratch avec NumPy pour la classification des caracteres manuscrits Tifinagh de la base AMHCD.

## Architecture du Modele
- **Entree** : 1024 caracteristiques (images redimensionnees en 32x32 pixels normalisees dans [0, 1])
- **Couche Cachee 1** : 64 neurones, activation ReLU
- **Couche Cachee 2** : 32 neurones, activation ReLU
- **Couche de Sortie** : 33 neurones, activation Softmax
- **Fonction de Perte** : Entropie croisee categorielle (Categorical Cross-Entropy)
- **Optimisation** : Mini-Batch SGD (taille de batch = 32, learning rate = 0.01)

## Resultats Obtenus
- **Train Accuracy** : 85.99%
- **Validation Accuracy** : 83.61%
- **Test Accuracy (5148 images)** : 84.00%
- **Macro / Weighted F1-Score** : 0.84

## Structure du Repertoire
- `src/` : code source (`activations.py`, `model.py`, `data_loader.py`, `utils.py`, `optimizers.py`)
- `reports/figures/` : graphiques d'evaluation (`loss_accuracy_plot.png`, `confusion_matrix.png`)
- `reports/` : rapport scientifique sous format IMRAD (`report_IMRAD.md`)
- `train.py` : script d'execution du pipeline complet

## Execution
```bash
pip install -r requirements.txt
python train.py