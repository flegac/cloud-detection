# Cloud detection experiments

## Datasets

- [38 clouds](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
- [95 clouds](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset)

## Intermediate Planning

- [X] Télécharger des images satellites de nuages
- [X] Create Dataset/Sample abstraction
- [ ] Train very small cnn (small dataset, overfitting, pyTorch)
- [ ] Create boosting train & prediction abstraction
- [ ] Train / predict with boosting (small dataset)

## Final Planning

- [X] (si besoin) Découper en tuiles de même taille
- [ ] Définir une classe `A` de nuages : "bien blancs / évident"
  - [ ] Choisir un dataset `D0_a` de 10 tuiles de la classe `A`
  - [ ] Entrainer un modèle `M0_a` sur `D0_a` (train/test)
  - [ ] Calculer un seuil `t0_a` permettant de garantir des TP à 90+%.
  - [ ] Réitérer sur les erreurs du modèle précédent (boosting)
- [ ] Définir une classe `B` de nuages : "vaporeux ?"
  - Idem
- [ ] Autres classes !

