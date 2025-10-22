# PINN-Project
Répertoire GitHub pour le projet intégrateur 3A - Physics-Informed Neural Network for the Macroscopic Estimation of Pedestrian Behavio

# Journal de projet

## 7 octobre
- Début du projet, rencontre avec Didier George et premières explications sur la méthode  
  _(Voir fiche de notes personnels de chacun)_  
- Code fourni et articles à lire pour la semaine prochaine

## 14 octobre
- Visio de 1h avec D.G  
- Questions + nouveau code pour inspiration

## 21 octobre
- Vérification du bon fonctionnement du GitHub  
- Initiation à Visual Studio Code

## 22 octobre
- **Valentin** :  
  - Modification de la simulation pour en faire une fonction réutilisable dans les prochains codes  
  - Création d’un fichier test  
- **CA** :  
  - Création d'une fonction permettant de quadriller une zone avec la précision voulue  
- **Enzo** :  
  - Création d'une fonction permettant de récupérer les positions et faire une représentation de la densité au cours du temps
 
### Résumé de la réunion 
- **aspect technique** :
  - Augmenter le pas de temps (dx ~ 0.25) pour avoir beaucoup de petit carré. On peut par exemple augmenter la longueur de la rue L à 14m.  On veut une résolution forte pour une densité optimale.
  - Augmenter le nombre d'individus dans la simulation (pas trop car cela peut ralentir les calculs)
- ** Prochaines étapes techniques ** :
    - calcul de la vitesse pour obtenir un *champ de vitesse empirique* ( on pourra le multiplier par le $\rho$ empirique et obtenir *le flux d'individus par unité de temps empirique*.
    - tracer le flux en fonction de la densité.
- **Division du travail** :
   - Quelqu'un peut se renseigner sur les PINN pendant que les autres travaillent sur les données. (Paralléliser les tâches)
   - Vidéo youtube recommandées : [Vidéo Steve Brunton ](https://youtu.be/-zrY7P2dVC4?si=1fccEmBpOoaG8Ipj) ou [Murad Intro to PINN](https://www.youtube.com/watch?v=G_hIppUWcsc) ou [Elastropy PINNS exemple d'application](https://www.youtube.com/watch?v=gXv1SGoL04c)
