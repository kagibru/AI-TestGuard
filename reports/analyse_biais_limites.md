
# AI-TestGuard — Analyse des biais et limites

## 1. Objectif

Identifier les biais potentiels et les limites
du modèle de détection de fraude, à partir
des caractéristiques du jeu de données et
des résultats expérimentaux.

## 2. Déséquilibre des classes

Le jeu Kaggle contient 284 807 transactions,
dont seulement 492 fraudes, soit environ 0,173 %.

Ce déséquilibre implique que l'accuracy seule
est insuffisante pour évaluer le modèle.

Nous privilégions notamment :
- la précision ;
- le rappel ;
- le F1-score ;
- le ROC-AUC ;
- l'Average Precision ;
- les faux positifs et les faux négatifs.

## 3. Limites liées à l'anonymisation

Les variables V1 à V28 sont anonymisées
et issues d'une transformation ACP.

Cette anonymisation limite :
- l'interprétation métier des variables ;
- l'identification des causes des prédictions ;
- l'analyse des biais démographiques.

Aucune conclusion sur l'équité entre groupes
démographiques ne peut être établie à partir
des variables disponibles.

## 4. Risques liés aux erreurs de classification

Sur Validation, la régression logistique
au seuil de 0,90 produit :

- 241 faux positifs ;
- 17 faux négatifs ;
- une précision d'environ 25,16 % ;
- un rappel d'environ 82,65 %.

Les faux positifs peuvent entraîner
des vérifications inutiles ou des perturbations
pour des clients légitimes.

Les faux négatifs représentent des fraudes
qui ne sont pas détectées.

Le coût métier réel de ces erreurs n'a pas
été quantifié.

## 5. Sensibilité aux perturbations

Les expériences de robustesse montrent :

- une faible variation du F1-score sous
  un bruit numérique de 1 % ;
- une diminution du F1-score de 0,3857
  à 0,3424 lorsque les montants
  augmentent artificiellement de 20 %.

Ces résultats suggèrent une sensibilité
à certaines modifications des données.

Ils ne démontrent pas à eux seuls
une dérive réelle en production.

## 6. Limites de généralisation

Les données correspondent à un jeu historique
de transactions anonymisées.

Les résultats ne garantissent pas des
performances identiques :
- dans une autre banque ;
- dans un autre pays ;
- sur une autre période ;
- face à de nouvelles stratégies de fraude.

Le partitionnement aléatoire stratifié
préserve les proportions de classes, mais
ne constitue pas une validation chronologique.

Il peut également laisser subsister des
dépendances entre observations que les
variables disponibles ne permettent pas
de contrôler complètement.

## 7. Recommandations

Avant une utilisation opérationnelle :

1. Réaliser une validation temporelle.
2. Définir le coût métier des faux positifs
   et des faux négatifs.
3. Tester le modèle sur des données récentes
   et représentatives de l'environnement cible.
4. Surveiller la dérive des distributions.
5. Suivre les performances après obtention
   des étiquettes réelles.
6. Réaliser une analyse d'équité si des
   données appropriées et légalement
   utilisables deviennent disponibles.
7. Prévoir une supervision humaine
   des alertes à fort impact.

## 8. Conclusion

Le modèle présente une capacité intéressante
de détection des fraudes sur le jeu étudié.

Toutefois, l'anonymisation, le déséquilibre
des classes, la sensibilité aux perturbations
et l'absence de validation sur un environnement
opérationnel réel limitent les conclusions.

Le modèle ne doit pas être considéré comme
apte à une mise en production sans contrôles
et validations complémentaires.
