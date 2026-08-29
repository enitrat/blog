# Recherche — étagère de livres façon Pléiade

Recherche effectuée le 29 août 2026. Objectif : trouver des implémentations
open source qui peuvent servir de base à une étagère statique Astro, avec des
tranches produites en HTML/CSS/SVG et du texte réel.

## Résultats classés

### 1. `petargyurov/virtual-bookshelf` — meilleure base pour ce site

- Dépôt : <https://github.com/petargyurov/virtual-bookshelf>
- Technique : HTML + CSS + JavaScript vanilla. Chaque livre est composé d’une
  tranche, d’un dessus et d’une couverture ; les transformations 3D CSS donnent
  l’effet de profondeur et de sélection au survol. La feuille de style utilise
  aussi `writing-mode: vertical-rl` pour le titre et des motifs CSS générés par
  gradients. Voir [le markup documenté dans le README](https://github.com/petargyurov/virtual-bookshelf#how-do-i-add-more-books)
  et [la feuille de style](https://github.com/petargyurov/virtual-bookshelf/blob/main/bookshelf.css).
- Licence : Unlicense, indiquée par le dépôt et présente dans
  [`LICENSE`](https://github.com/petargyurov/virtual-bookshelf/blob/main/LICENSE).
- État : petit projet personnel de 29 commits, 441 étoiles et 52 forks au
  moment de la recherche ; le README signale lui-même les limites sur les titres
  longs. Il n’y a pas besoin d’en reprendre le JavaScript pour Astro.
- Réutilisable ici : reprendre le modèle de données HTML et les idées de
  profondeur/gradients, puis supprimer le JS, la couverture Picsum et la
  randomisation. Remplacer les motifs par un gabarit contrôlé : couleur de
  reliure, filets, cartouche, titre et auteur en texte accessible.

### 2. `steveworkman/cssbookshelf` — référence CSS historique

- Dépôt : <https://github.com/steveworkman/cssbookshelf>
- Technique : étagère CSS3 ; l’auteur décrit explicitement l’usage de
  gradients CSS pour les tranches afin d’éviter des images pour la plupart des
  livres dans [l’article de présentation](https://www.steveworkman.com/2012/css-bookshelf-now-on-github/).
- Licence : le dépôt est bien public, mais la licence n’est pas identifiable
  de façon fiable depuis sa page actuelle ; ne pas copier le code sans vérifier
  le fichier de licence dans un clone.
- État : projet ancien (l’article date de 2012) et donc utile comme référence
  visuelle, pas comme dépendance à intégrer.
- Réutilisable ici : l’idée « une tranche est un assemblage de gradients et de
  bordures » confirme qu’un rendu Pléiade-like n’a pas besoin de bitmap ni de
  librairie JS.

### 3. `capjamesg/cv-book-svg` — utile seulement pour le modèle SVG interactif

- Dépôt : <https://github.com/capjamesg/cv-book-svg>
- Technique : segmentation d’une photo de bibliothèque avec Grounding DINO,
  Segment Anything, OpenCV et GPT-4 Vision, puis génération d’un fichier HTML
  contenant un SVG superposé ; chaque tranche devient un polygone cliquable.
  Les détails sont dans le [README](https://github.com/capjamesg/cv-book-svg#how-it-works).
- Licence : MIT, voir [`LICENSE`](https://github.com/capjamesg/cv-book-svg/blob/main/LICENSE).
- État : 23 commits, 134 étoiles et 17 forks au moment de la recherche ; le
  dépôt documente des erreurs possibles d’identification et de liens.
- Réutilisable ici : uniquement le principe d’un SVG avec des zones cliquables.
  Le pipeline de vision est hors sujet pour une liste de livres connue et
  ajouterait des dépendances, du coût et une surface d’erreur inutiles.

### 4. `aloglu/bookshelf` — application complète, pas un moteur de tranches

- Dépôt : <https://github.com/aloglu/bookshelf>
- Technique : application Go qui génère un site avec vues étagère, pile et
  coverflow ; elle convertit aussi les couvertures et extrait des couleurs de
  tranche. La documentation du dépôt décrit le build statique, les données JSON
  et les différentes vues.
- Licence : MIT, voir [`LICENSE`](https://github.com/aloglu/bookshelf/blob/main/LICENSE).
- État : projet actif mais beaucoup plus large que le besoin ; sa page indique
  une gestion de bibliothèque complète et des contrôles de publication.
- Réutilisable ici : éventuellement l’idée de dériver une couleur depuis une
  couverture si le site décide d’utiliser les couvertures réelles. Ne pas
  intégrer l’application : Astro possède déjà la source de vérité et le rendu.

## Ce que « façon Pléiade » doit vouloir dire dans le projet

Les dépôts ci-dessus ne fournissent pas un gabarit Pléiade prêt à l’emploi.
Il faut donc définir le style comme un petit système visuel, pas comme une
ressemblance vague :

1. un fond de reliure uni par livre, dans une palette limitée ;
2. des filets horizontaux dorés, serrés sur presque toute la hauteur, interrompus
   par un panneau central réservé à l'auteur et au titre ;
3. typographie claire, centrée, dorée ou crème, composée horizontalement sur
   plusieurs lignes avec titre et auteur en texte HTML/SVG sélectionnable ;
4. proportions étroites et hauteurs légèrement variables, sans texture
   photoréaliste ;
5. un seul gabarit dont les variables sont la couleur, la largeur, le titre,
   l’auteur et éventuellement une variante d’ornement.

La validation ne devrait pas consister à demander si chaque tranche « ressemble
à une vraie Pléiade », mais à comparer une planche de 6 à 10 tranches au
référentiel choisi : palette, densité des filets, hiérarchie typographique,
lisibilité à petite taille et cohérence entre livres. Pour une inspiration de
collection, utiliser le [catalogue officiel 2025 de Gallimard](https://www.gallimard.fr/system/files/inline-files/Catalogue-Pleiade-2025.pdf)
comme référence visuelle, sans copier logos, couvertures ou éléments protégés.

## Intégration livrée

Le rendu livré utilise `PleiadeSpine.astro` et `PleiadeShelf.astro`, sans
dépendance supplémentaire. Les liens exposent un `aria-label`, tandis que le
titre et l'auteur restent du texte HTML réel. La homepage utilise la variante
`showcase`; la page Bookshelf utilise `extended` et regroupe les livres lus par
année.

Le meilleur code à reprendre est donc la géométrie CSS de
`virtual-bookshelf`, pas son apparence aléatoire. Le style Pléiade doit être
conçu localement comme un gabarit déterministe, ce qui rend chaque correction
globale et chaque ajout de livre prévisible.
