# Spécification des tranches Pléiade

Cette spécification décrit l'interprétation web de la Bibliothèque de la
Pléiade utilisée sur la homepage et la page Bookshelf. Elle ne demande pas de
reproduire les marques Gallimard ou NRF.

## Référence principale

- Photographie de référence : <https://commons.wikimedia.org/wiki/File:Biblioth%C3%A8que_de_la_Pl%C3%A9iade.JPG>
- Sujet : 26 volumes réels de la Bibliothèque de la Pléiade vus de face
- Auteur : LPLT
- Source : <https://commons.wikimedia.org/wiki/File:Biblioth%C3%A8que_de_la_Pl%C3%A9iade.JPG>
- Licence : CC BY-SA 3.0
- Dimensions : 2288 × 1712 pixels

Le [catalogue officiel 2025](https://www.gallimard.fr/system/files/inline-files/Catalogue-Pleiade-2025.pdf)
complète la photographie pour la nomenclature de la collection.

## Anatomie observée

- Format physique d'environ 11,5 × 17,5 cm. Le rapport hauteur sur largeur de
  la couverture vaut environ 1,52. Vue de dos, la hauteur domine fortement la
  largeur variable de la tranche.
- Dos lisse aux coins légèrement arrondis. La courbure produit un centre plus
  clair et des bords plus sombres.
- Filets dorés horizontaux très rapprochés sur la majorité du dos.
- Panneau central uni qui interrompt les filets. Il occupe environ 25 à 35 % de
  la hauteur.
- Auteur puis titre en capitales, centrés et composés sur plusieurs lignes.
- Tomaison ou petits signes sous le titre quand le volume appartient à une
  série.
- Peu de variation de hauteur entre volumes. La largeur varie davantage.

## Palette de couleurs

Le rendu utilise une palette de huit cuirs inspirée de la collection. La
couleur n'est pas une classification historique stricte : elle est attribuée
une fois par auteur dans `PLEIADE_AUTHOR_COLORS`, afin qu'une étagère conserve
une vraie variété tout en gardant tous les livres d'un auteur cohérents.

| Emplacement de palette | Couleur de départ du rendu |
| --- | --- |
| Vert | `#416653` |
| Violet | `#574363` |
| Corinthe | `#70483f` |
| Rouge vénitien | `#813b34` |
| Bleu | `#315a78` |
| Vert émeraude | `#2f6652` |
| Havane | `#76533e` |
| Gris | `#62615d` |

La dorure commence à `#c8aa62`. Ces valeurs sont des points de départ ajustés
pour l'écran, pas des mesures colorimétriques des cuirs.

## Règles de rendu

- HTML et CSS uniquement pour la tranche. Le titre et l'auteur restent du vrai
  texte.
- Hauteur cible desktop : 300 à 330 px. Largeur : 42 à 90 px selon la
  composition du titre.
- Une seule anatomie. La couleur, la largeur, le titre et l'auteur sont les
  seules données variables.
- Une rangée physique contient au plus dix livres et se replie sur plusieurs
  rangées sans défilement horizontal.
- Le relief reste discret. Aucun livre ne flotte et aucun mouvement n'est
  nécessaire pour comprendre le contenu.
- Le focus clavier identifie clairement le livre. Le texte complet est aussi
  présent dans son nom accessible.

## Critères de validation

1. À côté de la photographie, la densité des filets et la position du panneau
   central doivent être reconnaissables avant même de lire le titre.
2. Une rangée doit ressembler à une collection éditoriale cohérente, pas à un
   assortiment de couvertures.
3. Un titre long doit rester contenu sans déborder.
4. Les huit couleurs doivent rester distinctes sous une lumière d'écran
   normale.
5. Le rendu doit tenir à 390 px sans réduire les zones cliquables sous 44 px.

## État validé

- Chaque tranche est construite en HTML et CSS. L'auteur et le titre restent du
  texte réel, sélectionnable et accessible.
- La dorure horizontale suit un cycle dense de 4 px : 3 px sans filet, puis
  1 px doré. Le panneau typographique central interrompt ce motif.
- La palette couvre les huit couleurs définies ci-dessus. La couleur est
  déterminée par auteur dans `PLEIADE_AUTHOR_COLORS`, donc deux livres du même
  auteur ont toujours le même dos sans dépendre de leur année de lecture.
- Les livres terminés de `booksData.ts` ont une largeur normalisée de 42 à 90 px
  selon le nombre de pages de l'édition identifiée par ISBN. Le livre le plus
  court reste ainsi lisible sans prétendre reproduire une épaisseur physique à
  l'échelle.
- Le composant expose deux échelles : `showcase` pour la vitrine compacte de la
  homepage, et `extended` pour les étagères annuelles. Cette dernière passe de
  58 à 118 px, monte à 330 px et centre les petits groupes.
- `PleiadeShelf.astro` découpe les grands groupes en niveaux de dix livres au
  maximum. Les niveaux restent entièrement visibles, y compris sur mobile.
- Chaque niveau occupe la largeur de l'étagère ; les livres se répartissent sur
  cette largeur, tandis que les petits groupes restent centrés et jointifs.
- Décision de revue : **ship**. Le rendu Pléiade est intégré à la homepage et à
  la page Bookshelf via `PleiadeShelf.astro`.

## Question de direction

Quelle quantité de matière et de profondeur faut-il conserver pour que
l'étagère évoque immédiatement la Pléiade tout en restant compatible avec le
« quiet personal index » du site ?

## Contrat de direction

- Seed key : `pleiade-reference-2008`
- THESIS : une rangée de dos codés par auteur transforme l'archive de lecture
  en objet reconnaissable sans remplacer les titres par des images.
- OWN-WORLD : le papier chaud, les espaces et la typographie éditoriale du site
  restent inchangés autour de l'étagère. Le cuir et l'or n'existent que sur les
  livres.
- STORY : la photographie établit la preuve, puis la vitrine et les étagères
  étendues adaptent la matière au contexte de lecture.
- FIRST VIEWPORT : le nom du site et la première étagère restent lisibles avant
  le défilement.
- FORM : une vue orthographique frontale, construite avec des dos étroits, une
  dorure horizontale dense et un panneau typographique central. La variante
  calme retire du relief, jamais les signes distinctifs de la collection.
