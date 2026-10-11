# Trips

One folder per trip. The folder name is the URL slug. Put `index.md` and the photos side by side:

```
src/content/trips/lisbon-2026/
  index.md
  cover.jpg
  01.jpg
  02.jpg
```

`index.md`:

```md
---
title: Lisbon
country: Portugal
date: 2026-05-12
coords: [38.72, -9.14] # optional, [lat, lon]
cover: ./cover.jpg
photos:
  - src: ./01.jpg
    caption: Optional caption
  - src: ./02.jpg
---

Optional travel notes in markdown.
```

`/places` lists trips by date, newest first. The "Places" link in the header appears once the first trip exists. Photos are optimised and dithered at build time.
