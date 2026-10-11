---
version: 1
slug: "src-pages-index-astro"
primary_target: "src/pages/index.astro"
related_targets: []
---

# Home: the bench

## Mode
Experience, with immediate orientation. Replaces the quiet personal index and the boxed living room.

## Job
A peer arriving from X or Telegram learns in one screen who Mathieu is and what holds his attention now, then follows whichever object interests them. Work is visible but not the front door.

## Content (real sources)
- Bio: Mathieu's own sentence (verbatim), role, X / GitHub / Telegram.
- Now: 1–4 short items Mathieu writes himself (new data file; empty state hides the tile).
- Reading: `src/booksData.ts` (current book + flickable shelf) → `/reading`.
- Writing: blog collection, latest first → `/writing/<slug>`.
- Places: trips collection (new; Mathieu supplies photos) → `/places`. Empty state: tile hidden until the first trip exists.
- The room: the existing live 3D living room (`LivingRoom.astro` + `scene.ts`) inside a tile; record player = the SoundCloud set.
- Work: work collection, compact.

## Direction contract
THESIS: The home page is a bench of small working objects from Mathieu's life, each one real and each one a door. It refuses the category default of a name, a role line and three stacked lists, and it refuses the room-as-whole-site.
OWN-WORLD: Cool grey bench, near-white paper tiles and one graphite tile family, a single signal orange used for state (playing, current, latest). Tiles separated by space and a 1px hairline, no drop shadows, no glass. Geist for everything, Geist Mono for small labels and data. Photos are ordered-dither duotones (graphite + orange) that reveal true colour on hover, focus or tap. Recognisable with content removed: grey field, flat paper rectangles of varied size, mono caps labels, one orange dot.
STORY: The visitor reads the bio, sees the room and clicks its glowing record player to play the set, notices the current book, the latest essay and the last trip, and leaves knowing three specific things about Mathieu plus a way to reach him.
FIRST VIEWPORT: Desktop 12-column bench. Left 5 columns: bio tile, Mathieu's sentence at about 40px Geist medium, role and three link pills at its foot. Right 7 columns: the room tile, the same height as the bio, running the live scene in place (poster first; on phones the scene loads after a tap), mono caption "The room" top-left, and a quiet "Now playing" status in the header while the set plays. No buttons over the render. Below the fold edge, the reading tile (3D book + shelf) and places tile begin. Phone: single column in the order bio, room, now, writing, reading, places, work.
FORM: Widget bench, user-pinned through three prototype rounds (October 2026; Refined Bench 4/5 with Editorial's flatness and dithered room). The rolled direction 14bc1c4d (The Network) was rejected by the user.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Signature interaction
The room tile is the live room in miniature: what glows can be clicked. Clicking an object opens the room large and performs that object's interaction (bookcase → shelves, record player → the set plays, desk → manuscripts); clicking empty space opens the overview. One scene moves between the tile and the dialog, so state is shared; Escape and Close return it to the tile.

## Must stay true
- Without JavaScript or WebGL, every tile is a plain link list with the poster image; no essential content behind hotspots.
- Reduced motion: no flicker, no drift, the dither still reveals on hover/focus.
- A phone loads the 3D room only after a tap (existing behaviour).
- No invented activity: every number and status comes from real data.
