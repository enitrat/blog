# Living room assets

`bun run room:bake` generates every file in this folder. Don't edit them by
hand. [Change the living room](../../../docs/room-workflow.md) explains how to
rebuild them. `bun run room:assets` checks them against the tables below.

## GLB files

Each GLB is one bake group. Its surfaces carry baked color and light on an
unlit material. Units are metres, and Y points up.

| File | Contents | What the browser needs from it |
|---|---|---|
| `shell.glb` | Walls, floor, window | Nothing beyond drawing |
| `furniture.glb` | Bookcase, credenza, desk, chairs, sofa, table | Nothing beyond drawing |
| `objects.glb` | Rug, curtains, lamps, turntable base, speakers, desk objects, shelf props | Nothing beyond drawing |
| `moving.glb` | Record platter and tonearm | `Platter` and `Tonearm` nodes, each with its origin on its pivot |
| `spines.glb` | Printed book jackets | One `spines` mesh. `book-slots.json` says which jacket is which book |
| `books.glb` | Book bodies | One `Body_<isbn>` node per shelved book |
| `covers.glb` | Front covers of books with a note | One `Cover_<isbn>` node per note |
| `bookmark.glb` | Cloth bookmark | One `Bookmark` node, origin at the book's head |
| `sheets.glb` | Manuscripts, one per English post | One `Sheet_<slug>` node per post |

`src/components/home/room/scene.ts` loads all nine. If any file fails to load,
the page keeps showing the poster.

Books, covers, the bookmark, and the manuscripts are baked with the rest of the
room hidden, so a surface that moves carries no shadow from where it stood.

## JSON manifests

| File | Read by | Contents |
|---|---|---|
| `book-slots.json` | `room.py`, `Bookshelf.astro`, `shelves.ts` | Each book's ISBN, shelf row, position, width, and height |
| `cabinet.json` | `room.py`, `shelves.ts` | Cabinet bounds, faces, and shelf heights |
| `sheets.json` | `room.py`, `LivingRoom.astro`, `scene.ts` | Each manuscript's slug, position, rotation, and clickable area |

The site build fails if a book in `src/booksData.ts` has no slot in
`book-slots.json`.
