# Living room assets

`bun run room:bake` generates every file here, following
[Change the living room](../../../docs/room-workflow.md). Never edit them by
hand. `bun run room:assets` checks them against these tables.

## GLB files

Each GLB is one unlit bake group with light baked in, in metres, Y up.
`src/components/home/room/scene.ts` loads all nine. If one fails, the poster
stays.

| `<group>.glb` | Contents | Required nodes |
|---|---|---|
| `shell` | Walls, floor, window | None |
| `furniture` | Bookcase, credenza, desk, chairs, sofa, table | None |
| `objects` | Rug, curtains, lamps, turntable base, speakers, desk objects, shelf props | None |
| `moving` | Record platter, tonearm | `Platter`, `Tonearm`, origins on pivots |
| `spines` | Printed jackets | `Book_<isbn>` per shelved book |
| `books` | Bodies, one atlas cell each | `Body_<isbn>` per shelved book |
| `covers` | Covers of noted books | `Cover_<isbn>` per note |
| `bookmark` | Cloth bookmark | `Bookmark`, origin at the book's head |
| `sheets` | One manuscript per English post | `Sheet_<slug>` per post |

- `shell`, `furniture`, and `objects` use Meshopt, which `scene.ts` decodes.
- `books`, `covers`, `bookmark`, and `sheets` bake with the room hidden, so
  they carry no shadow from where they rest.

## JSON manifests

| File | Read by | Contents |
|---|---|---|
| `book-slots.json` | `room.py`, `Bookshelf.astro`, `shelves.ts` | Per book: ISBN, shelf row, position, width, height |
| `cabinet.json` | `room.py`, `shelves.ts` | Bounds, faces, and shelf heights |
| `sheets.json` | `room.py`, `LivingRoom.astro`, `scene.ts` | Per manuscript: slug, position, rotation, clickable area |

The build fails if a `src/booksData.ts` book has no `book-slots.json` slot.
