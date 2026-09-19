# Living room asset reference

The browser loads nine GLBs and three JSON manifests from this directory. The
[room workflow](../../../docs/room-workflow.md) explains how to change and
rebuild them.

## Ownership

```text
scripts/
├── room.py                 # Blender geometry, materials, lights, bake, export
├── bake-room.mjs           # content input, layout, artwork, Blender process
├── spine-art.mjs           # jacket and cover artwork
└── sheet-art.mjs           # manuscript artwork

src/assets/room/
├── *.glb                   # baked browser assets
├── book-slots.json         # book positions and dimensions
├── cabinet.json            # bookcase dimensions
└── sheets.json             # manuscript positions and link bounds

src/components/home/
├── LivingRoom.astro        # room host, poster, and manuscript links
└── room/
    ├── Bookshelf.astro     # controls, notes, and HTML fallback
    ├── scene.ts            # Three.js loading, camera, and object movement
    ├── bookshelf.ts        # navigation, focus, picking, and notes
    └── shelves.ts          # frames and bounds derived from the manifests
```

`scripts/room.py` and `scripts/bake-room.mjs` are the source of truth. A live
Blender scene is a disposable preview.

## GLB files

| File | Contents | Runtime contract |
|---|---|---|
| `shell.glb` | Walls, floor, and fixed room structure | One baked, unlit group |
| `furniture.glb` | Cabinet, desk, chairs, and other furniture | Fixed geometry |
| `objects.glb` | Lamps, desk objects, record player base, and fixed props | Fixed geometry |
| `moving.glb` | Platter and tonearm | Separate nodes with usable pivot origins |
| `spines.glb` | Printed book jackets | One shared atlas. `book-slots.json` owns identity |
| `books.glb` | Book bodies | One `Body_<isbn>` node per book |
| `covers.glb` | Front covers for books with notes | One `Cover_<isbn>` node per available note |
| `bookmark.glb` | Reusable cloth bookmark | One `Bookmark` node with its origin at the book head |
| `sheets.glb` | Published English writing | One `Sheet_<slug>` node per piece |

`scene.ts` loads all nine files. A failed load keeps the poster fallback in
place.

## JSON manifests

`bake-room.mjs` writes every manifest on each run, including `--layout-only`.

| File | Consumers | Contents |
|---|---|---|
| `book-slots.json` | Blender, `Bookshelf.astro`, `shelves.ts` | ISBN, row, position, width, and height |
| `cabinet.json` | Blender and `shelves.ts` | Cabinet bounds, faces, and shelf heights |
| `sheets.json` | `LivingRoom.astro` and `scene.ts` | Slug, position, rotation, and visible link bounds |

The generator rejects a book that does not fit in the cabinet. The site build
rejects a missing ISBN.

## Bake settings

| Setting | Value |
|---|---|
| Blender | 4.5.3 |
| Units | Metres |
| Source axes | Blender Z-up |
| Renderer | Cycles |
| Default samples | 256 |
| Runtime material | Unlit glTF material with baked color and light |

The default atlas sizes are:

| Group | Size |
|---|---:|
| Shell and furniture | 2048 px |
| Objects, spines, and sheets | 4096 px |
| Books | 2048 px |
| Covers | 2048 to 4096 px, based on the cover count |
| Moving parts and bookmark | 1024 px |

Books, the bookmark, and manuscript sheets bake in isolation. This prevents
nearby meshes from adding permanent occlusion to movable surfaces. Source
artwork UVs stay separate from bake UVs.

## Content changes

| Change | Required output |
|---|---|
| Add or reorder a book | Full bake and all manifests |
| Add a note | `books.glb` and `covers.glb` |
| Change reading status | No bake. The runtime reuses `bookmark.glb` |
| Publish English writing | `sheets.glb` and `sheets.json` |
| Change cabinet dimensions | Full bake and all manifests |

Vite fingerprints the GLBs. Astro generates the poster formats.
