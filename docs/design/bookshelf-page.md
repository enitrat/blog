# The bookshelf page

`/bookshelf/` shows Mathieu's whole reading archive as an oak bookcase of
Pléiade volumes, drawn with Three.js. Hover a spine and the book slides part
way out. Select it and the book rotates to face you. A book with a note then
opens into pages you can turn, with the bookcase still behind it. Closing the
book returns it to its slot.

Without JavaScript or WebGL, the same page shows the archive as a plain
reading list instead.

This page is separate from the homepage's living room. The living room is
baked in Blender. This bookcase is painted in the browser.

## Run and check it

```sh
bun run library                     # opens /bookshelf/ in a dev server
bun test                            # shelf packing
bun run library:check               # every engine and viewport
bun run library:check webkit 390    # one case, while you iterate
```

`library:check` starts and stops its own dev server. To use one you already
run, set `DEV_URL`. A failure names the engine, the viewport, and the step,
and saves a `FAIL-*.png` in `/tmp/library-check`.

The checks use the notes in `src/content/notes/`, so at least one note must
exist.

`playwright` is pinned to an exact version. Its downloaded browsers only work
with the version that downloaded them, so a caret range would break them on
the next minor update.

## Decisions

- **Two canvases.** The selected book flies across a second, transparent
  canvas above the dialog. On the shelf canvas, the canvas edge would cut it
  off.
- **The reader is an iframe.** StPageFlip 2.0.7, the page-turning library,
  keeps scheduling animation frames after its `destroy()` call. Removing the
  iframe is the only thing that stops it.
- **Drawing happens on demand.** The shelf redraws only when something moves,
  not every frame.
- **Spines are painted, not lit.** They use `MeshBasicMaterial` with canvas
  textures. Tone mapping only desaturated the leather and dulled the gilt.
- **Texture size follows display size.** A spine draws about 130 CSS pixels
  wide, so its texture is 256 pixels. At 512 pixels, 55 books used four times
  the GPU memory and looked the same.

## Why the checks run the way they do

- **Each case gets its own browser.** Sharing one browser per engine put about
  28 WebGL contexts in one process. Past WebKit's limit per process, WebKit
  does not raise an error. It stalls, sometimes for more than 20 minutes.
- **Two cases run at a time.** With three, a starved page makes WebKit slow
  its animation frames, and the closing animation stalls before the dialog
  finishes. A real background tab catches up when it returns, because motion
  runs off timestamps.
- **Every case has a time limit,** per action and overall. A check that can
  hang forever is not a check.

## Open problems

- The WebKit cases fail at "hand over a reading record for a book with no
  note" (East of Eden). The Chromium cases pass.
- Nobody has measured frame rate on a real phone. Viewport emulation can't.

## Credits

- [StPageFlip](https://github.com/Nodlik/StPageFlip), MIT licence, turns the
  pages.
- [The Complete Shelf](https://mengto.github.io/complete-shelf/) inspired the
  selection sequence. We found no reuse licence, so no code or assets were
  copied.
