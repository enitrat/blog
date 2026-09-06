# The bookcase at /bookshelf/

An oak case of Pléiade volumes. Hovering draws a book part-way out; selecting it
rotates the cover forward and opens it into turning pages, with the library still
behind. Closing returns it to its slot. `/bookshelf/archive/` is the same catalogue
as an ordinary list, and is the whole experience without JavaScript.

## Try it

```sh
bun install
bun run library                      # opens /bookshelf/
bun test                             # shelf packing
bun run library:check                # 7 engine/viewport cases, ~30s
bun run library:check webkit 390     # just one, while iterating
```

`bunx playwright install chromium webkit` first. `library:check` starts and stops
its own dev server; set `LIBRARY_URL` to point it at one you are already running.
Failures name the engine, the viewport and the step, and leave a `FAIL-*.png`.

`playwright` is pinned exactly, not to a caret range: its browser binaries are
revision-locked to the library version, so a floating minor silently invalidates
the browsers you have downloaded.

## Decisions worth remembering

- **Two renderers.** The selected book flies across a second, transparent canvas
  over the dialog so it is not clipped by the shelf canvas's bounds.
- **The reader is an iframe.** StPageFlip 2.0.7 keeps scheduling animation frames
  after its documented `destroy()`. Removing the iframe is what actually stops it.
- **The shelf renders on demand**, not every frame. Only motion schedules a draw.
- **Spines are painted, not lit.** They are `MeshBasicMaterial` with canvas
  textures; tone mapping only desaturated the leather and washed out the gilt.
- **Texture sizes track display size.** A spine draws ~130 CSS px wide, so its
  texture is 256 px — 55 books at 512 was four times the GPU memory for no gain.
- **One palette.** `PLEIADE_HEX` in `src/utils/pleiade.ts` feeds both the CSS
  spines and the painted 3D ones. They drifted when it was duplicated.

## Verification notes

- **One browser per case.** Sharing one per engine piled ~28 WebGL contexts into
  a single process. Past WebKit's per-process limit it does not error, it blocks:
  20+ minute stalls with no request leaving the page.
- **Two cases at a time.** At three, the machine starves a page enough that
  WebKit throttles its rAF, stalling the return animation before the dialog
  reaches its terminal phase. Starvation, not a defect — motion runs off
  absolute timestamps, so a real backgrounded tab catches up on resume.
- **Every case is bounded**, per action and overall. A check that can hang
  forever is not a check.

## Still open

- Real authored notes. The pages are labelled samples with fixed breaks; paginating
  arbitrary Markdown is not solved.
- Phone GPU performance. Viewport emulation does not establish frame rate.
- Resizing while reading closes the book and reframes the case.

## References

- [StPageFlip](https://github.com/Nodlik/StPageFlip), MIT, supplies the page mechanics.
- [Three.js rendering on demand](https://threejs.org/manual/en/rendering-on-demand.html).
- [The Complete Shelf](https://mengto.github.io/complete-shelf/) inspired the selection
  sequence. No code or assets were copied; no reuse licence was found.
