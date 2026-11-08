// Runs before every page of a session opened by `scripts/browser`.

// motion/react springs run on requestAnimationFrame and write inline `style` every frame, so the
// time of the last inline-style write tells whether one is still moving. An infinite loop (a
// spinning record) keeps writing and never settles; wait on its state instead.
let lastStyleWrite = performance.now()
new MutationObserver(() => {
  lastStyleWrite = performance.now()
}).observe(document, { attributes: true, attributeFilter: ["style"], subtree: true })

// `wait --fn "pageSettled()"`: true once web fonts have loaded, every image in the viewport has
// decoded, no finite CSS or WAAPI animation is running, and no inline style changed for 300 ms
// (motion/react springs, including the home's staggered pop-in, have come to rest).
window.pageSettled = () => {
  if (document.readyState !== "complete" || document.fonts.status !== "loaded") return false
  const inView = (el) => {
    const r = el.getBoundingClientRect()
    return r.bottom > 0 && r.top < innerHeight && r.width > 0
  }
  if ([...document.images].some((img) => inView(img) && !(img.complete && img.naturalWidth > 0))) return false
  if (performance.now() - lastStyleWrite < 300) return false
  return !document
    .getAnimations()
    .some((a) => a.playState === "running" && a.effect?.getComputedTiming().iterations !== Infinity)
}

// The dev-only badges stay out of screenshots. Runtime error overlays still render.
addEventListener("DOMContentLoaded", () => {
  const style = document.createElement("style")
  style.textContent = "nextjs-portal, [data-nextjs-dev-tools-button] { display: none !important }"
  document.head.append(style)
})
