---
name: app-exploration
description:
  Drive this site (or a scratch prototype of it) in a real browser and come back with evidence.
  Use to explore UX, review a design or motion change, verify a change end to end, reproduce a
  visual bug, compare prototype variants, or capture screenshots at phone and desktop widths in
  light and dark mode.
---

# Explore the site

Drive the pages through their visible controls and capture the states the user requested. You
look at every screenshot yourself and judge it; do not write a script that loops over routes and
widths. The reason to use a browser an LLM drives is that you notice what a fixed script would not
(a clipped card, an empty tile, a hover that never settles) and go look closer.

Leave the source repository as you found it unless the user asks for code changes. Evidence goes in
your scratchpad (or `$TMPDIR/site-explore/` without one) and outlives every cleanup.

## Start the server

Measure a `next start` of a finished build, never a shared `next dev` (AGENTS.md). Never build
while another agent edits the tree.

```sh
bun run build
bunx next start -p <port> &
```

Pick a port nobody holds (`lsof -ti:<port>` prints nothing). For a scratch prototype, serve its
folder statically instead (`python3 -m http.server <port>` from that folder) and skip the build.

## Drive the browser and capture

Read [browser.md](references/browser.md) for selectors, waits, screenshots, and shell gotchas.
Open your own session for each state you capture:

```sh
mkdir -p <scratchpad>/evidence
.claude/skills/app-exploration/scripts/browser <prefix>                  # 1440x900 @2x, light
.claude/skills/app-exploration/scripts/browser <prefix>-m 390 844 3 dark  # phone, dark
.claude/skills/app-exploration/scripts/browser <prefix>-rm 1440 900 2 light reduce
```

```sh
ab() { agent-browser --session <name> "$@"; }
ab open "http://localhost:<port>/<route>"
ab wait --fn "pageSettled()"
ab snapshot -i -c
ab screenshot "<absolute-evidence-path>/home-desktop-light.png"
```

The script prints the session name for `<name>`. The theme follows the emulated color scheme,
because next-themes defaults to `system`; clicking the theme toggle stores a choice in
`localStorage.theme` that overrides it for the rest of the session.

The states this site most needs looked at:

- Every route at 1440 and 390, light and dark (AGENTS.md: a visual change is done only when the
  changed pages have been looked at in all four).
- Home at 1024 and 1280 too: the bench must fit one viewport from 1024 up without scrolling.
- Hover and press on cards, the deck fan, the record player, the nav highlight.
- Reduced motion: state still changes, nothing springs.
- An essay mid-scroll, and a nonexistent URL.

## Finish

Close every session you opened (`agent-browser --session <name> close`) and kill the server by
port: `next start` leaves a `next-server` child running, so `lsof -ti:<port> | xargs kill`.
Confirm `git status` lists no file you created outside the request.

Return the screenshots with a short description of each state and the actions that reached it.
Name anything requested that you could not reach, and whether cleanup completed.
