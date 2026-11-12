# Navbar: Masthead

Decided 12 November 2026, from a live picker of five headers on the real pages, in both themes.

## Chosen

- **Masthead**: plain text links in `type-control`, no container. The current page takes the
  foreground colour and a primary underline that slides to the next page on navigation
  (`spring.layout`, no motion under reduced motion).
- Header **54px** (`--spacing-header`, 3.375rem), pinned from `sm` up. On a phone it takes two
  rows (name and toggle, then the links) and scrolls away.
- Link gap **16px**. Underline **2.5px**, **4px** below the text.
- A **1px `border` rule** along the header's bottom edge, the full width of the window, separating
  it from the page the way the rule beside home's sidebar does. On home the two meet.
- The theme toggle is a ghost icon button, no glass.
- Every link is 44px tall (`min-h-11`), on every width.

## Where it lives

- `src/components/site/header.tsx` renders it; `src/lib/nav.ts` holds the one link list.
- `src/app/layout.tsx` puts the header outside the site container, so its rule spans the window;
  its content sits in the container (`max-w-wide`).
- Home's sidebar pins under it (`top-header`) and the bento's rows subtract `--spacing-header`.

## Rejected

- **The glass segmented pill** (the previous header): a translucent pill with a border, shadow
  and blur, and a second glass circle for the toggle. It read as product chrome on a personal
  site, was the loudest thing on screen while reading an essay, took 116px on a phone, and its
  links were 32px tall on tablets.
- **Index**: the same links in Fragment Mono, numbered 01–05 like "In short", a dot for the
  current page. Distinctive, but the numbers were new copy and the mono labels read as a terminal.
- **Wayfinder**: one row everywhere, "Mathieu Saugier / Writing", the section name opening a menu
  of all sections. It said where you are best and was the shortest on a phone (56px), but it hid
  the sections behind a click and needed a label for home.
- **Dock**: the home tiles' bezel and well as a tray, the current page a raised key; at the bottom
  of the screen on a phone. It belonged to the home bench, but it was the heaviest option, and a
  bottom tab bar is an app pattern on a reading site.
- **No separator, with a fade under the header**: the fade washed out any rule, so the header did
  not read as separate from the page.
