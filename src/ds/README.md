# `src/ds`: the design system

Every value the site draws with is a token. It is stated once, in this folder. Pages and site
components (`src/components/**`) compose what this folder exports, and they read only roles.

The site is Next.js (App Router, static export into `dist/`), React 19, Tailwind CSS v4, and
shadcn components on Base UI (`components.json`, style `base-nova`). `shadcn add <name>` writes
into `src/ds/components/ui/`. Restyle what it writes onto the tokens before using it. The
restyling rules are in [Add a component](#add-a-component).

## Layers

Layers run one way. A layer reads only the layer above it.

| Layer                  | File                                                        | Holds                                                                                                                                                                   | Read by                                                 |
| ---------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Primitives             | `tokens/primitives.css`                                     | raw colours (`--grey-*`, `--orange-*`, `--sand-200`), raw shadows, the room's art colours                                                                               | `semantic.css`, `elevation.css`, `room.tokens.css` only |
| Semantic roles         | `tokens/semantic.css`, `surfaces.css`, `elevation.css`      | `--background`, `--surface`, `--foreground`, `--muted-foreground`, `--border`, `--accent`, `--accent-graphic`, `--ring`, `--scrim`, the `--elevation-*`, light and dark | component tokens and classes                            |
| Scales                 | `radius.css`, `spacing.css`, `typography.css`, `motion.css` | Tailwind `@theme` steps. The defaults are reset, so only these compile.                                                                                                 | classes                                                 |
| Layout roles           | `layout.css`                                                | `--page-max`, `--measure`, `--gap-bench`, `--pad-tile`, `--radius-tile`, `--target-min`, control heights, z-index                                                       | component tokens and classes                            |
| Component tokens       | `components/**/<name>.tokens.css`                           | `--<component>[-<part>]-<property>[-<state>]`                                                                                                                           | that component's classes                                |
| Variants and utilities | `variants.css`, `utilities.css`                             | `hovered:`, `pressed:`, `focused:`, `open:`, `inverse:`, `dark:`; `focus-ring`, `hit-area`, `press`, `edge`, `underline-accent`, `type-*`                               | everything                                              |

## Rules

- **Every value is a token, stated once.** A class reads a component token (`bg-(--button-primary-bg)`)
  or a role utility (`bg-surface`, `text-muted-foreground`, `shadow-tile`, `rounded-pill`).
  Spacing utilities (`gap-3`, `p-4`) read the 4px scale and count as tokens. Do not write hex,
  `px`, or `ms` literals in a class or a style prop. Do not write opacity shades (`bg-foreground/10`).
  Do not write arbitrary values (`p-[13px]`) where a token exists. If no token exists, add one.
- **A component token points at a role**, or at a `color-mix()` of roles. It never points at a
  primitive. It is declared on the component's own element, wrapped in `@layer base` and
  `:where(...)`, so it resolves against the surface it sits on, and a class can still override it.
- **Atoms never take an inverse prop.** `data-surface="inverse"` (or a `Tile` with
  `variant="inverse"` or `"media"`) re-maps the paper names once, in `surfaces.css`. Write against
  the paper names. An inverse fill is `bg-foreground text-surface`, and it flips by itself.
- **Text uses a role.** Use `type-title`, `type-heading`, `type-body`, `type-ui`, `type-control`,
  `type-small`, `type-label` (mono caps), or `type-data` (mono digits). Do not set a font size
  directly.
- **Orange is state.** It means playing, current, latest, pressed, or focus. Small marks (dots,
  underlines) use `--accent-graphic`. Fills use `--accent`.
- **Objects cast shadows; only links lift.** Tiles use `shadow-tile`. Objects on tiles use
  `shadow-object`. A linked tile lifts to `shadow-tile-lift` by `--lift-distance`. Do not add glass,
  blur, or gradients. The room is the one exception.
- **Keyboard, focus, dismiss, and scroll lock come from Base UI.** Do not hand-roll them.
- **A second look is a new key in the component's `const` map** (`VARIANTS`, `SIZES`, `TONES`).
  Do not add a boolean prop for it, and do not copy the component. A shared effect is a
  utility in `utilities.css`.
- **Use the state variants.** Write `hovered:` (real hover pointers only), `pressed:`,
  `focused:`, and `open:`. Do not write bare `hover:` or `active:`. A focus ring is `focus-ring`.
- **Every interactive target reaches 44px.** Small controls add `hit-area` (both axes) or
  `hit-area-y` (height only).
- **Motion.** CSS reads `duration-fast` (state), `duration-base` (hover lift, every exit),
  `duration-moderate` with `ease-spring` (an object opening), and `duration-slow`. Scripts take a
  spring from `src/lib/motion.ts` and never inline one. Animate transform, opacity, and
  clip-path only. Under reduced motion, nothing translates or spins. State still changes.
- **Markup the browser checks query stays stable.** Keep the class and data hooks that
  `scripts/check-*.mjs` and the client controllers select on, such as `.tile`, `.room-tile`,
  `[data-deck]`, `[data-glide-item]`, and `[data-readout]`. Tailwind classes sit beside them.

## Client behaviour

Pages are React Server Components. Each interactive object keeps its framework-free controller
(`glide-controller.ts`, `motion/pointer.ts`, and the deck, readout, and room scripts). The
controller exports `init(root): () => void`. A small `'use client'` component renders the markup
and calls the controller from `useEffect`. The markup stays server-rendered and works without JS.
The behaviour is the same as before the port.

## Where to look

| Task                                                   | Read first                                | Edit                                                                                                |
| ------------------------------------------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Change one component's look                            | `<name>.tokens.css`                       | `<name>.tokens.css`                                                                                 |
| Change a colour everywhere                             | `tokens/semantic.css`                     | the role, in light and in both dark blocks                                                          |
| Change spacing, radius, or a control height everywhere | `tokens/layout.css`                       | the role                                                                                            |
| Change a text role                                     | `tokens/typography.css`                   | the `--text-<role>` group, or the `type-*` utility                                                  |
| Add a variant or size                                  | the component's `const` maps              | the map and its tokens                                                                              |
| Add a component                                        | an existing component in `components/ui/` | `<name>.tsx`, `<name>.tokens.css`, an `@import` in `tokens/components.css`, an export in `index.ts` |
| Change motion                                          | `src/lib/motion.ts`, `tokens/motion.css`  | the same files                                                                                      |

## Add a component

1. Look for an existing component first. If one is close, add a variant to it.
2. Run `bunx shadcn add <name>` to get the Base UI implementation.
3. Restyle what it wrote. Replace `cva` with `const` maps. Move every value into
   `<name>.tokens.css`. Replace `hover:` with `hovered:`. Remove opacity shades and blur.
4. Add the `@import` to `tokens/components.css` and the export to `index.ts`.
5. Show it on `/system`.

## Components

`components/ui/`: `Button`, `Pill` and `PillToggle`, `Link`, `Label`, `Tag`, `StatusDot`,
`Icon`, `Rule`, `Card`, `Dialog`, `Tile` and `Bench`, `Page` and `PageTitle`, `ListRow` and
`ListRows`, `Glide`.

`components/motion/`: `Tilt`, `Magnetic`, `Equalizer`, and the shared `pointer.ts`.
