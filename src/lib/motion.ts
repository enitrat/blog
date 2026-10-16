/**
 * Motion vocabulary for client scripts, ported from beUI's `lib/ease.ts` so names and feel match
 * the exploration libraries. CSS keeps `--duration-*` / `--ease-*`; a script that needs a spring
 * (interruptible, velocity-carrying motion) takes one of these presets and never inlines its own.
 */
import type { AnimationOptions } from 'motion';

/** Press feedback: buttons, pills, the deck's tapped card. */
export const SPRING_PRESS = { type: 'spring', stiffness: 500, damping: 30, mass: 0.6 } as const;
/** An object changing place: a deck card dealing out, a cover fanning. */
export const SPRING_LAYOUT = { type: 'spring', stiffness: 360, damping: 32, mass: 0.6 } as const;
/** ana.sh photo pile: the front card dealt to the back, with a little overshoot. */
export const SPRING_DEAL = { type: 'spring', stiffness: 320, damping: 18, mass: 1 } as const;
/** Seconds a dealt card needs to cross the pile before it may drop behind it (a spring has no fixed duration). */
export const DEAL_CROSS = 0.22;
/** Seconds between neighbouring items entering (a deck fanning open). Exits do not stagger. */
export const STAGGER = 0.04;
/** Pointer-follow: tilt and magnetic pull. Soft, so it trails the cursor instead of snapping. */
export const SPRING_MOUSE = { type: 'spring', stiffness: 200, damping: 15, mass: 0.3 } as const;
/** Critically damped glide for indicators. */
export const SPRING_GLIDE = { type: 'spring', stiffness: 700, damping: 50, mass: 0.5 } as const;
/** A record's turning speed: critically damped and slow, so play spins up and pause coasts to rest (about 1.5s). */
export const SPRING_RECORD = { type: 'spring', stiffness: 30, damping: 11, mass: 1 } as const;

/** Set down: a short spring with a light overshoot, so a returning object lands instead of stopping. */
export const SPRING_SETTLE = { type: 'spring', stiffness: 420, damping: 15, mass: 0.6 } as const;
/** Seconds a deck takes to gather before its one knock on the table. */
export const GATHER = 0.12;
/** Readout roll: critically damped, text must never bounce. */
export const SPRING_ROLL = { type: 'spring', stiffness: 520, damping: 40, mass: 0.6 } as const;
/** Readout under reduced motion: a plain crossfade. */
export const ROLL_FADE = { duration: 0.15, ease: 'easeOut' } as const;

/** Instant: what every spring collapses to under reduced motion. */
export const INSTANT: AnimationOptions = { duration: 0 };

const query = (q: string) => typeof matchMedia === 'function' && matchMedia(q).matches;

/** A real hover pointer. Touch fires phantom hovers on tap, so pointer effects check this first. */
export const canHover = () => query('(hover: hover) and (pointer: fine)');

/** Reduced motion keeps opacity and colour, drops movement. */
export const prefersReducedMotion = () => query('(prefers-reduced-motion: reduce)');

/** The spring to use for `preset`, or `INSTANT` under reduced motion. */
export const transition = (preset: AnimationOptions): AnimationOptions =>
	prefersReducedMotion() ? INSTANT : preset;
