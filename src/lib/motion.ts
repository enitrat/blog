import type { Transition } from 'motion/react';

/** Spring presets. Stiffness 250–550, damping 12–25 playful and 32–40 settled, bounce .1–.25. */
export const spring = {
	/** Hover on cards and media: scale 1.02. */
	hover: { type: 'spring', stiffness: 450, damping: 18, bounce: 0.1 },
	/** Hover on icons and avatars: scale 1.08–1.1. */
	pop: { type: 'spring', stiffness: 420, damping: 16, bounce: 0.2 },
	/** Tap: scale .95. */
	tap: { type: 'spring', stiffness: 200, damping: 12, duration: 0.2 },
	/** Pop-in and exit of messages and cards. */
	enter: { type: 'spring', stiffness: 450, damping: 32, bounce: 0.15 },
	/** Layout shifts, such as the navbar resizing. */
	layout: { type: 'spring', stiffness: 400, damping: 40 },
	/** A tile's object waking: covers fanning, sheets rising, icons rippling. */
	fan: { type: 'spring', stiffness: 320, damping: 18 },
} as const satisfies Record<string, Transition>;

/** Fallback tween. */
export const ease = { duration: 0.2, ease: [0.4, 0, 0.2, 1] } as const satisfies Transition;
