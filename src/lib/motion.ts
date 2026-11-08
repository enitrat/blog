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
	/** Drop-in from above. */
	drop: { type: 'spring', stiffness: 250, damping: 18, delay: 0.05 },
	/** 3D lid or folder. */
	lid: { type: 'spring', stiffness: 400, damping: 25 },
	/** Layout shifts, such as the navbar resizing. */
	layout: { type: 'spring', stiffness: 400, damping: 40 },
	/** Photo fan on group hover; stagger by `fanStagger * index`. */
	fan: { type: 'spring', stiffness: 320, damping: 18 },
} as const satisfies Record<string, Transition>;

export const fanStagger = 0.0335;

/** Fallback tween. */
export const ease = { duration: 0.2, ease: [0.4, 0, 0.2, 1] } as const satisfies Transition;

/** Hover scale: big surfaces and small ones. */
export const hoverScale = { surface: 1.02, small: 1.08 } as const;

/** Exit for popped-in elements. */
export const exit = { y: -30, scale: 0.9, opacity: 0 } as const;
