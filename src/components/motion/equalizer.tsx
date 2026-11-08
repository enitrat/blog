'use client';

import { motion, useReducedMotion } from 'motion/react';
import { cn } from '@/lib/utils';

/** Resting height of each bar, and how long one beat takes, s. */
const BARS = [
	{ rest: 0.55, beat: 0.6 },
	{ rest: 1, beat: 0.45 },
	{ rest: 0.4, beat: 0.75 },
	{ rest: 0.75, beat: 0.52 },
];

/** Four bars that bounce while `playing`. Decorative: the control beside it announces the state. Still under reduced motion. */
export function Equalizer({ playing, className }: { playing: boolean; className?: string }) {
	const still = useReducedMotion() || !playing;
	return (
		<span
			aria-hidden="true"
			className={cn(
				'inline-flex h-3 flex-none items-end gap-0.5 transition-colors',
				playing ? 'text-brand-500 dark:text-brand-400' : 'text-gray-400 dark:text-iron-500',
				className,
			)}
		>
			{BARS.map(({ rest, beat }) => (
				<motion.i
					key={beat}
					className="block h-full w-0.5 origin-bottom rounded-full bg-current"
					initial={{ scaleY: rest }}
					animate={still ? { scaleY: rest } : { scaleY: [0.2, 1] }}
					transition={
						still
							? { duration: 0.2 }
							: {
									duration: beat,
									repeat: Number.POSITIVE_INFINITY,
									repeatType: 'reverse',
									ease: 'easeInOut',
								}
					}
				/>
			))}
		</span>
	);
}
