'use client';

import { DumbbellIcon } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { Card, CardHeader } from '@/components/ui/card';
import { spring } from '@/lib/motion';

/** Mathieu's bench as a ring filling toward the goal. The page renders it only when `bench` is set. */
export function BenchTile({
	current,
	goal,
	index,
	className,
}: {
	current: number;
	goal: number;
	index: number;
	className?: string;
}) {
	const reduce = useReducedMotion();
	const filled = Math.min(current / goal, 1) * 100;
	return (
		<Card index={index} labelledBy="bench-title" className={className}>
			<CardHeader id="bench-title" icon={<DumbbellIcon />} label="Bench" />
			<div className="mt-3 flex min-h-0 flex-1 items-center justify-center">
				<div className="relative aspect-square h-full max-h-[110px]">
					<svg viewBox="0 0 100 100" aria-hidden className="size-full -rotate-90 fill-none">
						<circle
							cx="50"
							cy="50"
							r="42"
							pathLength="100"
							strokeWidth="9"
							className="stroke-muted"
						/>
						<motion.circle
							cx="50"
							cy="50"
							r="42"
							pathLength="100"
							strokeWidth="9"
							strokeLinecap="round"
							strokeDasharray="100 100"
							className="stroke-primary"
							initial={{ strokeDashoffset: reduce ? 100 - filled : 100 }}
							animate={{ strokeDashoffset: 100 - filled }}
							transition={reduce ? { duration: 0 } : { ...spring.fan, delay: 0.4 }}
						/>
					</svg>
					<p className="absolute inset-0 grid place-content-center text-center">
						<span className="font-display text-3xl leading-[0.9]">{current}</span>
						<span className="text-xs text-faint">of {goal} kg</span>
					</p>
				</div>
			</div>
		</Card>
	);
}
