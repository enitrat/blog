'use client';

import { motion, useReducedMotion } from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import Link from 'next/link';
import { hoverScale, spring } from '@/lib/motion';

/**
 * One essay: date, title and description, and the banner as a thumbnail when the post has one.
 * Hover lifts the row and zooms the banner on a spring; under reduced motion nothing moves.
 * The thumbnail slot keeps its width without a banner, so every row lines up.
 */
export function WritingRow({
	href,
	date,
	datetime,
	title,
	description,
	banner,
}: {
	href: string;
	date: string;
	datetime: string;
	title: string;
	description: string;
	banner?: StaticImageData;
}) {
	const motile = !useReducedMotion();
	return (
		<motion.li
			className="border-t border-border first:border-t-0"
			initial="rest"
			animate="rest"
			whileHover={motile ? 'hover' : undefined}
			variants={{ rest: { y: 0 }, hover: { y: -2 } }}
			transition={spring.hover}
		>
			<Link
				href={href}
				className="group -mx-3 flex flex-col gap-3 rounded-xl px-3 py-5 outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring sm:flex-row sm:items-center sm:gap-8"
			>
				<time
					dateTime={datetime}
					className="font-mono text-xs tracking-[-0.01em] text-faint tabular-nums sm:w-24 sm:flex-none"
				>
					{date}
				</time>
				<div className="min-w-0 flex-1">
					<h2 className="max-w-[44ch] text-xl leading-snug font-semibold tracking-tight text-balance text-foreground">
						{title}
					</h2>
					<p className="mt-1.5 text-sm leading-relaxed text-pretty text-muted-foreground">
						{description}
					</p>
				</div>
				<div
					className={
						banner ? 'sm:h-35 sm:w-56 sm:flex-none' : 'max-sm:hidden sm:h-35 sm:w-56 sm:flex-none'
					}
				>
					{banner && (
						<div className="aspect-[16/10] overflow-hidden sm:aspect-auto sm:h-full rounded-media bg-muted shadow-card">
							<motion.div
								className="size-full"
								variants={{ rest: { scale: 1 }, hover: { scale: hoverScale.surface * 1.03 } }}
								transition={spring.hover}
							>
								<Image
									src={banner}
									alt=""
									sizes="(min-width: 40rem) 224px, 100vw"
									className="size-full object-cover"
								/>
							</motion.div>
						</div>
					)}
				</div>
			</Link>
		</motion.li>
	);
}
