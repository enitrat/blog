'use client';

import { ChevronRightIcon, PauseIcon, PlayIcon } from 'lucide-react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { ease, spring } from '@/lib/motion';
import { cn } from '@/lib/utils';

export type CarouselPost = {
	title: string;
	description: string;
	date: string;
	iso: string;
	href: string;
	banner?: StaticImageData;
};

/** Seconds each post stays before the next one. */
const DWELL = 6;
const FOCUS = 'outline-none focus-visible:ring-2 focus-visible:ring-ring';

/**
 * The latest posts, one at a time: date, title, summary and Read more on the left, the banner
 * fading into the well on the right. Read more stretches over the slide, so the whole slide opens
 * the post; while the tile is held the title takes the accent and the banner slowly pushes in. It advances on its own, holds while the pointer or focus is
 * inside, and never advances under reduced motion. The pause button stops it for good, which
 * is the only way to on touch, where nothing hovers; the dots pick a post directly. Both sit in a
 * row under the slide.
 */
export function WritingCarousel({ posts }: { posts: CarouselPost[] }) {
	const [index, setIndex] = useState(0);
	const [held, setHeld] = useState(false);
	const [paused, setPaused] = useState(false);
	const reduce = useReducedMotion();
	const rotates = !reduce && posts.length > 1;
	// biome-ignore lint/correctness/useExhaustiveDependencies: `index` restarts the dwell after a pick.
	useEffect(() => {
		if (held || paused || !rotates) return;
		const timer = setTimeout(() => setIndex((n) => (n + 1) % posts.length), DWELL * 1000);
		return () => clearTimeout(timer);
	}, [index, held, paused, rotates, posts.length]);
	const post = posts[index];
	return (
		<section
			aria-roledescription="carousel"
			aria-label="Latest writing"
			className="relative mt-3 flex min-h-0 flex-1 flex-col"
			aria-live={held || paused || !rotates ? 'polite' : 'off'}
			// A tap fires enter and leave together, so only a mouse holds the carousel.
			onPointerEnter={(e) => e.pointerType === 'mouse' && setHeld(true)}
			onPointerLeave={(e) => e.pointerType === 'mouse' && setHeld(false)}
			onFocus={() => setHeld(true)}
			onBlur={(e) => {
				if (!e.currentTarget.contains(e.relatedTarget)) setHeld(false);
			}}
		>
			<AnimatePresence mode="popLayout" initial={false}>
				<motion.article
					key={post.href}
					aria-roledescription="slide"
					aria-label={`${index + 1} of ${posts.length}`}
					className="grid min-h-0 flex-1 grid-cols-1 gap-4 sm:grid-cols-[1.25fr_1fr]"
					initial={{ opacity: 0, x: 16 }}
					animate={{ opacity: 1, x: 0 }}
					exit={{ opacity: 0, x: -16, transition: reduce ? { duration: 0 } : ease }}
					transition={reduce ? { duration: 0 } : spring.enter}
				>
					{/* Anchored to the bottom like every tile's title; `safe` keeps the date reachable when
					    a long title fills the column. */}
					<div className="flex min-h-0 flex-col justify-end-safe">
						<time dateTime={post.iso} className="type-meta text-faint">
							{post.date}
						</time>
						<h3 className="mt-1.5 type-title text-balance transition-colors duration-150 ease-out group-data-[held]/tile:text-primary">
							{post.title}
						</h3>
						<p className="mt-2 line-clamp-3 type-body text-pretty text-muted-foreground">
							{post.description}
						</p>
						<Link
							href={post.href}
							className={`mt-4 inline-flex min-h-10 items-center gap-0.5 self-start rounded-control bg-card pr-2.5 pl-3.5 type-control shadow-card transition-colors duration-150 ease-out after:absolute after:inset-0 group-data-[held]/tile:text-primary ${FOCUS}`}
						>
							Read more
							<ChevronRightIcon
								aria-hidden
								className="size-4 transition-transform duration-150 ease-out group-data-[held]/tile:translate-x-0.5 motion-reduce:transition-none"
							/>
						</Link>
					</div>
					{post.banner && (
						<div className="relative min-h-40 overflow-hidden rounded-media fade-inline-start max-sm:hidden">
							<Image
								src={post.banner}
								alt=""
								fill
								sizes="(min-width: 64rem) 25vw, 50vw"
								className="object-cover transition-transform duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] group-data-[held]/tile:scale-[1.04] motion-reduce:transition-none"
							/>
						</div>
					)}
				</motion.article>
			</AnimatePresence>
			{posts.length > 1 && (
				// A row of its own under the slide: beside Read more it ran under the banner when the tile
				// was narrow. Phones hide the banner and the tile is one row short, so there it goes back
				// beside Read more.
				<div className="-mb-1.5 -ml-2.5 flex pt-2 max-sm:absolute max-sm:right-0 max-sm:bottom-1 max-sm:m-0 max-sm:pt-0">
					{rotates && (
						<button
							type="button"
							aria-label={paused ? 'Play slideshow' : 'Pause slideshow'}
							onClick={() => setPaused((p) => !p)}
							className={`relative grid size-8 cursor-pointer place-items-center rounded-full text-faint transition-colors after:absolute after:-inset-1.5 hover:text-foreground ${FOCUS}`}
						>
							{paused ? (
								<PlayIcon aria-hidden className="size-3.5 fill-current stroke-none" />
							) : (
								<PauseIcon aria-hidden className="size-3.5 fill-current stroke-none" />
							)}
						</button>
					)}
					{posts.map((p, k) => (
						<button
							key={p.href}
							type="button"
							aria-label={`Show ${p.title}`}
							aria-current={k === index ? 'true' : undefined}
							onClick={() => setIndex(k)}
							className={`relative grid h-8 w-9 cursor-pointer place-items-center rounded-full after:absolute after:-inset-x-1 after:-inset-y-1.5 ${FOCUS}`}
						>
							<span
								// The pill is always 16px wide; a clip trims it to a dot, so nothing reflows.
								className={cn(
									'block h-1.5 w-4 rounded-full transition-[clip-path,background-color] duration-300 ease-out motion-reduce:transition-none',
									k === index
										? 'bg-foreground [clip-path:inset(0_round_9999px)]'
										: 'bg-border-strong [clip-path:inset(0_5px_round_9999px)]',
								)}
							/>
						</button>
					))}
				</div>
			)}
		</section>
	);
}
