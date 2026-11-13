'use client';

import { ChevronRightIcon, PauseIcon, PlayIcon } from 'lucide-react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import Link from 'next/link';
import { type KeyboardEvent, useEffect, useRef, useState } from 'react';
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
/** Characters past which a title keeps the title role instead of the headline: a long headline
 *  outgrows a one-screen tile, as the share card sizes its title by length too. */
const HEADLINE_MAX = 48;

/**
 * The latest posts, one at a time. It lays out by its own width (`@container/slide`), since the tile
 * it sits in is sized by the bento, not the window: from 32rem the banner fills the right column
 * and the controls take a row under the slide; narrower, the banner goes and the controls sit
 * beside Read more.
 *
 * A slide reads like a magazine card: date and title at the top, summary and Read more at the
 * bottom, a size up beside the banner, so a tall tile reads as a cover instead of a blank top half. Read more
 * stretches over the slide. It advances on its own and holds while the pointer or focus is inside;
 * under reduced motion it never advances. Pause stops it for good, which is the only way to on
 * touch. The dots are one tab stop: arrow keys move between posts.
 */
export function WritingCarousel({ posts }: { posts: CarouselPost[] }) {
	const [index, setIndex] = useState(0);
	const [held, setHeld] = useState(false);
	const [paused, setPaused] = useState(false);
	const dots = useRef<(HTMLButtonElement | null)[]>([]);
	const rotates = !useReducedMotion() && posts.length > 1;
	// biome-ignore lint/correctness/useExhaustiveDependencies: `index` restarts the dwell after a pick.
	useEffect(() => {
		if (held || paused || !rotates) return;
		const timer = setTimeout(() => setIndex((n) => (n + 1) % posts.length), DWELL * 1000);
		return () => clearTimeout(timer);
	}, [index, held, paused, rotates, posts.length]);

	const step = (e: KeyboardEvent<HTMLButtonElement>) => {
		const by = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
		if (!by) return;
		e.preventDefault();
		const next = (index + by + posts.length) % posts.length;
		setIndex(next);
		dots.current[next]?.focus();
	};

	const post = posts[index];
	return (
		<section
			aria-roledescription="carousel"
			aria-label="Latest writing"
			className="@container/slide relative mt-3 flex min-h-0 flex-1 flex-col"
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
					className={cn(
						'grid min-h-0 flex-1 grid-cols-1 gap-6',
						post.banner && '@[32rem]/slide:grid-cols-[1.25fr_1fr]',
					)}
					initial={{ opacity: 0, x: 16 }}
					animate={{ opacity: 1, x: 0 }}
					exit={{ opacity: 0, x: -16, transition: ease }}
					transition={spring.enter}
				>
					<div className="flex min-h-0 flex-col">
						<time dateTime={post.iso} className="type-meta text-faint">
							{post.date}
						</time>
						<h3
							className={cn(
								'mt-1.5 type-title text-balance transition-colors duration-150 ease-out group-data-[held]/tile:text-primary',
								post.title.length <= HEADLINE_MAX && '@[32rem]/slide:type-headline',
							)}
						>
							{post.title}
						</h3>
						<div className="mt-auto pt-3">
							<p className="line-clamp-3 type-body text-pretty text-muted-foreground @[32rem]/slide:type-lead">
								{post.description}
							</p>
							<Link
								href={post.href}
								className="mt-4 inline-flex min-h-10 items-center gap-0.5 rounded-control bg-card pr-2.5 pl-3.5 type-control shadow-card transition-colors duration-150 ease-out link-cover group-data-[held]/tile:text-primary"
							>
								Read more
								<ChevronRightIcon
									aria-hidden
									className="size-4 transition-transform duration-150 ease-out group-data-[held]/tile:translate-x-0.5"
								/>
							</Link>
						</div>
					</div>
					{post.banner && (
						<div className="relative hidden min-h-40 overflow-hidden rounded-media fade-inline-start @[32rem]/slide:block">
							<Image
								src={post.banner}
								alt=""
								fill
								sizes="(min-width: 64rem) 25vw, 50vw"
								className="object-cover transition-transform duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] group-data-[held]/tile:scale-[1.04]"
							/>
						</div>
					)}
				</motion.article>
			</AnimatePresence>
			{posts.length > 1 && (
				<div className="absolute right-0 bottom-1 flex @[32rem]/slide:static @[32rem]/slide:-mb-1.5 @[32rem]/slide:-ml-2.5 @[32rem]/slide:pt-2">
					{rotates && (
						<button
							type="button"
							aria-label={paused ? 'Play slideshow' : 'Pause slideshow'}
							onClick={() => setPaused((p) => !p)}
							className="relative grid size-8 cursor-pointer place-items-center rounded-full text-faint transition-colors hit-area hover:text-foreground"
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
							ref={(el) => {
								dots.current[k] = el;
							}}
							type="button"
							aria-label={`Show ${p.title}`}
							aria-current={k === index ? 'true' : undefined}
							tabIndex={k === index ? 0 : -1}
							onClick={() => setIndex(k)}
							onKeyDown={step}
							className="relative grid h-8 w-9 cursor-pointer place-items-center rounded-full hit-area"
						>
							<span
								// The pill is always 16px wide; a clip trims it to a dot, so nothing reflows.
								className={cn(
									'block h-1.5 w-4 rounded-full transition-[clip-path,background-color] duration-300 ease-out',
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
