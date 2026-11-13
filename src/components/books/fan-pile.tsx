'use client';

import { motion } from 'motion/react';
import { useState } from 'react';
import { spring } from '@/lib/motion';
import type { Pile } from '@/lib/piles';
import { BookFaces, RatingSticker, Ribbon } from './book-cover';

const hash = (s: string) => {
	let h = 2166136261;
	for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
	return h;
};
/** A stable pseudo-random number in [-1, 1], so a pile looks the same on every visit. */
const jitter = (key: string, salt: number) => ((hash(key + salt) % 2001) - 1000) / 1000;

/** Past this many books a pile stops growing taller or wider; the dock shows the rest. */
const DEPTH = 8;

const plural = (n: number) => `${n} ${n === 1 ? 'book' : 'books'}`;

/**
 * A pile of covers that fans out like a hand of cards while it is held. Pointing at a cover, or
 * choosing one in the dock (`active`), raises it. `order` is the pile's place in the whole deal,
 * so a re-sort sends the covers to their new piles one after another.
 */
export function FanPile({
	pile,
	order,
	active,
	onPeek,
}: {
	pile: Pile;
	order: number;
	active: number | null;
	onPeek: (index: number | null) => void;
}) {
	const [held, setHeld] = useState(false);
	const n = pile.books.length;
	const open = held || active !== null;
	const mid = (n - 1) / 2;
	const spread = Math.min(10, 36 / Math.max(1, n - 1));
	return (
		<section
			aria-label={`${pile.label}, ${plural(n)}`}
			className="flex flex-col items-center"
			onMouseEnter={() => setHeld(true)}
			onMouseLeave={() => {
				setHeld(false);
				onPeek(null);
			}}
			onBlur={(e) => {
				if (e.currentTarget.contains(e.relatedTarget)) return;
				setHeld(false);
				onPeek(null);
			}}
		>
			<div className="relative aspect-2/3 w-29" style={{ zIndex: open ? 20 : 1 }}>
				{pile.books.map((b, i) => {
					const raised = active === i;
					const fan = {
						x: (i - mid) * spread * 2,
						y: Math.min(Math.abs(i - mid), DEPTH) * 2.5 - 6,
						rotate: (i - mid) * spread * 0.5,
					};
					const rest = {
						x: Math.min(i, DEPTH) * 3 + jitter(b.isbn, 1) * 2,
						y: -Math.min(i, DEPTH) * 3,
						rotate: jitter(b.isbn, 2) * (n > 1 ? 5 : 3),
					};
					return (
						<motion.button
							key={b.isbn}
							layoutId={b.isbn}
							type="button"
							tabIndex={open || i === 0 ? 0 : -1}
							aria-label={`${b.title}, ${b.author}`}
							aria-current={raised || undefined}
							onFocus={() => {
								setHeld(true);
								onPeek(i);
							}}
							onHoverStart={() => held && onPeek(i)}
							onClick={() => {
								setHeld(true);
								onPeek(i);
							}}
							initial={false}
							animate={
								raised
									? { x: fan.x, y: fan.y - 18, rotate: fan.rotate * 0.4, scale: 1.04 }
									: { ...(open ? fan : rest), scale: 1 }
							}
							transition={{
								...spring.fan,
								layout: { ...spring.layout, delay: Math.min(order + i, 20) * 0.01 },
							}}
							className="absolute inset-0 origin-bottom cursor-pointer"
							style={{ zIndex: raised ? 100 : n - i }}
						>
							<BookFaces book={b} flipped={false} sizes="8rem" />
							{b.favourite && b.rating !== null && (open || i === 0) && (
								<RatingSticker rating={b.rating} className="-top-3 -right-3 scale-75" />
							)}
							{pile.reading && i === 0 && <Ribbon className="-bottom-4 left-4" />}
						</motion.button>
					);
				})}
			</div>
			<div className="mt-5 max-w-44 text-center">
				<h2 className="type-body-strong text-balance text-foreground">{pile.label}</h2>
				<p className="type-meta text-faint">{n === 1 ? pile.books[0].title : plural(n)}</p>
			</div>
		</section>
	);
}
