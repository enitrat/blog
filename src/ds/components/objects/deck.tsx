'use client';

import { type CSSProperties, type ReactNode, useEffect, useRef } from 'react';
import { cn } from '@/ds/lib/cn';
import { initDeck } from './deck-controller';
import { DIM, type Hold, restPose, transformOf } from './deck-poses';

/**
 * A hand of cards. At rest it is stacked (card 0 in front, the next ones peeking, slightly
 * rotated). While open (host hovered, or focus inside) it fans like a hand of cards, or pulls one
 * card up out of the pile, and the active card comes to the front. Without JS, or without a hover
 * pointer, the deck stays stacked and every card is still a link. Positions come from the card
 * count (`deck.ts`), never from callers.
 *
 * A `Readout` with `follow={id}` shows `captions[i][field]` of the held card.
 *
 * Inherited hooks a container may set: `--deck-fit` (width the open hand may fill, default 96%),
 * `--deck-max` (widest card), `--deck-glow` (a resting card's `Plate` glow), `--deck-radius`.
 */
export function Deck({
	cards,
	id,
	captions = [],
	hold = 'fan',
	ratio,
	host = 'self',
	spread: spreadProp = 0.38,
	angle = 6,
	lift = hold === 'pull' ? 12 : 5,
	cycle,
	label,
	className,
}: {
	/** One rendered card per item, front first. */
	cards: readonly ReactNode[];
	id?: string;
	/** One record per card (`{ title, author }`), read by a following `Readout`. */
	captions?: readonly Record<string, string>[];
	/** `fan` spreads the hand sideways (index cards); `pull` draws one card up out of the pile (books) and ignores `spread` and `angle`. */
	hold?: Hold;
	/** Minimum card shape as a CSS ratio ("5/7"). */
	ratio?: string;
	/** `self` opens on the Deck's own hover and focus; `ancestor` on the nearest `[data-deck-host]`. */
	host?: 'self' | 'ancestor';
	/** Distance between neighbouring cards when open, as a fraction of card width. */
	spread?: number;
	/** Rotation per step from the centre when open, deg. */
	angle?: number;
	/** How far the active card rises, % of its height. */
	lift?: number;
	/** Seconds between ambient deals (off by default). */
	cycle?: number;
	/** Accessible name of the group. */
	label?: string;
	className?: string;
}) {
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => (ref.current ? initDeck(ref.current) : undefined), []);
	const n = cards.length;
	const spread = hold === 'pull' ? 0 : spreadProp;
	const vars = {
		'--deck-n': n,
		'--deck-spread': spread,
		...(ratio && { '--deck-ratio': ratio }),
	} as CSSProperties;
	return (
		<div
			ref={ref}
			id={id}
			className={cn('deck', className)}
			data-deck=""
			data-hold={hold}
			data-host={host}
			data-deck-host={host === 'self' ? '' : undefined}
			data-spread={spread}
			data-angle={angle}
			data-lift={lift}
			data-cycle={cycle}
			{...(label && { role: 'group', 'aria-label': label })}
			style={vars}
		>
			{cards.map((card, i) => {
				const p = restPose(i, n, hold);
				return (
					<div
						// biome-ignore lint/suspicious/noArrayIndexKey: the cards never reorder in React; the controller deals them in the DOM
						key={i}
						className="deck-card"
						data-deck-card=""
						data-i={i}
						data-caption={JSON.stringify(captions[i] ?? {})}
						style={{ transform: transformOf(p), zIndex: n - i }}
					>
						{card}
						<span className="deck-dim" aria-hidden="true" style={{ opacity: p.d * DIM }} />
					</div>
				);
			})}
		</div>
	);
}
