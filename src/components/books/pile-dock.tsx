'use client';

import { AnimatePresence, motion } from 'motion/react';
import {
	type KeyboardEvent,
	type MouseEvent,
	type PointerEvent,
	type RefObject,
	useEffect,
	useLayoutEffect,
	useRef,
	useState,
} from 'react';
import { formatCount, formatRating } from '@/lib/format';
import { spring } from '@/lib/motion';
import type { Pile, ShelfBook } from '@/lib/piles';
import { BookFaces, Cover } from './book-cover';

export type Peek = { pile: Pile; index: number };

/** The hand never holds more than this; the rail reaches the rest. */
const HAND = 9;
const CARD = 96;
/** The dock width the arc and cards are drawn for; a narrower dock draws both smaller. */
const FULL = 576;
/** Every card in the hand pivots well below it, like a wrist. */
const PIVOT = '50% 320%';

const meta = (b: ShelfBook) =>
	[
		b.status === 'reading' ? 'Reading now' : (b.when ?? b.year),
		`${formatCount(b.pages)} pages`,
		b.rating !== null ? `${formatRating(b.rating)}/5` : null,
	]
		.filter(Boolean)
		.join(' · ');

const centred = (n: number, size: number, i: number) =>
	Math.max(0, Math.min(n - size, i - Math.floor(size / 2)));

/**
 * Which stretch of the pile is in hand. It follows the chosen book only when that book leaves it,
 * so pointing at a card never slides the hand under the pointer. A jump from the rail re-centres it.
 */
function useHand(n: number, index: number, recentre: { current: boolean }) {
	const size = Math.min(n, HAND);
	const [start, setStart] = useState(() => centred(n, size, index));
	useEffect(() => {
		if (recentre.current) {
			recentre.current = false;
			setStart(centred(n, size, index));
		} else setStart((s) => (index < s ? index : index >= s + size ? index - size + 1 : s));
	}, [n, size, index, recentre]);
	return { start, end: start + size };
}

function Hand({
	peek,
	choose,
	start,
	end,
	width,
}: {
	peek: Peek;
	choose: (i: number) => void;
	start: number;
	end: number;
	width: number;
}) {
	const [flipped, setFlipped] = useState<string | null>(null);
	const shown = peek.pile.books.slice(start, end);
	const mid = (shown.length - 1) / 2;
	const fit = Math.min(1, width / FULL);
	const step = Math.min(10, 64 / Math.max(1, shown.length - 1)) * fit;
	const card = Math.round(CARD * Math.max(0.8, fit));
	return (
		<AnimatePresence initial={false}>
			{shown.map((b, j) => {
				const i = start + j;
				const chosen = i === peek.index;
				return (
					<motion.button
						key={b.isbn}
						type="button"
						aria-label={chosen ? `${b.title}. Turn over` : b.title}
						aria-current={chosen || undefined}
						aria-pressed={chosen ? flipped === b.isbn : undefined}
						onMouseEnter={() => choose(i)}
						onFocus={() => choose(i)}
						onClick={() => (chosen ? setFlipped(flipped === b.isbn ? null : b.isbn) : choose(i))}
						className="absolute top-8 aspect-2/3 cursor-pointer"
						style={{
							width: card,
							left: `calc(50% - ${card / 2}px)`,
							transformOrigin: PIVOT,
							zIndex: chosen ? 100 : j,
						}}
						initial={{ rotate: (j - mid) * step, y: 40, opacity: 0 }}
						animate={{ rotate: (j - mid) * step, y: 0, opacity: 1 }}
						exit={{ opacity: 0, y: 30 }}
						transition={spring.fan}
					>
						<motion.span
							className="block size-full"
							animate={{ y: chosen ? -22 : 0, scale: chosen ? 1.06 : 1 }}
							transition={spring.fan}
						>
							<BookFaces book={b} flipped={flipped === b.isbn} sizes="7rem" />
						</motion.span>
					</motion.button>
				);
			})}
		</AnimatePresence>
	);
}

/** Travel, in pixels, that turns the hand by one book. */
const STEP = 40;

/**
 * A sideways trackpad scroll anywhere on the dock, or a sideways swipe across the hand, turns the
 * hand one book per `STEP`. Vertical scrolling is left to the page, and a swipe never also taps
 * the card it ends on.
 */
function useSwipe(
	dock: RefObject<HTMLElement | null>,
	index: number,
	n: number,
	jump: (i: number) => void,
) {
	const at = useRef(index);
	at.current = index;
	const go = useRef(jump);
	go.current = jump;
	const turn = (d: number) => {
		const to = Math.max(0, Math.min(n - 1, at.current + d));
		if (to === at.current) return;
		at.current = to;
		go.current(to);
	};
	const turnRef = useRef(turn);
	turnRef.current = turn;

	useEffect(() => {
		const el = dock.current;
		if (!el) return;
		let travel = 0;
		let idle: ReturnType<typeof setTimeout> | undefined;
		const onWheel = (e: WheelEvent) => {
			if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
			e.preventDefault();
			travel += e.deltaX;
			clearTimeout(idle);
			idle = setTimeout(() => {
				travel = 0;
			}, 160);
			while (Math.abs(travel) >= STEP) {
				turnRef.current(Math.sign(travel));
				travel -= Math.sign(travel) * STEP;
			}
		};
		el.addEventListener('wheel', onWheel, { passive: false });
		return () => {
			el.removeEventListener('wheel', onWheel);
			clearTimeout(idle);
		};
	}, [dock]);

	const touch = useRef<{ from: number; last: number; swiped: boolean } | null>(null);
	return {
		onPointerDown: (e: PointerEvent<HTMLDivElement>) => {
			if (e.pointerType === 'touch')
				touch.current = { from: e.clientX, last: e.clientX, swiped: false };
		},
		onPointerMove: (e: PointerEvent<HTMLDivElement>) => {
			const t = touch.current;
			if (!t) return;
			if (Math.abs(e.clientX - t.from) > 8) t.swiped = true;
			while (Math.abs(e.clientX - t.last) >= STEP) {
				const dir = Math.sign(e.clientX - t.last);
				turn(-dir);
				t.last += dir * STEP;
			}
		},
		onPointerUp: () => {
			setTimeout(() => {
				touch.current = null;
			}, 0);
		},
		onPointerCancel: () => {
			touch.current = null;
		},
		onClickCapture: (e: MouseEvent) => {
			if (touch.current?.swiped) e.stopPropagation();
		},
	};
}

function useWidth(ref: RefObject<HTMLElement | null>) {
	const [width, setWidth] = useState(FULL);
	useLayoutEffect(() => {
		const el = ref.current;
		if (!el) return;
		const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
		observer.observe(el);
		return () => observer.disconnect();
	}, [ref]);
	return width;
}

/** A sliver of every cover in the pile, end to end: sweeping across it reaches any book at once. */
function Rail({ peek, jump }: { peek: Peek; jump: (i: number) => void }) {
	const n = peek.pile.books.length;
	const at = (e: PointerEvent<HTMLDivElement>) => {
		const r = e.currentTarget.getBoundingClientRect();
		jump(Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n))));
	};
	const key = (e: KeyboardEvent<HTMLDivElement>) => {
		const to = { ArrowLeft: peek.index - 1, ArrowRight: peek.index + 1, Home: 0, End: n - 1 }[
			e.key
		];
		if (to === undefined) return;
		e.preventDefault();
		jump(Math.max(0, Math.min(n - 1, to)));
	};
	return (
		<div
			role="slider"
			tabIndex={0}
			aria-label={`Books in ${peek.pile.label}`}
			aria-valuemin={1}
			aria-valuemax={n}
			aria-valuenow={peek.index + 1}
			aria-valuetext={peek.pile.books[peek.index].title}
			onPointerMove={at}
			onPointerDown={at}
			onKeyDown={key}
			className="relative flex h-7 cursor-ew-resize touch-none gap-px overflow-hidden rounded-chip"
		>
			{peek.pile.books.map((b, i) => (
				<span
					key={b.isbn}
					className="relative min-w-0 flex-1 overflow-hidden transition-[opacity,transform] duration-150 ease-out"
					style={{
						opacity: i === peek.index ? 1 : 0.45,
						transform: `scaleY(${i === peek.index ? 1 : 0.72})`,
					}}
				>
					<span className="absolute inset-y-0 left-1/2 w-24 -translate-x-1/2">
						<Cover book={b} sizes="4rem" />
					</span>
				</span>
			))}
		</div>
	);
}

/**
 * The dock: the chosen pile held as a hand of cards, its book named above it, and on big piles a
 * rail to reach the rest. Choosing a book here raises it in the pile too.
 */
export function PileDock({ peek, choose }: { peek: Peek; choose: (i: number) => void }) {
	const n = peek.pile.books.length;
	const book = peek.pile.books[peek.index];
	const recentre = useRef(false);
	const { start, end } = useHand(n, peek.index, recentre);
	const hand = useRef<HTMLDivElement>(null);
	const dock = useRef<HTMLDivElement>(null);
	const width = useWidth(hand);
	const jump = (i: number) => {
		if (i !== peek.index) recentre.current = true;
		choose(i);
	};
	const swipe = useSwipe(dock, peek.index, n, jump);
	return (
		<div
			ref={dock}
			className="w-[min(36rem,calc(100vw-2rem))] overflow-hidden rounded-card bg-card/95 pt-4 shadow-lifted backdrop-blur-sm"
		>
			<div aria-live="polite" className="px-6 text-center">
				<p className="truncate type-body-strong text-foreground">
					{book.title}
					<span className="type-body text-muted-foreground"> · {book.author}</span>
				</p>
				<p className="mt-0.5 type-meta text-faint">{meta(book)}</p>
			</div>
			<div ref={hand} {...swipe} className="relative mt-1 h-42 touch-pan-y overflow-hidden">
				<Hand peek={peek} choose={choose} start={start} end={end} width={width} />
			</div>
			{n > HAND && (
				<div className="border-t border-border px-4 pt-3 pb-4">
					<Rail peek={peek} jump={jump} />
				</div>
			)}
		</div>
	);
}
