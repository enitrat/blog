'use client';

import { AnimatePresence, MotionConfig, motion, useInView } from 'motion/react';
import {
	type CSSProperties,
	type FocusEvent,
	type PointerEvent,
	type ReactNode,
	useEffect,
	useRef,
	useState,
} from 'react';
import { exit, fanStagger, spring } from '@/lib/motion';
import { cn } from '@/lib/utils';
import { DIM, fanZone, type Hold, openPose, pullZone, restPose } from './deck-poses';

/**
 * A hand of cards. At rest it is stacked: card 0 in front, the next ones peeking. While open
 * (hovered, or keyboard focus inside) it fans like a hand of cards, or pulls one card up out of the
 * pile, and the held card comes forward. The held card is picked from the pointer's place over
 * the pile's fixed footprint, never from a moving card. A caption under the pile shows the held
 * card's entry in `captions`. Without hover the deck stays stacked and every card is still a link.
 */
export function Deck({
	cards,
	captions,
	hold = 'fan',
	spread = 0.38,
	angle = 6,
	lift = hold === 'pull' ? 12 : 5,
	cycle,
	label,
	max = '100%',
	fit = '96%',
	className,
	captionClassName,
}: {
	/** One rendered card per item, front first. */
	cards: readonly ReactNode[];
	/** One caption per card, shown for the held card. */
	captions?: readonly ReactNode[];
	/** `fan` spreads the hand sideways (index cards); `pull` draws one card up out of the pile (books) and ignores `spread` and `angle`. */
	hold?: Hold;
	/** Distance between neighbouring cards when open, as a fraction of card width. */
	spread?: number;
	/** Rotation per step from the centre when open, deg. */
	angle?: number;
	/** How far the held card rises, % of its height. */
	lift?: number;
	/** Seconds between ambient deals, front card to the back. Off by default; ends at the first interaction. */
	cycle?: number;
	/** Accessible name of the group. */
	label?: string;
	/** Widest card, a CSS length. */
	max?: string;
	/** Width the open hand may fill, a CSS length. */
	fit?: string;
	className?: string;
	captionClassName?: string;
}) {
	const n = cards.length;
	const options = { spread: hold === 'pull' ? 0 : spread, angle, lift, hold };
	const [open, setOpen] = useState(false);
	/** Stagger the cards only on the move from rest to open. */
	const [opening, setOpening] = useState(false);
	/** Hand position of the held card while open. */
	const [active, setActive] = useState(0);
	/** Item in front at rest; a deal rotates it, so card i sits at hand position (i - top + n) % n. */
	const [top, setTop] = useState(0);
	const [touched, setTouched] = useState(false);
	const root = useRef<HTMLDivElement>(null);
	const pile = useRef<HTMLDivElement>(null);
	const rect = useRef<DOMRect | null>(null);
	const hovering = useRef(false);
	const focused = useRef(false);
	const inView = useInView(root);

	const pos = (i: number) => (i - top + n) % n;
	const held = open ? (active + top) % n : top;

	const show = (at: number) => {
		setTouched(true);
		setOpening(!open);
		setOpen(true);
		setActive(at);
	};
	const hide = () => {
		setOpening(false);
		setOpen(false);
		setActive(0);
	};

	useEffect(() => {
		if (!cycle || n < 2 || touched || open || !inView) return;
		if (
			!matchMedia('(hover: hover)').matches ||
			matchMedia('(prefers-reduced-motion: reduce)').matches
		)
			return;
		const timer = setInterval(() => {
			if (!document.hidden) setTop((t) => (t + 1) % n);
		}, cycle * 1000);
		return () => clearInterval(timer);
	}, [cycle, n, touched, open, inView]);

	const track = (e: PointerEvent) => {
		if (e.pointerType === 'touch' || !pile.current) return;
		rect.current ??= pile.current.getBoundingClientRect();
		const r = rect.current;
		const u = ((e.clientX - r.left) / r.width) * 100;
		const v = ((e.clientY - r.top) / r.height) * 100;
		const at = hold === 'pull' ? pullZone(u, v, n) : fanZone(u, n, options.spread);
		if (!open || at !== active) show(at);
	};
	const onFocus = (e: FocusEvent, i: number) => {
		if (!e.target.matches(':focus-visible')) return;
		focused.current = true;
		show(pos(i));
	};
	const onBlur = (e: FocusEvent) => {
		if (root.current?.contains(e.relatedTarget)) return;
		focused.current = false;
		if (!hovering.current) hide();
	};

	const order = cards.map((_, i) => i).sort((a, b) => pos(a) - pos(b));
	const width = `min(calc(${fit} / (1 + ${options.spread * (n - 1)})), ${max})`;

	return (
		<MotionConfig reducedMotion="user">
			{/* biome-ignore lint/a11y/noStaticElementInteractions: pointer tracking only; keyboard users reach each card's own link */}
			<div
				ref={root}
				className={cn('flex flex-col', className)}
				onPointerEnter={(e) => {
					hovering.current = true;
					rect.current = null;
					track(e);
				}}
				onPointerMove={track}
				onPointerLeave={() => {
					hovering.current = false;
					rect.current = null;
					if (!focused.current) hide();
				}}
				onBlur={onBlur}
			>
				<div
					ref={pile}
					{...(label && { role: 'group', 'aria-label': label })}
					className="relative mx-auto grid self-center"
					style={{ width }}
				>
					{order.map((i) => {
						const p = pos(i);
						const to = open ? openPose(p, n, options, active) : restPose(p, n, hold);
						const z = open ? (p === active ? n + 1 : n - Math.abs(p - active)) : n - p;
						return (
							<motion.div
								key={i}
								className="relative col-start-1 row-start-1"
								style={
									{
										zIndex: z,
										transformOrigin: '50% 90%',
										transformPerspective: 900,
									} as CSSProperties
								}
								initial={false}
								animate={{
									x: `${to.x}%`,
									y: `${to.y}%`,
									rotate: to.r,
									rotateY: to.ry,
									scale: to.s,
								}}
								transition={{ ...spring.fan, delay: opening ? p * fanStagger : 0 }}
								onFocus={(e) => onFocus(e, i)}
							>
								{cards[i]}
								<motion.span
									aria-hidden="true"
									className="pointer-events-none absolute inset-0 rounded-md bg-gray-900 dark:bg-black"
									initial={false}
									animate={{ opacity: to.d * DIM }}
									transition={spring.fan}
								/>
							</motion.div>
						);
					})}
				</div>
				{captions && (
					<div className={cn('grid', captionClassName)}>
						<AnimatePresence initial={false}>
							<motion.div
								key={held}
								className="col-start-1 row-start-1 min-w-0"
								initial={{ opacity: 0, y: 10 }}
								animate={{ opacity: 1, y: 0, scale: 1 }}
								exit={exit}
								transition={spring.enter}
							>
								{captions[held]}
							</motion.div>
						</AnimatePresence>
					</div>
				)}
			</div>
		</MotionConfig>
	);
}
