'use client';

import { AnimatePresence, LayoutGroup, motion } from 'motion/react';
import { useEffect, useRef, useState } from 'react';
import { Eyebrow } from '@/components/ui/eyebrow';
import { ease, spring } from '@/lib/motion';
import { type Pile, type PileSort, pile, type ShelfBook, SORTS } from '@/lib/piles';
import { FanPile } from './fan-pile';
import { type Peek, PileDock } from './pile-dock';

/** Grace for the trip from a pile down to the dock, and for leaving the dock. */
const LEAVE_PILE_MS = 450;
const LEAVE_DOCK_MS = 250;
/** Another pile takes the dock only once the pointer rests on it, so crossing piles on the way does not. */
const TAKE_OVER_MS = 140;

/**
 * Every shelved book in piles, re-dealt when the sort changes. Holding a pile opens the dock; the
 * pointer can carry it down into the dock, which then drives the pile, and it closes once the
 * pointer has left both. A tap elsewhere or Escape closes it too.
 */
export function PileTable({ books }: { books: ShelfBook[] }) {
	const [sort, setSort] = useState<PileSort>('author');
	const [peek, setPeekState] = useState<Peek | null>(null);
	const current = useRef<Peek | null>(null);
	const close = useRef<ReturnType<typeof setTimeout>>(undefined);
	const takeOver = useRef<ReturnType<typeof setTimeout>>(undefined);
	const [arrived, setArrived] = useState(false);

	const setPeek = (p: Peek | null) => {
		current.current = p;
		setPeekState(p);
	};
	const closeIn = (ms: number) => {
		clearTimeout(close.current);
		close.current = setTimeout(() => setPeek(null), ms);
	};
	const stay = () => {
		clearTimeout(close.current);
		clearTimeout(takeOver.current);
	};
	const onPeek = (p: Pile, index: number | null) => {
		clearTimeout(takeOver.current);
		if (index === null) return closeIn(LEAVE_PILE_MS);
		const open = current.current;
		if (open && open.pile.label !== p.label) {
			takeOver.current = setTimeout(() => {
				clearTimeout(close.current);
				setPeek({ pile: p, index });
			}, TAKE_OVER_MS);
			return;
		}
		clearTimeout(close.current);
		setPeek({ pile: p, index });
	};

	useEffect(() => {
		setArrived(true);
		const shut = () => {
			current.current = null;
			setPeekState(null);
		};
		const onKey = (e: KeyboardEvent) => e.key === 'Escape' && shut();
		const onTap = (e: PointerEvent) => {
			if (!(e.target as Element).closest('[data-pile-table]')) shut();
		};
		addEventListener('keydown', onKey);
		addEventListener('pointerdown', onTap);
		return () => {
			removeEventListener('keydown', onKey);
			removeEventListener('pointerdown', onTap);
			clearTimeout(close.current);
			clearTimeout(takeOver.current);
		};
	}, []);

	const piles = pile(sort, books);
	const starts = piles.map((_, i) => piles.slice(0, i).reduce((s, p) => s + p.books.length, 0));

	return (
		<div data-pile-table className="flex flex-col gap-6">
			<div className="flex flex-wrap items-center gap-x-5 gap-y-1">
				<Eyebrow as="span">Piled by</Eyebrow>
				<fieldset aria-label="Pile the books by" className="flex min-w-0 gap-4">
					{SORTS.map((s) => (
						<button
							key={s.id}
							type="button"
							aria-pressed={sort === s.id}
							onClick={() => {
								setPeek(null);
								setSort(s.id);
							}}
							className="relative inline-flex min-h-11 cursor-pointer items-center rounded-inline type-control text-muted-foreground transition-colors duration-150 hover:text-foreground aria-pressed:text-foreground"
						>
							{s.label}
							{sort === s.id && (
								<motion.span
									layoutId="pile-sort"
									transition={spring.layout}
									className="absolute inset-x-0 top-[calc(50%+0.875rem)] h-[2.5px] rounded-full bg-primary"
								/>
							)}
						</button>
					))}
				</fieldset>
			</div>
			<LayoutGroup>
				<div className="grid grid-cols-2 gap-x-4 gap-y-14 pt-6 sm:grid-cols-4">
					{piles.map((p, i) => (
						<motion.div
							key={`${sort}-${p.label}`}
							initial={arrived ? false : { opacity: 0, y: 8 }}
							animate={{ opacity: 1, y: 0 }}
							transition={{ ...ease, delay: Math.min(i, 8) * 0.025 }}
						>
							<FanPile
								pile={p}
								order={starts[i]}
								active={peek?.pile.label === p.label ? peek.index : null}
								onPeek={(index) => onPeek(p, index)}
							/>
						</motion.div>
					))}
				</div>
			</LayoutGroup>
			<div className="pointer-events-none fixed inset-x-0 bottom-4 z-40 flex justify-center px-4 pb-[env(safe-area-inset-bottom)]">
				<AnimatePresence>
					{peek && (
						<motion.div
							initial={{ opacity: 0, y: 24, scale: 0.98 }}
							animate={{ opacity: 1, y: 0, scale: 1 }}
							exit={{ opacity: 0, y: 16, transition: ease }}
							transition={spring.enter}
							onMouseEnter={stay}
							onMouseLeave={() => closeIn(LEAVE_DOCK_MS)}
							className="pointer-events-auto max-w-full"
						>
							<PileDock peek={peek} choose={(index) => setPeek({ ...peek, index })} />
						</motion.div>
					)}
				</AnimatePresence>
			</div>
		</div>
	);
}
