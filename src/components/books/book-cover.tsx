'use client';

import { motion } from 'motion/react';
import Image from 'next/image';
import { formatRating } from '@/lib/format';
import { spring } from '@/lib/motion';
import type { ShelfBook } from '@/lib/piles';
import { cn } from '@/lib/utils';

/** A book's spine edge is squarer than its fore-edge. */
const JACKET = 'rounded-[2px_4px_4px_2px]';
const JACKET_BACK = 'rounded-[4px_2px_2px_4px]';

/** The front cover, or the author's leather with the title on it when no cover is checked in. */
export function Cover({ book, sizes }: { book: ShelfBook; sizes: string }) {
	if (book.cover)
		return (
			<Image
				src={book.cover}
				alt=""
				sizes={sizes}
				draggable={false}
				className={cn('size-full object-cover', JACKET)}
			/>
		);
	return (
		<span
			className={cn('flex size-full flex-col justify-between p-2.5 text-left', JACKET)}
			style={{ background: book.leather }}
		>
			<span className="type-body-strong text-balance text-scrim-foreground">{book.title}</span>
			<span className="type-small text-scrim-foreground">{book.author}</span>
		</span>
	);
}

function Back({ book }: { book: ShelfBook }) {
	return (
		<span
			className={cn(
				'flex size-full flex-col gap-1 overflow-hidden bg-card p-2.5 text-left shadow-card',
				JACKET_BACK,
			)}
		>
			<span className="line-clamp-3 flex-none type-control-sm text-foreground">{book.title}</span>
			<span className="line-clamp-2 flex-none type-small text-muted-foreground">{book.author}</span>
			{book.note && (
				<span className="line-clamp-3 flex-none type-small text-primary">“{book.note}”</span>
			)}
			<span className="mt-auto type-meta text-faint">{book.when ?? book.year}</span>
		</span>
	);
}

/** Both faces of a book, the back turned away until `flipped`. */
export function BookFaces({
	book,
	flipped,
	sizes,
}: {
	book: ShelfBook;
	flipped: boolean;
	sizes: string;
}) {
	return (
		<motion.span
			className="relative block size-full [transform-style:preserve-3d]"
			initial={false}
			animate={{ rotateY: flipped ? 180 : 0 }}
			transition={spring.fan}
		>
			<span className={cn('absolute inset-0 shadow-book [backface-visibility:hidden]', JACKET)}>
				<Cover book={book} sizes={sizes} />
			</span>
			<span
				className={cn(
					'absolute inset-0 shadow-book [backface-visibility:hidden] [transform:rotateY(180deg)]',
					JACKET_BACK,
				)}
			>
				<Back book={book} />
			</span>
		</motion.span>
	);
}

/** A favourite's rating, stuck on the cover's corner in the wordmark's sticker inks. */
export function RatingSticker({ rating, className }: { rating: number; className?: string }) {
	return (
		<span
			aria-hidden
			className={cn(
				'absolute z-10 grid size-9 rotate-[14deg] place-items-center rounded-full border-2 border-sticker-outline bg-sticker-fill type-meta text-sticker-outline shadow-[2px_2px_0_var(--sticker-shadow)]',
				className,
			)}
		>
			{formatRating(rating)}
		</span>
	);
}

/** The bookmark ribbon hanging out of a book being read. */
export function Ribbon({ className }: { className?: string }) {
	return (
		<span
			aria-hidden
			className={cn(
				'absolute h-8 w-2 bg-primary [clip-path:polygon(0_0,100%_0,100%_100%,50%_80%,0_100%)]',
				className,
			)}
		/>
	);
}
