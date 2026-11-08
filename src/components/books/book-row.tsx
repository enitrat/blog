import Image, { type StaticImageData } from 'next/image';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/** A book's real cover cropped to a 2:3 jacket, or its initial on a blank board when none is checked in. */
export function BookThumb({
	cover,
	title,
	className,
}: {
	cover: StaticImageData | undefined;
	title: string;
	/** Sets the width, such as `w-10`. */
	className?: string;
}) {
	const frame = cn('aspect-[2/3] flex-none rounded-[3px] shadow-card', className);
	if (!cover) {
		return (
			<span
				aria-hidden="true"
				className={cn(frame, 'grid place-items-center bg-muted font-display text-lg text-faint')}
			>
				{title[0]}
			</span>
		);
	}
	// A landscape image is a wraparound jacket: its front cover is the right half.
	return (
		<Image
			src={cover}
			alt=""
			sizes="8rem"
			className={cn(
				frame,
				'bg-muted object-cover',
				cover.width > cover.height ? 'object-right' : 'object-center',
			)}
		/>
	);
}

/**
 * One finished book in the archive, in the writing-row grammar: cover, the title in the serif, the
 * author beneath, and the month and rating in mono on the right.
 */
export function BookRow({
	title,
	author,
	cover,
	month,
	datetime,
	rating,
	high,
}: {
	title: string;
	author: string;
	cover: StaticImageData | undefined;
	/** Short month, such as "Aug"; omitted for books known only by period. */
	month?: string;
	datetime?: string;
	rating: number | null;
	/** Sets the rating in the accent. */
	high: boolean;
}): ReactNode {
	return (
		<li className="grid grid-cols-[2.5rem_minmax(0,1fr)_auto] items-center gap-4 border-t border-border py-3 first:border-t-0">
			<BookThumb cover={cover} title={title} className="w-10" />
			<div className="min-w-0">
				<h3 className="font-display text-xl leading-tight text-balance text-foreground">{title}</h3>
				<p className="text-sm text-muted-foreground">{author}</p>
			</div>
			<div className="grid justify-items-end gap-0.5 font-mono text-xs tracking-[-0.01em] text-faint tabular-nums">
				{month && <time dateTime={datetime}>{month}</time>}
				{rating !== null && (
					<span className={high ? 'text-primary' : 'text-foreground'}>
						{rating}
						<span className="text-faint">/5</span>
						<span className="sr-only"> rating</span>
					</span>
				)}
			</div>
		</li>
	);
}
