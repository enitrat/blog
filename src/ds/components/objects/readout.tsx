'use client';

import { useEffect, useRef } from 'react';
import { cn } from '@/ds/lib/cn';
import { followDeck } from './readout-controller';

const MODES = {
	/** Rolls the whole string and truncates to one line. */
	text: 'block',
	/** Rolls only the characters that change, with tabular numerals (a clock, a count). */
	char: 'inline-block tabular-nums',
} as const;

export type ReadoutMode = keyof typeof MODES;

type Base = {
	/** The text shown first, and what a screen reader gets. */
	value: string;
	as?: 'span' | 'p' | 'div' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' | 'time' | 'strong';
	mode?: ReadoutMode;
	/** Announce changes politely. Only for a value that changes on its own and matters. */
	live?: boolean;
	id?: string;
	className?: string;
};

/**
 * Text that reports a held object's state (the active card's title, the elapsed time) and changes
 * by rolling, never by cutting. Server-rendered as plain text, so it reads fine without JS. Update
 * it from a script with `setReadout(el, text)`, or let a Deck drive it with `follow`.
 * `--readout-justify` (start or end, inherited) places the text in a wider box.
 */
export function Readout({
	value,
	as: Tag = 'span',
	mode = 'text',
	live = false,
	follow,
	field,
	id,
	className,
}: Base & ({ follow: string; field: string } | { follow?: undefined; field?: undefined })) {
	const ref = useRef<HTMLElement>(null);
	useEffect(() => (ref.current && follow ? followDeck(ref.current) : undefined), [follow]);
	return (
		<Tag
			ref={ref as never}
			id={id}
			className={cn('readout relative min-w-0', MODES[mode], className)}
			data-readout=""
			data-mode={mode}
			data-readout-follow={follow}
			data-readout-field={follow ? field : undefined}
			aria-live={live ? 'polite' : undefined}
			aria-atomic={live ? 'true' : undefined}
		>
			<span className="readout-sr sr-only">{value}</span>
			<span className="readout-stage" aria-hidden="true">
				{value}
			</span>
		</Tag>
	);
}
