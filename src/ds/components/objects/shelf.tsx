'use client';

import { type ReactNode, useEffect, useRef } from 'react';
import { Button } from '@/ds/components/ui/button';
import { Icon } from '@/ds/components/ui/icon';
import { cn } from '@/ds/lib/cn';
import { initShelf } from './shelf-controller';

const ALIGN = {
	/** Items sit on one baseline, as books do. */
	end: 'items-end',
	start: 'items-start',
} as const;

/**
 * A horizontal scroller for books or photos. Without JS a plain overflow scroller; with JS, mouse
 * drag with momentum, prev and next buttons, and the arrow keys while focus is inside. Each item
 * should be `flex-none`.
 */
export function Shelf({
	label = 'Scrollable shelf',
	snap = false,
	align = 'end',
	children,
}: {
	/** Accessible name of the scroll region. */
	label?: string;
	/** Snap items to their start edge. */
	snap?: boolean;
	align?: keyof typeof ALIGN;
	children: ReactNode;
}) {
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => (ref.current ? initShelf(ref.current) : undefined), []);
	return (
		<div ref={ref} data-slot="shelf" data-shelf="" className="shelf grid min-w-0 gap-3">
			{/* biome-ignore lint/a11y/useSemanticElements: a scroll track, not a form group */}
			<div
				data-shelf-track=""
				data-snap={snap ? '' : undefined}
				data-align={align}
				role="group"
				aria-label={label}
				// biome-ignore lint/a11y/noNoninteractiveTabindex: a scroll region must be reachable by keyboard
				tabIndex={0}
				className={cn(
					'track focus-ring flex gap-(--shelf-gap) overflow-x-auto overscroll-x-contain border-b border-border pt-4',
					ALIGN[align],
					snap && 'snap-x snap-proximity [&>*]:snap-start',
				)}
			>
				{children}
			</div>
			<div data-shelf-nav="" hidden className="nav flex justify-end gap-2 [&[hidden]]:hidden">
				<Button variant="ghost" size="sm" aria-label="Scroll back">
					<Icon name="chevron-left" />
				</Button>
				<Button variant="ghost" size="sm" aria-label="Scroll forward">
					<Icon name="chevron-right" />
				</Button>
			</div>
		</div>
	);
}
