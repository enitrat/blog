import type { CSSProperties } from 'react';
import { cn } from '@/ds/lib/cn';

/** Six cloth tones, each a ground and ink from `book-spine.tokens.css`; a title hash picks one. */
const TONES = 6;

const STATUS = {
	reading: 'currently reading',
	finished: 'finished',
	'want-to-read': 'want to read',
} as const;

/**
 * One book standing on a shelf. Tone comes from a hash of the title, width and height from the
 * page count. A linked spine lifts on hover and focus; nothing else moves.
 */
export function BookSpine({
	title,
	author,
	pages,
	status,
	href,
}: {
	/** A trailing "(series, tome)" is dropped from the spine. */
	title: string;
	/** Shown as the tooltip and to screen readers. */
	author: string;
	/** Taller and wider for longer books. */
	pages: number;
	/** Reaches screen readers only: the title line carries the current dot. */
	status?: keyof typeof STATUS;
	href?: string;
}) {
	let hash = 0;
	for (const ch of title) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
	const width = Math.min(2.9, Math.max(1.5, pages / 260));
	const height = 8 + (Math.min(pages, 700) / 700) * 4 + (hash % 5) * 0.25;
	// A straight apostrophe turns into a dash in vertical text, so use the typographic one.
	const short = title.replace(/\s*\(.*\)\s*$/, '').replaceAll("'", '’');
	const Root = href ? 'a' : 'div';
	return (
		<Root
			data-slot="book-spine"
			data-tone={hash % TONES}
			data-status={status}
			href={href}
			title={`${short}, ${author}`}
			className={cn(
				'spine relative h-(--book-spine-h) w-(--book-spine-w) flex-none overflow-hidden border border-b-0 border-border bg-(--book-spine-bg) py-3 text-(--book-spine-fg) no-underline',
				href &&
					'transition-transform duration-base ease-out focus-visible:outline-(length:--focus-ring-width) focus-visible:outline-ring focus-visible:-outline-offset-3 hovered:-translate-y-2 focus-visible:-translate-y-2 motion-reduce:transition-none motion-reduce:hovered:translate-y-0 motion-reduce:focus-visible:translate-y-0',
			)}
			style={
				{
					'--book-spine-w': `${width.toFixed(2)}rem`,
					'--book-spine-h': `${height.toFixed(2)}rem`,
				} as CSSProperties
			}
		>
			<span className="block max-h-full overflow-hidden text-ellipsis whitespace-nowrap font-sans text-label font-semibold leading-(--book-spine-w) [writing-mode:vertical-rl]">
				{short}
				<span className="sr-only">
					{' '}
					by {author}, {pages} pages{status && `, ${STATUS[status]}`}
				</span>
			</span>
		</Root>
	);
}
