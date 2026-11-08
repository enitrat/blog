import { cn } from '@/lib/utils';

/** Cloth tones, readable in both themes; a title hash picks one. */
const TONES = [
	'bg-gray-800 text-white dark:bg-iron-100 dark:text-iron-950',
	'bg-gray-500 text-white dark:bg-iron-500 dark:text-iron-950',
	'bg-gray-100 text-gray-800 dark:bg-iron-800 dark:text-iron-100',
	'bg-gray-600 text-white dark:bg-iron-300 dark:text-iron-950',
	'bg-white text-gray-800 dark:bg-iron-900 dark:text-iron-100',
	'bg-gray-300 text-gray-800 dark:bg-iron-700 dark:text-iron-100',
] as const;

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
			href={href}
			title={`${short}, ${author}`}
			className={cn(
				'relative flex-none overflow-hidden rounded-t-xs border border-b-0 border-gray-200 py-3 no-underline shadow-sm shadow-zinc-800/15 dark:border-white/10',
				TONES[hash % TONES.length],
				href &&
					'transition-transform duration-150 ease-out outline-none hover:-translate-y-2 focus-visible:-translate-y-2 focus-visible:ring-2 focus-visible:ring-brand-500/60 motion-reduce:transition-none motion-reduce:hover:translate-y-0 motion-reduce:focus-visible:translate-y-0',
			)}
			style={{ width: `${width.toFixed(2)}rem`, height: `${height.toFixed(2)}rem` }}
		>
			<span
				className="block max-h-full truncate text-xs font-semibold [writing-mode:vertical-rl]"
				style={{ lineHeight: `${width.toFixed(2)}rem` }}
			>
				{short}
				<span className="sr-only">
					{' '}
					by {author}, {pages} pages{status && `, ${STATUS[status]}`}
				</span>
			</span>
		</Root>
	);
}
