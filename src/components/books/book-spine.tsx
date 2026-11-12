import { cn } from '@/lib/utils';

/** Cloth tones, readable in both themes; a title hash picks one. */
const TONES = [
	'bg-foreground text-background',
	'bg-faint text-background',
	'bg-muted text-foreground',
	'bg-muted-foreground text-background',
	'bg-card text-foreground',
	'bg-muted-foreground/30 text-foreground',
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
				'relative flex flex-none justify-center overflow-hidden rounded-t-[2px] py-3 no-underline shadow-card',
				TONES[hash % TONES.length],
				href &&
					'transition-transform duration-150 ease-out outline-none hover:-translate-y-2 focus-visible:-translate-y-2 focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none motion-reduce:hover:translate-y-0 motion-reduce:focus-visible:translate-y-0',
			)}
			style={{ width: `${width.toFixed(2)}rem`, height: `${height.toFixed(2)}rem` }}
		>
			<span className="block max-h-full truncate type-control-sm [writing-mode:vertical-rl]">
				{short}
				<span className="sr-only">
					{' '}
					by {author}, {pages} pages{status && `, ${STATUS[status]}`}
				</span>
			</span>
		</Root>
	);
}
