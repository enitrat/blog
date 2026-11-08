import { cn } from '@/lib/utils';

/** Pages across the full column, the same for every year, so years compare by weight. */
const FULL = 2500;

/** Cloth tones for the strip; a title hash picks one. */
const TONES = ['bg-foreground', 'bg-faint', 'bg-muted-foreground/35'] as const;

const hash = (s: string) => {
	let h = 0;
	for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
	return h;
};

/**
 * A year's books laid end to end as one rule, oldest first: each segment as wide as the book is
 * long. Highly rated books take the accent. Decorative: the rows below name every book.
 */
export function YearStrip({
	books,
}: {
	/** Oldest first. */
	books: { title: string; pages: number; high: boolean }[];
}) {
	const total = books.reduce((s, b) => s + b.pages, 0);
	return (
		<div
			aria-hidden="true"
			className="flex h-2 gap-0.5"
			style={{ width: `${Math.min(100, (total / FULL) * 100)}%` }}
		>
			{books.map((b, i) => (
				<span
					// biome-ignore lint/suspicious/noArrayIndexKey: a strip is positional; a title may sit twice
					key={i}
					title={`${b.title}, ${b.pages} pages`}
					className={cn(
						'min-w-0.5 rounded-[1px]',
						b.high ? 'bg-primary' : TONES[hash(b.title) % 3],
					)}
					style={{ flex: `${b.pages} 0 0` }}
				/>
			))}
		</div>
	);
}
