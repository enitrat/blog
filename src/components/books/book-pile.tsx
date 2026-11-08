import { cn } from '@/lib/utils';
import { PILE_WIDTH, type PileBook, stack } from './book';

const TONES = ['bg-card', 'bg-muted', 'bg-muted-foreground/15'] as const;

/** Spines thinner than this many pixels carry no title. */
const TITLE_MIN = 9;

/**
 * Books lying flat in one pile, first book at the bottom, spines out. Each is as thick as its
 * page count, at one scale for every pile on the site, so piles compare by weight; cut long runs
 * with `piles()`. Decorative: the page lists the same books as text. A spine thick enough carries
 * its title; every book also shows it as a hover tooltip.
 */
export function BookPile({
	books,
	scale = 5,
	className,
}: {
	/** Bottom first. */
	books: PileBook[];
	/** Pixels per world unit; piles that are compared share one value. */
	scale?: number;
	className?: string;
}) {
	return (
		<div
			aria-hidden="true"
			className={cn('flex max-w-full flex-col-reverse items-start', className)}
			style={{ width: PILE_WIDTH * scale }}
		>
			{stack(books).map((b, i) => {
				const height = b.thickness * scale;
				return (
					<div
						// biome-ignore lint/suspicious/noArrayIndexKey: a pile is positional; a title may sit twice
						key={i}
						title={b.title}
						className={cn(
							'flex max-w-full items-center overflow-hidden rounded-b-xs px-1.5 shadow-card',
							TONES[b.tone],
						)}
						style={{ width: b.length * scale, height, marginLeft: b.offset * scale }}
					>
						{height >= TITLE_MIN && (
							<span
								className="truncate leading-none font-semibold text-muted-foreground"
								style={{ fontSize: Math.min(12, height * 0.6) }}
							>
								{b.title}
							</span>
						)}
					</div>
				);
			})}
		</div>
	);
}
