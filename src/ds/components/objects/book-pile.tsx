import { cn } from '@/ds/lib/cn';
import { frameOf, type PileBook, stack } from './book';

/**
 * Books lying flat in one isometric pile, first book at the bottom. Each is a hardback (two
 * boards, a page block, a cloth spine facing front left) as thick as its page count, at one scale
 * for every pile on the site, so piles compare by weight. Geometry in `book.ts`; cut long runs
 * with `piles()`. Decorative: the page lists the same books as text. A spine thick enough
 * carries its title; every book also shows it as a hover tooltip.
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
	const [x, y, w, h] = frameOf(books);
	return (
		<svg
			data-slot="book-pile"
			className={cn('pile block h-auto max-w-full overflow-visible', className)}
			width={+(w * scale).toFixed(1)}
			height={+(h * scale).toFixed(1)}
			viewBox={`${x} ${y} ${w} ${h}`}
			aria-hidden="true"
			focusable="false"
			shapeRendering="geometricPrecision"
		>
			{stack(books).map((b, i) => (
				// biome-ignore lint/suspicious/noArrayIndexKey: a pile is positional; a title may sit twice
				<g key={i} data-tone={b.tone}>
					<title>{b.title}</title>
					<polygon className="r" points={b.lower.right} />
					<polygon className="t" points={b.lower.top} />
					<polygon className="pg" points={b.block.right} />
					<polygon className="pg" points={b.block.top} />
					{b.leaves.map((p) => (
						<polyline key={p} className="hair" points={p} />
					))}
					<polygon className="r" points={b.upper.right} />
					<polygon className="t" points={b.upper.top} />
					<polygon className="spine" points={b.spine} />
					{b.bands.map((p) => (
						<polyline key={p} className="hair" points={p} />
					))}
					{b.text && (
						<text transform={b.textMatrix} fontSize={b.font} dominantBaseline="central">
							{b.text}
						</text>
					)}
				</g>
			))}
		</svg>
	);
}
