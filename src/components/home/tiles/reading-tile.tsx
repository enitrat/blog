import 'server-only';
import { books } from '@/booksData';
import { Tile } from '@/components/ui/tile';
import { findImage } from '@/lib/media';
import { Deck } from '../deck';
import { Plate } from '../plate';

const stamp = (d: Date | null) => d?.getTime() ?? 0;
const short = (t: string) => t.replace(/\s*\(.*\)\s*$/, '');

/**
 * The books being read now as a pile of covers, each glowing in its own colours. Hover pulls the
 * book under the pointer up and out of the pile; letting go squares the pile. The caption under it
 * follows the held book (title, author). The whole tile links to /reading. With no current book it
 * shows the last one finished.
 */
export async function ReadingTile() {
	const current = books
		.filter((b) => b.status === 'reading')
		.sort((a, b) => stamp(b.dateStarted) - stamp(a.dateStarted));
	const last = books
		.filter((b) => b.status === 'finished')
		.sort((a, b) => stamp(b.dateFinished) - stamp(a.dateFinished))[0];
	const label = current.length ? 'Reading' : 'Last read';
	const shown = (current.length ? current : last ? [last] : []).slice(0, 2);
	const stack = await Promise.all(
		shown.map(async (book) => ({ book, cover: await findImage(`covers/${book.edition.isbn13}`) })),
	);
	return (
		<Tile
			size="compact"
			label={label}
			href="/reading/"
			meta={current.length > 1 ? `+${current.length - 1} more` : undefined}
		>
			{stack.length > 0 && (
				<Deck
					hold="pull"
					lift={9}
					max="var(--deck-max, min(6.5rem, 62%))"
					className="h-full justify-between gap-2 pt-2 text-right max-lg:[--deck-max:min(9rem,62%)]"
					captionClassName="justify-items-end"
					cards={stack.map(({ book, cover }) => (
						<Plate key={book.edition.isbn13} src={cover} alt="" glow sizes="12rem">
							<span className="grid h-full content-end gap-1 p-3 text-left text-xs text-gray-500 dark:text-iron-400">
								<strong className="font-semibold text-gray-800 dark:text-iron-100">
									{short(book.title)}
								</strong>
								<span>{book.author}</span>
							</span>
						</Plate>
					))}
					captions={stack.map(({ book }) => (
						<span key={book.edition.isbn13} className="grid">
							<span className="truncate text-sm font-semibold">{short(book.title)}</span>
							<span className="truncate text-xs text-gray-500 dark:text-iron-400">
								{book.author}
							</span>
						</span>
					))}
				/>
			)}
		</Tile>
	);
}
