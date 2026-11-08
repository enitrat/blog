import 'server-only';
import { books } from '@/booksData';
import { Tile } from '@/ds';
import { Deck } from '@/ds/components/objects/deck';
import { Plate } from '@/ds/components/objects/plate';
import { Readout } from '@/ds/components/objects/readout';
import { coverFor } from '@/utils/covers';
import { glowOf } from '@/utils/glow';

const stamp = (d: Date | null) => d?.getTime() ?? 0;
const short = (t: string) => t.replace(/\s*\(.*\)\s*$/, '');

/**
 * The books being read now as a pile of covers, each with a glow in its dominant colour. Hover
 * pulls the book under the pointer up and out of the pile; letting go squares the pile. A Readout
 * under it follows the held book (title, author). The whole tile links to /reading. With no
 * current book it shows the last one finished.
 */
export async function ReadingTile({ order }: { order?: number }) {
	const current = books
		.filter((b) => b.status === 'reading')
		.sort((a, b) => stamp(b.dateStarted) - stamp(a.dateStarted));
	const last = books
		.filter((b) => b.status === 'finished')
		.sort((a, b) => stamp(b.dateFinished) - stamp(a.dateFinished))[0];
	const label = current.length ? 'Reading' : 'Last read';
	const shown = (current.length ? current : last ? [last] : []).slice(0, 2);
	const stack = await Promise.all(
		shown.map(async (book) => {
			const cover = coverFor(book.edition.isbn13);
			return { book, cover, glow: cover ? await glowOf(cover) : undefined };
		}),
	);
	const captions = stack.map(({ book }) => ({ title: short(book.title), author: book.author }));
	return (
		<Tile
			label={label}
			href="/reading/"
			data-deck-host=""
			span={{ md: 3, lg: 4 }}
			order={order}
			meta={current.length > 1 ? `+${current.length - 1} more` : undefined}
		>
			{captions.length > 0 && (
				<div className="reading-tile-body">
					<Deck
						id="reading-deck"
						captions={captions}
						hold="pull"
						lift={9}
						host="ancestor"
						className="reading-deck"
						cards={stack.map(({ book, cover, glow }) => (
							<Plate key={book.edition.isbn13} src={cover} alt="" glow={glow} sizes="12rem">
								<span className="grid h-full content-end gap-1 p-3 text-left text-small text-muted-foreground">
									<strong className="font-semibold text-foreground">{short(book.title)}</strong>
									<span>{book.author}</span>
								</span>
							</Plate>
						))}
					/>
					<div className="mt-4">
						<Readout
							follow="reading-deck"
							field="title"
							value={captions[0].title}
							className="text-ui font-semibold leading-normal"
						/>
						<Readout
							follow="reading-deck"
							field="author"
							value={captions[0].author}
							className="text-small leading-normal text-muted-foreground"
						/>
					</div>
				</div>
			)}
		</Tile>
	);
}
