import { type Book, books } from '@/booksData';
import { PileTable } from '@/components/books/pile-table';
import { Eyebrow } from '@/components/ui/eyebrow';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';
import { shortTitle } from '@/lib/books';
import { getNotes } from '@/lib/content';
import { formatCount, formatMonthShortYear } from '@/lib/format';
import { findImage } from '@/lib/media';
import { pageMeta } from '@/lib/meta';
import { FAVOURITE, type ShelfBook } from '@/lib/piles';
import { PLEIADE_HEX, pleiadeStyleFor } from '@/utils/pleiade';

export const metadata = pageMeta({
	title: 'Reading',
	description: 'What I am reading, and everything I have read.',
	path: '/reading/',
});

const date = (b: Book) => b.dateFinished ?? b.dateStarted;
const recent = (a: Book, b: Book) => (date(b)?.getTime() ?? 0) - (date(a)?.getTime() ?? 0);
const BEFORE_2020 = 'Before 2020';

async function shelve(b: Book): Promise<ShelfBook> {
	const d = b.readingPeriod === 'before-2020' ? null : date(b);
	return {
		isbn: b.edition.isbn13,
		title: shortTitle(b.title),
		author: b.author,
		pages: b.edition.pageCount,
		status: b.status,
		year: d ? String(d.getUTCFullYear()) : BEFORE_2020,
		when: d ? formatMonthShortYear(d) : undefined,
		rating: b.rating,
		favourite: (b.rating ?? 0) >= FAVOURITE,
		leather: PLEIADE_HEX[pleiadeStyleFor(b.author).color],
		cover: await findImage(`covers/${b.edition.isbn13}`),
		note: notes.get(b.edition.isbn13),
	};
}

const notes = new Map(getNotes().map((n) => [n.id, n.body.trim()]));

export default async function ReadingPage() {
	const sorted = [...books].sort(recent);
	const shelved = await Promise.all(sorted.filter((b) => b.status !== 'want-to-read').map(shelve));
	const finished = sorted.filter((b) => b.status === 'finished');
	const pages = finished.reduce((s, b) => s + b.edition.pageCount, 0);
	const wanted = sorted.filter((b) => b.status === 'want-to-read');

	return (
		<Page>
			<PageHeader
				lead={
					<span className="type-meta text-faint">{`${finished.length} finished · ${formatCount(pages)} pages`}</span>
				}
			>
				Reading
			</PageHeader>
			<PileTable books={shelved} />
			{wanted.length > 0 && (
				<section aria-labelledby="g-want">
					<Eyebrow as="h2" id="g-want" className="mb-4" meta={wanted.length}>
						Reading list
					</Eyebrow>
					<ListRows>
						{wanted.map((b) => (
							<ListRow key={b.title} title={shortTitle(b.title)} description={b.author} />
						))}
					</ListRows>
				</section>
			)}
		</Page>
	);
}
