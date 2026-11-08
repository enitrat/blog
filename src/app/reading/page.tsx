import type { StaticImageData } from 'next/image';
import { type Book, books } from '@/booksData';
import { BookRow, BookThumb } from '@/components/books/book-row';
import { YearStrip } from '@/components/books/year-strip';
import { Eyebrow } from '@/components/ui/eyebrow';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';
import { StatusDot } from '@/components/ui/status-dot';
import { findImage } from '@/lib/media';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Reading',
	description: 'What I am reading, and everything I have read.',
	path: '/reading/',
});

const BEFORE_2020 = 'Before 2020';
/** A rating at or above this is set in the accent, in its row and in the year's strip. */
const HIGH = 4.2;

const MONTH = new Intl.DateTimeFormat('en-GB', { month: 'short' });
const COUNT = new Intl.NumberFormat('en-GB');

const date = (b: Book) => b.dateFinished ?? b.dateStarted;
const time = (b: Book) => date(b)?.getTime() ?? 0;
const recent = (a: Book, b: Book) => time(b) - time(a);
const short = (t: string) => t.replace(/\s*\(.*\)\s*$/, '');
const pages = (bs: Book[]) => bs.reduce((s, b) => s + b.edition.pageCount, 0);
const high = (b: Book) => (b.rating ?? 0) >= HIGH;
const yearOf = (b: Book) =>
	b.readingPeriod === 'before-2020' ? BEFORE_2020 : String(date(b)?.getFullYear());
const tally = (bs: Book[]) =>
	`${bs.length} ${bs.length === 1 ? 'book' : 'books'} · ${COUNT.format(pages(bs))} pages`;
const id = (s: string) => `y-${s.toLowerCase().replace(/\W+/g, '-')}`;

type Shown = { book: Book; cover: StaticImageData | undefined };
const withCover = (b: Book): Promise<Shown> =>
	findImage(`covers/${b.edition.isbn13}`).then((cover) => ({ book: b, cover }));

const meta = 'font-mono text-xs tracking-[-0.01em] text-faint tabular-nums';

export default async function ReadingPage() {
	const reading = await Promise.all(
		books
			.filter((b) => b.status === 'reading')
			.sort(recent)
			.map(withCover),
	);
	const finished = await Promise.all(
		books
			.filter((b) => b.status === 'finished')
			.sort(recent)
			.map(withCover),
	);
	const picks = finished
		.filter(({ book }) => (book.rating ?? 0) >= 4)
		.sort((a, b) => (b.book.rating ?? 0) - (a.book.rating ?? 0))
		.slice(0, 6);
	// The archive by year, newest first; books known only as "before 2020" close it, folded.
	const years = [...Map.groupBy(finished, ({ book }) => yearOf(book))].map(([year, shown]) => ({
		year,
		shown,
		folded: year === BEFORE_2020,
	}));
	const wanted = books.filter((b) => b.status === 'want-to-read').sort(recent);
	const all = finished.map(({ book }) => book);

	return (
		<Page>
			<PageHeader
				lead={
					<span
						className={meta}
					>{`${all.length} finished · ${COUNT.format(pages(all))} pages`}</span>
				}
			>
				Reading
			</PageHeader>

			{reading.length > 0 && (
				<section aria-label="Reading now" className="grid gap-5">
					{reading.map(({ book, cover }) => (
						<div key={book.title} className="grid grid-cols-[auto_minmax(0,1fr)] items-end gap-5">
							<BookThumb cover={cover} title={book.title} className="w-24 shadow-book sm:w-30" />
							<div className="min-w-0">
								<h2 className="text-2xl leading-tight font-semibold tracking-tight text-balance text-foreground sm:text-3xl">
									{short(book.title)}
								</h2>
								<p className="mt-1.5 text-sm text-muted-foreground">{book.author}</p>
								<p className={`mt-3 flex items-center gap-2 ${meta}`}>
									<StatusDot />
									Reading now · {book.edition.pageCount} pages
								</p>
							</div>
						</div>
					))}
				</section>
			)}

			<section aria-labelledby="picks">
				<Eyebrow as="h2" id="picks" className="mb-4">
					Rated highest
				</Eyebrow>
				<ul className="grid grid-cols-3 gap-x-4 gap-y-6 sm:grid-cols-6">
					{picks.map(({ book, cover }) => (
						<li key={book.title} className="min-w-0">
							<BookThumb cover={cover} title={book.title} className="w-full shadow-book" />
							<p className="mt-2.5 text-sm leading-snug font-semibold text-balance text-foreground">
								{short(book.title)}
							</p>
							<p className={`mt-1 ${meta}`}>
								{book.author.split(' ').at(-1)} · {book.rating}
							</p>
						</li>
					))}
				</ul>
			</section>

			{years.map(({ year, shown, folded }) => {
				const head = (
					<>
						<div className="flex items-baseline justify-between gap-4">
							<h2
								id={id(year)}
								className="text-2xl leading-none font-semibold tracking-tight text-foreground tabular-nums"
							>
								{year}
							</h2>
							<span className={meta}>
								{tally(shown.map((s) => s.book))}
								{folded && (
									<span className="ms-3 underline underline-offset-[22%] group-hover:text-foreground">
										<span className="group-open:hidden">Show</span>
										<span className="hidden group-open:inline">Hide</span>
									</span>
								)}
							</span>
						</div>
						<div className="mt-3 mb-1">
							<YearStrip
								books={[...shown].reverse().map(({ book }) => ({
									title: short(book.title),
									pages: book.edition.pageCount,
									high: high(book),
								}))}
							/>
						</div>
					</>
				);
				const rows = (
					<ol>
						{shown.map(({ book, cover }) => {
							const d = date(book);
							return (
								<BookRow
									key={book.title}
									title={short(book.title)}
									author={book.author}
									cover={cover}
									month={folded || !d ? undefined : MONTH.format(d)}
									datetime={d?.toISOString().slice(0, 7)}
									rating={book.rating}
									high={high(book)}
								/>
							);
						})}
					</ol>
				);
				return (
					<section key={year} aria-labelledby={id(year)}>
						{folded ? (
							<details className="group">
								<summary className="cursor-pointer list-none rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
									{head}
								</summary>
								{rows}
							</details>
						) : (
							<>
								{head}
								{rows}
							</>
						)}
					</section>
				);
			})}

			{wanted.length > 0 && (
				<section aria-labelledby="g-want">
					<Eyebrow as="h2" id="g-want" className="mb-4" meta={wanted.length}>
						Reading list
					</Eyebrow>
					<ListRows>
						{wanted.map((b) => (
							<ListRow key={b.title} title={short(b.title)} description={b.author} />
						))}
					</ListRows>
				</section>
			)}
		</Page>
	);
}
