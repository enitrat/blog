import type { ReactNode } from 'react';
import { type Book, books } from '@/booksData';
import { piles } from '@/components/books/book';
import { BookLine } from '@/components/books/book-line';
import { BookPile } from '@/components/books/book-pile';
import { Eyebrow } from '@/components/ui/eyebrow';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';
import { StatusDot } from '@/components/ui/status-dot';
import { pageMeta } from '@/lib/meta';
import { cn } from '@/lib/utils';
import { readingLine } from '@/utils/bookRecord';

export const metadata = pageMeta({
	title: 'Reading',
	description: 'What I am reading, and everything I have read.',
	path: '/reading/',
});

const BEFORE_2020 = 'Before 2020';

const time = (b: Book) => (b.dateFinished ?? b.dateStarted)?.getTime() ?? 0;
const recent = (a: Book, b: Book) => time(b) - time(a);
const short = (t: string) => t.replace(/\s*\(.*\)\s*$/, '');
const pile = (bs: Book[]) => bs.map((b) => ({ title: b.title, pages: b.edition.pageCount }));
// Author, date and rating only: the page stays a reading log, not a catalogue.
const when = (b: Book) =>
	[readingLine(b), b.rating === null ? undefined : `rated ${b.rating} of 5`]
		.filter(Boolean)
		.join(' · ');
const yearOf = (b: Book) =>
	b.readingPeriod === 'before-2020'
		? BEFORE_2020
		: String((b.dateFinished ?? b.dateStarted)?.getFullYear());
const writingHref = (b: Book) => (b.writingSlug ? `/writing/${b.writingSlug}/` : undefined);
const id = (s: string) => `y-${s.toLowerCase().replace(/\W+/g, '-')}`;

/** A book with its own small pile beside its title line. */
function BookItem({ book, children }: { book: Book; children: ReactNode }) {
	return (
		<li className="flex items-center gap-4">
			<div className="flex-none">
				<BookPile books={pile([book])} scale={3} />
			</div>
			<div>{children}</div>
		</li>
	);
}

export default function ReadingPage() {
	const reading = books.filter((b) => b.status === 'reading').sort(recent);
	const finished = books.filter((b) => b.status === 'finished');
	const picks = finished
		.filter((b) => (b.rating ?? 0) >= 4)
		.sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0))
		.slice(0, 6);
	const unrated = finished.filter((b) => b.rating === null).length;
	// The archive by year, newest first; a book known only as "before 2020" closes it.
	const years = [...Map.groupBy([...finished].sort(recent), yearOf)].map(([year, bs]) => ({
		year,
		books: bs,
		// Oldest at the bottom: the pile grows as the year goes on.
		piles: piles(pile([...bs].reverse())),
		// Books before 2020 fold away behind their heading, closed by default.
		folded: year === BEFORE_2020,
	}));
	const wanted = books.filter((b) => b.status === 'want-to-read').sort(recent);

	return (
		<Page>
			<PageHeader>Reading</PageHeader>

			{reading.length > 0 && (
				<section aria-label="Reading now">
					<ul className="grid gap-4">
						{reading.map((b) => (
							<BookItem key={b.title} book={b}>
								<BookLine title={b.title} author={b.author} size="lg" tag={<StatusDot />} />
							</BookItem>
						))}
					</ul>
				</section>
			)}

			<section aria-labelledby="picks">
				<div id="picks" className="mb-4">
					<Eyebrow
						as="h2"
						meta={
							<>
								[TK] {unrated} of {finished.length} unrated
							</>
						}
					>
						[TK] Rated highest
					</Eyebrow>
				</div>
				<ul className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,18rem),1fr))] gap-4">
					{picks.map((b) => (
						<BookItem key={b.title} book={b}>
							<BookLine title={b.title} author={b.author}>
								{when(b)}
							</BookLine>
						</BookItem>
					))}
				</ul>
			</section>

			{years.map(({ year, books: bs, piles: ps, folded }) => {
				const Wrap = folded ? 'details' : 'div';
				const Head = folded ? 'summary' : 'div';
				const link = 'underline underline-offset-[22%] hover:text-foreground';
				return (
					<section key={year} aria-labelledby={id(year)}>
						<Wrap className="group">
							<Head
								id={id(year)}
								className={
									folded
										? 'cursor-pointer list-none rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden'
										: undefined
								}
							>
								<Eyebrow
									as="h2"
									className="mb-4"
									meta={
										<>
											{bs.length}
											{folded && (
												<>
													<span className={cn('ms-3 group-open:hidden', link)}>[TK] show</span>
													<span className={cn('ms-3 hidden group-open:inline', link)}>
														[TK] hide
													</span>
												</>
											)}
										</>
									}
								>
									{year}
								</Eyebrow>
							</Head>
							<div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:items-start">
								<div className="flex flex-wrap items-end gap-x-4 gap-y-2">
									{ps.map((p) => (
										<BookPile key={p[0].title} books={p} />
									))}
								</div>
								<ListRows>
									{bs.map((b) => (
										<ListRow
											key={b.title}
											title={short(b.title)}
											description={`${b.author} · ${when(b)}`}
											href={writingHref(b)}
										/>
									))}
								</ListRows>
							</div>
						</Wrap>
					</section>
				);
			})}

			{wanted.length > 0 && (
				<section aria-labelledby="g-want">
					<div id="g-want" className="mb-4">
						<Eyebrow as="h2" meta={wanted.length}>
							Reading list
						</Eyebrow>
					</div>
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
