import type { ReactNode } from 'react';
import { type Book, books } from '@/booksData';
import {
	BookLine,
	BookPile,
	Label,
	ListRow,
	ListRows,
	Page,
	PageTitle,
	piles,
	StatusDot,
} from '@/ds';
import { pageMeta } from '@/lib/meta';
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
		<li className="book flex items-center gap-4">
			<div className="cover flex-none">
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
		<div data-slot="reading-page">
			<Page>
				<PageTitle>Reading</PageTitle>

				{reading.length > 0 && (
					<section aria-label="Reading now">
						<ul className="books m-0 grid list-none gap-6 p-0">
							{reading.map((b) => (
								<BookItem key={b.title} book={b}>
									<BookLine title={b.title} author={b.author} size="lg" tag={<StatusDot />} />
								</BookItem>
							))}
						</ul>
					</section>
				)}

				<section aria-labelledby="picks">
					<Label
						as="h2"
						id="picks"
						space="sm"
						meta={
							<>
								[TK] {unrated} of {finished.length} unrated
							</>
						}
					>
						[TK] Rated highest
					</Label>
					<ul className="books picks m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(min(100%,var(--reading-pick-min)),1fr))] gap-6 p-0">
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
					return (
						<section key={year} aria-labelledby={id(year)}>
							<Wrap>
								<Head>
									<Label
										as="h2"
										id={id(year)}
										space="sm"
										meta={
											<>
												{bs.length}
												{folded && (
													<>
														<span className="toggle ms-3 underline" data-show="">
															[TK] show
														</span>
														<span className="toggle ms-3 underline" data-hide="">
															[TK] hide
														</span>
													</>
												)}
											</>
										}
									>
										{year}
									</Label>
								</Head>
								<div className="year grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:items-start">
									<div className="piles flex flex-wrap items-end gap-x-4 gap-y-2">
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
						<Label as="h2" id="g-want" space="sm" meta={wanted.length}>
							Reading list
						</Label>
						<ListRows>
							{wanted.map((b) => (
								<ListRow key={b.title} title={short(b.title)} description={b.author} />
							))}
						</ListRows>
					</section>
				)}
			</Page>
		</div>
	);
}
