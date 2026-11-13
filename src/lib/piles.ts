import type { StaticImageData } from 'next/image';
import type { BookStatus } from '@/booksData';

/** A book as the Reading page shows it: everything already formatted, safe to hand to the client. */
export type ShelfBook = {
	isbn: string;
	title: string;
	author: string;
	pages: number;
	status: BookStatus;
	/** "2026", or "Before 2020" for books known only by period. */
	year: string;
	/** "Aug 2026"; absent for books known only by period. */
	when?: string;
	rating: number | null;
	/** A favourite: rated at or above `FAVOURITE`. */
	favourite: boolean;
	/** The author's Pléiade leather, shown when no cover is checked in. */
	leather: string;
	cover?: StaticImageData;
	note?: string;
};

export const FAVOURITE = 4.2;

export type Pile = { label: string; books: ShelfBook[]; reading?: boolean };

export const SORTS = [
	{ id: 'author', label: 'Author' },
	{ id: 'length', label: 'Length' },
	{ id: 'rating', label: 'Rating' },
	{ id: 'year', label: 'Year' },
] as const;

export type PileSort = (typeof SORTS)[number]['id'];

const READING = 'Reading now';
const ONE_EACH = 'One book each';

const LENGTHS = [
	{ label: 'Under 200 pages', below: 200 },
	{ label: '200 to 400 pages', below: 400 },
	{ label: '400 to 700 pages', below: 700 },
	{ label: 'Over 700 pages', below: Number.POSITIVE_INFINITY },
];

const groups = (books: ShelfBook[], key: (b: ShelfBook) => string): Pile[] =>
	[...Map.groupBy(books, key)].map(([label, books]) => ({ label, books }));

/**
 * The piles for one sort. Every shelved book lands in exactly one pile; empty piles are dropped.
 * `books` arrive most recent first, and each pile keeps that order.
 */
export function pile(sort: PileSort, books: ShelfBook[]): Pile[] {
	const reading = books.filter((b) => b.status === 'reading');
	const finished = books.filter((b) => b.status === 'finished');
	const shelved = [...reading, ...finished];
	const piles = ((): Pile[] => {
		switch (sort) {
			case 'author': {
				const authors = groups(finished, (b) => b.author).sort(
					(a, b) => b.books.length - a.books.length,
				);
				return [
					{ label: READING, books: reading, reading: true },
					...authors.filter((p) => p.books.length > 1),
					{
						label: ONE_EACH,
						books: authors.filter((p) => p.books.length === 1).flatMap((p) => p.books),
					},
				];
			}
			case 'year':
				return [
					{ label: READING, books: reading, reading: true },
					...groups(finished, (b) => b.year),
				];
			case 'length':
				return LENGTHS.map((l, i) => ({
					label: l.label,
					books: shelved.filter(
						(b) => b.pages < l.below && b.pages >= (LENGTHS[i - 1]?.below ?? 0),
					),
				}));
			case 'rating':
				return [
					{ label: 'Favourites', books: shelved.filter((b) => b.favourite) },
					{ label: 'Rated', books: shelved.filter((b) => b.rating !== null && !b.favourite) },
					{ label: 'Not rated yet', books: shelved.filter((b) => b.rating === null) },
				];
		}
	})();
	return piles.filter((p) => p.books.length > 0);
}
