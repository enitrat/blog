/**
 * A pile of books seen from the front, spines out: each book is as thick as its page count, at
 * one scale for every pile on the site, so piles compare by weight.
 */

/** Pages per world unit of thickness, the same for every book, so piles compare honestly. */
export const PAGES_PER_UNIT = 40;
/** The tallest pile, in world units; a longer run of books starts a new pile beside it. */
export const PILE_MAX = 64;

/** Spine length in world units, before the per-book trim. */
const LEN = 30;
/** How far a book may sit off the one under it, per step of its hash. */
const SHIFT = 0.6;

export interface PileBook {
	title: string;
	pages: number;
}

export const thickness = (pages: number) => pages / PAGES_PER_UNIT;

/** Cuts a run of books into piles no taller than `PILE_MAX`; a single taller book stands alone. */
export function piles<T extends PileBook>(books: T[]): T[][] {
	const out: T[][] = [];
	let height = Number.POSITIVE_INFINITY;
	for (const book of books) {
		const t = thickness(book.pages);
		if (height + t > PILE_MAX) {
			out.push([]);
			height = 0;
		}
		out[out.length - 1].push(book);
		height += t;
	}
	return out;
}

const hash = (s: string) => {
	let h = 0;
	for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
	return h;
};

/** One book's place in a pile, in world units, bottom of the pile first. */
export function stack(books: PileBook[]) {
	return books.map((book) => {
		const h = hash(book.title);
		return {
			title: book.title.replace(/\s*\(.*\)\s*$/, ''),
			tone: h % 3,
			thickness: thickness(book.pages),
			// A hand-made pile: each book a little shorter or longer, a little off the one under it.
			length: LEN - (h % 4),
			offset: (h % 5) * SHIFT,
		};
	});
}

/** The widest a pile gets, in world units, so every pile reserves the same width. */
export const PILE_WIDTH = LEN + 4 * SHIFT;
