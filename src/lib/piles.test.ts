import { expect, test } from 'bun:test';
import { pile, type ShelfBook, SORTS } from './piles';

const book = (isbn: string, over: Partial<ShelfBook> = {}): ShelfBook => ({
	isbn,
	title: `Title ${isbn}`,
	author: 'A',
	pages: 300,
	status: 'finished',
	year: '2025',
	rating: null,
	favourite: false,
	leather: '#000',
	...over,
});

const shelf = [
	book('1', { status: 'reading', author: 'R' }),
	book('2', { author: 'Dostoevsky', pages: 120, rating: 4.5, favourite: true }),
	book('3', { author: 'Dostoevsky', pages: 800, year: 'Before 2020' }),
	book('4', { author: 'Camus', rating: 3.9 }),
	book('5', { author: 'Kafka', status: 'want-to-read' }),
];

test('every shelved book lands in exactly one pile, whatever the sort', () => {
	for (const { id } of SORTS) {
		const isbns = pile(id, shelf).flatMap((p) => p.books.map((b) => b.isbn));
		expect(isbns.sort()).toEqual(['1', '2', '3', '4']);
	}
});

test('by author, the books being read come first and lone authors share a pile', () => {
	expect(pile('author', shelf).map((p) => [p.label, p.books.length])).toEqual([
		['Reading now', 1],
		['Dostoevsky', 2],
		['One book each', 1],
	]);
});

test('empty piles are left out', () => {
	const labels = pile('length', shelf).map((p) => p.label);
	expect(labels).toEqual(['Under 200 pages', '200 to 400 pages', 'Over 700 pages']);
	expect(pile('author', shelf.slice(1)).map((p) => p.label)).not.toContain('Reading now');
});
