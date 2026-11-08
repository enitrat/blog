import { expect, test } from 'bun:test';
import { PILE_MAX, PILE_WIDTH, piles, stack, thickness } from './book';

test('a pile draws every book, bottom first, without its series', () => {
	const s = stack([
		{ title: 'A (Series, 1)', pages: 600 },
		{ title: 'B', pages: 100 },
	]);
	expect(s.map((b) => b.title)).toEqual(['A', 'B']);
	expect(s.map((b) => b.thickness)).toEqual([thickness(600), thickness(100)]);
});

test('a book is as thick as its pages, at one scale', () => {
	expect(thickness(800)).toBe(2 * thickness(400));
});

test('every book fits the pile width', () => {
	const s = stack(['a', 'bb', 'ccc', 'dddd', 'eeeee'].map((title) => ({ title, pages: 200 })));
	for (const b of s) {
		expect(b.offset).toBeGreaterThanOrEqual(0);
		expect(b.offset + b.length).toBeLessThanOrEqual(PILE_WIDTH);
	}
});

test('piles stay under the cap and keep every book in order', () => {
	const books = [900, 300, 1312, 120, 400, 2000, 50].map((pages, i) => ({ title: `b${i}`, pages }));
	const out = piles(books);
	expect(out.flat()).toEqual(books);
	for (const p of out)
		if (p.length > 1)
			expect(p.reduce((z, b) => z + thickness(b.pages), 0)).toBeLessThanOrEqual(PILE_MAX);
});
