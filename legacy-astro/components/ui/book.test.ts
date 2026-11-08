import { expect, test } from 'bun:test';
import { PILE_MAX, piles, spineTitle, stack, thickness } from './book';

test('a pile draws every book, with fore-edge leaves on thick ones', () => {
	const s = stack([
		{ title: 'A', pages: 600 },
		{ title: 'B', pages: 100 },
	]);
	expect(s.map((b) => b.leaves.length)).toEqual([4, 0]);
});

test('a book is as thick as its pages, at one scale', () => {
	expect(thickness(800)).toBe(2 * thickness(400));
});

test('piles stay under the cap and keep every book in order', () => {
	const books = [900, 300, 1312, 120, 400, 2000, 50].map((pages, i) => ({ title: `b${i}`, pages }));
	const out = piles(books);
	expect(out.flat()).toEqual(books);
	for (const p of out)
		if (p.length > 1)
			expect(p.reduce((z, b) => z + thickness(b.pages), 0)).toBeLessThanOrEqual(PILE_MAX);
});

test('a long title is cut to its spine with an ellipsis', () => {
	expect(spineTitle('Short', 30, 3)).toBe('Short');
	expect(spineTitle('The Unbearable Lightness of Being', 20, 3)).toMatch(/…$/);
});
