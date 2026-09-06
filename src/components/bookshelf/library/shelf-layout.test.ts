import { expect, test } from 'bun:test';
import { packShelves } from './shelf-layout';

const widths = (shelves: number[][]) => shelves.map((row) => row.reduce((a, b) => a + b, 0));
const pack = (items: number[], rows: number) => packShelves(items, (n) => n, rows);

test('keeps every item, in order', () => {
	const items = [3, 1, 4, 1, 5, 9, 2, 6];
	expect(pack(items, 3).flat()).toEqual(items);
});

test('fills every shelf rather than leaving a stub row', () => {
	// Greedy-to-average would put 5 on the first shelf and strand the last.
	expect(pack([1, 1, 1, 1, 1], 3).every((row) => row.length > 0)).toBe(true);
	expect(pack([1, 1, 1, 1, 1], 3)).toHaveLength(3);
});

test('balances by width, not by count', () => {
	// Four thin spines weigh the same as one thick one.
	const [a, b] = widths(pack([1, 1, 1, 1, 4], 2));
	expect(Math.abs(a - b)).toBeLessThanOrEqual(1);
});

test('never exceeds the requested number of shelves', () => {
	for (let rows = 1; rows <= 8; rows++) {
		const items = Array.from({ length: 55 }, (_, i) => 0.5 + ((i * 7) % 11) / 20);
		expect(pack(items, rows).length).toBeLessThanOrEqual(rows);
	}
});

test('degenerate inputs stay empty rather than throwing', () => {
	expect(pack([], 3)).toEqual([]);
	expect(pack([1, 2], 0)).toEqual([]);
});
