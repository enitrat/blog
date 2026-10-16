import { expect, test } from 'bun:test';
import { pullZone } from './deck';

test('pullZone: the front book wins where it is hit, the peeking strip pulls the back one', () => {
	expect(pullZone(40, 50, 2)).toBe(0);
	// n=2: the back book rests 12% right and 7% up of the front one, so its strip is right of the front box
	expect(pullZone(105, 10, 2)).toBe(1);
	// off the pile, the offset direction decides
	expect(pullZone(150, -20, 2)).toBe(1);
	expect(pullZone(-50, 120, 2)).toBe(0);
	// three books: far off the pile, beside the middle book's centre line, the middle one; the pile is centred on the front card's box
	expect(pullZone(120, 170, 3)).toBe(1);
	expect(pullZone(-300, 400, 3)).toBe(0);
	expect(pullZone(400, -300, 3)).toBe(2);
});
