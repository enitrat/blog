import { expect, test } from 'bun:test';
import { fanZone, pullZone } from './deck-poses';

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

test('fanZone: the open footprint splits evenly by x', () => {
	// three cards at spread .26: the hand runs from -26% to 126% of the front card's box
	expect(fanZone(-20, 3, 0.26)).toBe(0);
	expect(fanZone(50, 3, 0.26)).toBe(1);
	expect(fanZone(120, 3, 0.26)).toBe(2);
	expect(fanZone(-400, 3, 0.26)).toBe(0);
	expect(fanZone(400, 3, 0.26)).toBe(2);
});
