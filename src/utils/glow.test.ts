import { expect, test } from 'bun:test';
import { glowOf } from './glow';

test('glowOf returns an rgb() colour', async () => {
	const file = new URL('../assets/covers/9780143129486.jpg', import.meta.url).pathname;
	expect(await glowOf({ file })).toMatch(/^rgb\(\d+ \d+ \d+\)$/);
});
