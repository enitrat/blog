import { expect, test } from 'bun:test';
import { glowOf } from './glow';

test('glowOf returns an rgb() colour', async () => {
	const fsPath = new URL('../assets/covers/9780143129486.jpg', import.meta.url).pathname;
	expect(await glowOf({ fsPath } as never)).toMatch(/^rgb\(\d+ \d+ \d+\)$/);
});
