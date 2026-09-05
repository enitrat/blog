import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium, webkit } from 'playwright';

// Run against `bun run prototype`; screenshots remain outside the repository.
const base = process.env.PROTOTYPE_URL ?? 'http://127.0.0.1:4321';
const output = '/tmp/cozy-library-check';
await mkdir(output, { recursive: true });
for (const engine of [chromium, webkit]) {
	const browser = await engine.launch();
	try {
		const sizes =
			engine === chromium
				? [
						[1440, 900],
						[979, 900],
						[760, 900],
						[390, 844],
						[320, 568],
					]
				: [
						[1440, 900],
						[390, 844],
					];
		for (const [width, height] of sizes) {
			const page = await browser.newPage({ viewport: { width, height }, hasTouch: width < 600 });
			const errors = [];
			page.on('pageerror', (error) => errors.push(error.message));
			await page.goto(`${base}/bookshelf/prototype/`);
			await page.waitForSelector('#library-loading[hidden]', { state: 'attached' });
			assert.equal(await page.locator('.book-target').count(), 55);
			assert.equal(
				await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),
				false,
			);
			assert.equal(
				await page.evaluate(() => document.documentElement.scrollHeight > innerHeight + 80),
				true,
				'The oak case continues below the first viewport',
			);
			await page.screenshot({ path: `${output}/${engine.name()}-${width}-shelf.png` });
			await page.getByRole('button', { name: /Open Marche ou crève/ }).scrollIntoViewIfNeeded();
			await page.waitForTimeout(250);
			await page.screenshot({ path: `${output}/${engine.name()}-${width}-shelf-end.png` });
			await page.evaluate(() => scrollTo(0, 0));
			await page.waitForTimeout(200);
			const book = page.getByRole('button', { name: /Open White Nights/ });
			await book.focus();
			await page.keyboard.press('Enter');
			await page.waitForSelector('#book-reader[data-phase="reading"]');
			await page.waitForFunction(() =>
				document.getElementById('page-status')?.textContent?.startsWith('Page 1'),
			);
			await page.waitForTimeout(550);
			const reader = page.frames().find((frame) => frame.url().includes('prototype-reader'));
			assert.ok(reader, 'Reader has its own lifetime');
			assert.equal(
				await reader
					.locator('.page-inner')
					.evaluateAll((nodes) =>
						nodes.some(
							(node) => node.clientHeight > 0 && node.scrollHeight > node.clientHeight + 2,
						),
					),
				false,
				'Sample text fits the page',
			);
			await page.screenshot({ path: `${output}/${engine.name()}-${width}-open.png` });
			await page.getByRole('button', { name: 'Next page', exact: true }).click();
			await page.waitForFunction(
				() => !document.getElementById('page-status')?.textContent?.startsWith('Page 1'),
			);
			await page.waitForTimeout(550);
			await page.getByRole('button', { name: 'Previous page', exact: true }).click();
			await page.waitForTimeout(550);
			await page.keyboard.press('Escape');
			await page.waitForSelector('#book-reader', { state: 'hidden' });
			assert.equal(
				await page.locator('#reader-mount iframe, .book-flight').count(),
				0,
				'Closing removes both renderers',
			);
			assert.equal(
				await book.evaluate((node) => document.activeElement === node),
				true,
				'Focus returns to the book',
			);
			// Closing during extraction must not let a stale callback reopen the reader.
			await book.click();
			await page.getByRole('button', { name: /Return to shelf/ }).click();
			await page.waitForSelector('#book-reader', { state: 'hidden' });
			await page.waitForTimeout(650);
			assert.equal(await page.locator('#book-reader').evaluate((node) => node.open), false);
			await page.emulateMedia({ reducedMotion: 'reduce' });
			await book.click();
			await page.waitForSelector('#book-reader[data-phase="reading"]');
			await page.keyboard.press('Escape');
			await page.waitForSelector('#book-reader', { state: 'hidden' });
			assert.deepEqual(errors, []);
			console.log(
				`${engine.name()} ${width}×${height}: open, turn, close, interrupt, reduced motion passed`,
			);
			await page.close();
		}
		const plain = await browser.newPage({ javaScriptEnabled: false });
		await plain.goto(`${base}/bookshelf/prototype/`);
		await plain.getByRole('link', { name: 'Browse the reading archive', exact: true }).click();
		assert.match(plain.url(), /\/bookshelf\/?$/);
		assert.equal(await plain.getByRole('heading', { name: 'Bookshelf', exact: true }).count(), 1);
		await plain.close();
	} finally {
		await browser.close();
	}
}
console.log(`Screenshots: ${output}`);
