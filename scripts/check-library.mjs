/**
 * Browser checks for the /bookshelf/ bookcase.
 *
 *   bun run library:check                 every configuration
 *   bun run library:check chromium        one engine
 *   bun run library:check webkit 390      one case, while iterating
 *
 * Starts its own dev server unless LIBRARY_URL points at one already.
 *
 * Each case gets its OWN browser. Sharing one browser per engine used to pile
 * ~28 WebGL contexts into a single process — the shelf plus one per book
 * opened, across every viewport. Past WebKit's per-process limit it does not
 * error, it just blocks: 20+ minute stalls with no request ever leaving the
 * page. A fresh browser per case runs the same assertions in seconds.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { chromium, webkit } from 'playwright';
import { books } from '../src/booksData.ts';

const OUTPUT = '/tmp/library-check';
/* Two at a time. Each case drives real WebGL, and at three the machine starves
   a page badly enough that WebKit throttles its requestAnimationFrame — the
   return animation stalls mid-flight and the dialog never reaches its terminal
   phase. That is starvation, not a defect: the app resumes correctly because
   motion is driven by absolute timestamps. Three was also slower, since the
   starved case sat burning step timeouts. */
const CONCURRENCY = Number(process.env.LIBRARY_CONCURRENCY ?? 2);
const PORT = 4321;
/* Per-action and whole-case ceilings. Browser launch, newPage and close have no
   Playwright timeout of their own, and on this machine they do occasionally
   wedge for tens of minutes. A check that can hang forever is not a check. */
const STEP_TIMEOUT = 20_000;
const CASE_TIMEOUT = 120_000;

const withTimeout = (promise, ms, label) => {
	let timer;
	return Promise.race([
		promise,
		new Promise((_, reject) => {
			timer = setTimeout(() => reject(new Error(`timed out after ${ms / 1000}s (${label})`)), ms);
		}),
	]).finally(() => clearTimeout(timer));
};

/** Viewports per engine. WebKit covers the extremes; Chromium covers the range. */
const CASES = [
	{ engine: chromium, width: 1440, height: 900 },
	{ engine: chromium, width: 979, height: 900 },
	{ engine: chromium, width: 760, height: 900 },
	{ engine: chromium, width: 390, height: 844 },
	{ engine: chromium, width: 320, height: 568 },
	{ engine: webkit, width: 1440, height: 900 },
	{ engine: webkit, width: 390, height: 844 },
];

const reachable = async (url) => {
	try {
		const response = await fetch(url, { signal: AbortSignal.timeout(2000) });
		return response.ok;
	} catch {
		return false;
	}
};

/** Use the caller's server if they gave one, otherwise run a private one. */
async function serve() {
	if (process.env.LIBRARY_URL) {
		const base = process.env.LIBRARY_URL.replace(/\/$/, '');
		assert.ok(await reachable(`${base}/bookshelf/`), `No dev server at ${base}`);
		return { base, stop: () => {} };
	}
	const base = `http://127.0.0.1:${PORT}`;
	if (await reachable(`${base}/bookshelf/`)) return { base, stop: () => {} };
	// A stale Vite dep cache serves 504s that surface as unrelated selector
	// timeouts, so a fresh server starts from a cleared one.
	const server = spawn(
		'bun',
		['x', 'astro', 'dev', '--host', '127.0.0.1', '--port', String(PORT), '--force'],
		{ stdio: 'ignore', detached: false },
	);
	for (let attempt = 0; attempt < 60; attempt++) {
		if (await reachable(`${base}/bookshelf/`)) {
			console.log(`dev server ready at ${base}`);
			return { base, stop: () => server.kill() };
		}
		await new Promise((resolve) => setTimeout(resolve, 1000));
	}
	server.kill();
	throw new Error('Dev server did not come up within 60s');
}

/** Run one engine/viewport pair. Throws with the step that failed. */
async function check({ engine, width, height }, base) {
	const name = `${engine.name()} ${width}×${height}`;
	const tag = `${engine.name()}-${width}`;
	const browser = await withTimeout(engine.launch(), STEP_TIMEOUT, `${name} launch`);
	let page;
	let step = 'open the page';
	const at = (label) => {
		step = label;
	};
	try {
		page = await browser.newPage({ viewport: { width, height }, hasTouch: width < 600 });
		page.setDefaultTimeout(STEP_TIMEOUT);
		const errors = [];
		page.on('pageerror', (error) => errors.push(error.message));
		page.on('console', (message) => {
			if (message.type() === 'error') errors.push(`console: ${message.text()}`);
		});
		page.on('response', (response) => {
			// 504 "Outdated Optimize Dep" means the dev server's dependency cache went
			// stale. Name it here; downstream it only looks like a timeout.
			if (response.status() === 504) errors.push(`504 (stale dev-server dep): ${response.url()}`);
		});

		at('load /bookshelf/ and build the shelf');
		await page.goto(`${base}/bookshelf/`);
		await page.waitForSelector('#library-loading[hidden]', { state: 'attached' });

		at('shelve every book without overflowing sideways');
		assert.equal(await page.locator('.book-target').count(), books.length);
		assert.equal(
			await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),
			false,
		);
		assert.equal(
			await page.evaluate(() => document.documentElement.scrollHeight > innerHeight + 80),
			true,
			'The oak case continues below the first viewport',
		);
		await page.screenshot({ path: `${OUTPUT}/${tag}-shelf.png` });

		at('scroll to the last shelf');
		await page.getByRole('button', { name: /Open Marche ou crève/ }).scrollIntoViewIfNeeded();
		await page.waitForTimeout(250);
		await page.screenshot({ path: `${OUTPUT}/${tag}-shelf-end.png` });
		await page.evaluate(() => scrollTo(0, 0));

		at('open a book from the keyboard');
		const book = page.getByRole('button', { name: /Open White Nights/ });
		await book.focus();
		await page.keyboard.press('Enter');
		await page.waitForSelector('#book-reader[data-phase="reading"]');
		await page.waitForFunction(() =>
			document.getElementById('page-status')?.textContent?.startsWith('Page 1'),
		);

		at('give the reader its own frame, with text that fits');
		const reader = page.frames().find((frame) => frame.url().includes('/bookshelf/reader'));
		assert.ok(reader, 'Reader has its own lifetime');
		assert.equal(
			await reader
				.locator('.page-inner')
				.evaluateAll((nodes) =>
					nodes.some((node) => node.clientHeight > 0 && node.scrollHeight > node.clientHeight + 2),
				),
			false,
			'Sample text fits the page',
		);
		await page.screenshot({ path: `${OUTPUT}/${tag}-open.png` });

		at('turn a page forward, then back');
		const status = () => page.locator('#page-status').textContent();
		await page.getByRole('button', { name: 'Next page', exact: true }).click();
		await page.waitForFunction(
			() => !document.getElementById('page-status')?.textContent?.startsWith('Page 1'),
		);
		const turned = await status();
		await page.getByRole('button', { name: 'Previous page', exact: true }).click();
		await page.waitForFunction(
			(was) => document.getElementById('page-status')?.textContent !== was,
			turned,
		);

		at('close with Escape, releasing both renderers');
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

		at('interrupt an extraction without letting it reopen');
		await book.click();
		await page.getByRole('button', { name: /Return to shelf/ }).click();
		await page.waitForSelector('#book-reader', { state: 'hidden' });
		await page.waitForTimeout(650);
		assert.equal(
			await page.locator('#book-reader').evaluate((node) => node.open),
			false,
			'A stale callback reopened the reader',
		);

		at('open and close under reduced motion');
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await book.click();
		await page.waitForSelector('#book-reader[data-phase="reading"]');
		await page.keyboard.press('Escape');
		await page.waitForSelector('#book-reader', { state: 'hidden' });

		at('finish without console or page errors');
		assert.deepEqual(errors, []);
		console.log(`  ✓ ${name}`);
	} catch (error) {
		await page?.screenshot({ path: `${OUTPUT}/FAIL-${tag}.png` }).catch(() => {});
		error.message = `${name} — failed at: ${step}\n${error.message}\n  see ${OUTPUT}/FAIL-${tag}.png`;
		throw error;
	} finally {
		// WebKit's close() can take longer than the whole check did. The assertions
		// are already done, and Playwright reaps its browsers when the process
		// exits, so a slow shutdown must not fail an otherwise passing case.
		await withTimeout(browser.close(), 10_000, `${name} close`).catch(() => {});
	}
}

/** The archive has to work with JavaScript switched off. */
async function checkWithoutScript(base) {
	const browser = await chromium.launch();
	try {
		const page = await browser.newPage({ javaScriptEnabled: false });
		await page.goto(`${base}/bookshelf/`);
		await page.getByRole('link', { name: 'Browse the reading archive', exact: true }).click();
		assert.match(page.url(), /\/bookshelf\/archive\/?$/);
		assert.equal(
			await page.getByRole('heading', { name: 'Reading archive', exact: true }).count(),
			1,
		);
		console.log('  ✓ no-JavaScript archive path');
	} catch (error) {
		error.message = `no-JavaScript archive path — ${error.message}`;
		throw error;
	} finally {
		await browser.close();
	}
}

/** Run tasks `limit` at a time, collecting every failure instead of aborting. */
async function pool(tasks, limit) {
	const failures = [];
	let next = 0;
	await Promise.all(
		Array.from({ length: Math.min(limit, tasks.length) }, async () => {
			while (next < tasks.length) {
				try {
					await tasks[next++]();
				} catch (error) {
					failures.push(error);
				}
			}
		}),
	);
	return failures;
}

const filters = process.argv.slice(2);
const selected = CASES.filter(
	(item) =>
		filters.length === 0 ||
		filters.every((filter) => item.engine.name() === filter || String(item.width) === filter),
);
assert.ok(selected.length > 0, `No cases match: ${filters.join(' ')}`);

await mkdir(OUTPUT, { recursive: true });
const { base, stop } = await serve();
console.log(`${selected.length} case(s), ${CONCURRENCY} at a time\n`);
const started = Date.now();
let failures;
try {
	failures = await pool(
		[
			...selected.map(
				(item) => () =>
					withTimeout(
						check(item, base),
						CASE_TIMEOUT,
						`${item.engine.name()} ${item.width}×${item.height}`,
					),
			),
			() => withTimeout(checkWithoutScript(base), CASE_TIMEOUT, 'no-JavaScript archive path'),
		],
		CONCURRENCY,
	);
} finally {
	stop();
}

console.log(`\n${((Date.now() - started) / 1000).toFixed(0)}s · screenshots in ${OUTPUT}`);
if (failures.length > 0) {
	for (const failure of failures) console.error(`\n✗ ${failure.message}`);
	process.exit(1);
}
console.log('All checks passed.');
