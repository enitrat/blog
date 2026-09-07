/**
 * Browser checks for the record player on /.
 *
 *   bun run record:check
 *
 * Starts its own dev server unless ROOM_URL points at one already. Hits the
 * real SoundCloud widget, so it needs a network.
 *
 * The interesting assertion is the one that branches: Chromium honours the
 * gesture delegated through the iframe's `allow="autoplay"` and plays; WebKit
 * fires PLAY and then pauses at position 0, which is Safari declining. Both
 * are allowed outcomes — what is not allowed is the button lying about which
 * one happened, so each branch checks the state the page is left in.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { chromium, webkit } from 'playwright';

const ENGINES = { chromium, webkit };

async function serve() {
	if (process.env.ROOM_URL) return { url: process.env.ROOM_URL, stop: () => {} };
	const port = 4390;
	const child = spawn('npx', ['astro', 'dev', '--port', String(port)], {
		stdio: 'ignore',
	});
	const url = `http://localhost:${port}/`;
	for (let i = 0; i < 60; i++) {
		try {
			if ((await fetch(url)).ok) return { url, stop: () => child.kill() };
		} catch {}
		await new Promise((r) => setTimeout(r, 500));
	}
	child.kill();
	throw new Error('no dev server');
}

const isPaused = (page) =>
	page.evaluate(
		() =>
			new Promise((resolve) => {
				const w = window.SC?.Widget(document.querySelector('.living-room__widget'));
				if (!w) return resolve('no widget');
				const timer = setTimeout(() => resolve('timeout'), 10000);
				w.getPosition((pos) =>
					w.isPaused((paused) => {
						clearTimeout(timer);
						resolve(`${paused ? 'paused' : 'playing'} at ${Math.round(pos)}ms`);
					}),
				);
			}),
	);

const { url, stop } = await serve();

for (const [name, engine] of Object.entries(ENGINES)) {
	const browser = await engine.launch();
	const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
	const requested = [];
	page.on('request', (r) => requested.push(r.url()));
	const results = [];
	const t = async (label, fn) => {
		try {
			await fn();
			results.push(`  ok   ${label}`);
		} catch (e) {
			results.push(`  FAIL ${label}: ${e.message}`);
		}
	};
	try {
		await page.goto(url, { waitUntil: 'load' });
		await page.waitForTimeout(2000);

		await t('nothing soundcloud before click', () => {
			assert.deepEqual(
				requested.filter((u) => /soundcloud/i.test(u)),
				[],
			);
		});
		await t('mix link exists and is hidden before play', async () => {
			const link = page.locator('.living-room__mix');
			assert.equal(await link.count(), 1);
			assert.equal(await link.getAttribute('href'), 'https://soundcloud.com/user2211512');
			assert.equal(await link.evaluate((el) => getComputedStyle(el).visibility), 'hidden');
		});

		const button = page.locator('button.living-room__hotspot');
		const mix = page.locator('.living-room__mix');
		await button.click();

		await t('crackle starts immediately', async () => {
			assert.equal(await button.getAttribute('aria-pressed'), 'true');
		});

		await t('widget reaches PLAY', async () => {
			// data-mix is set from the widget's own PLAY event, never from the click.
			await page.waitForFunction(
				() => document.querySelector('[data-room]')?.hasAttribute('data-mix'),
				{ timeout: 30000 },
			);
		});

		await t('soundcloud only requested after the click', () => {
			assert.ok(requested.filter((u) => /soundcloud/i.test(u)).length > 0);
		});

		await t('button label flipped', async () => {
			assert.equal(await button.textContent(), 'Stop the record');
		});

		await t('mix link is visible on PLAY', async () => {
			await page.waitForTimeout(600);
			assert.equal(await mix.evaluate((el) => getComputedStyle(el).visibility), 'visible');
			assert.equal(await mix.evaluate((el) => getComputedStyle(el).opacity), '1');
		});

		await t('mix link clears the stop button', async () => {
			const a = await mix.boundingBox();
			const b = await button.boundingBox();
			const overlap =
				a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
			assert.equal(overlap, false, `mix ${JSON.stringify(a)} vs button ${JSON.stringify(b)}`);
		});

		await t('button stays visible while playing', async () => {
			assert.equal(await button.evaluate((el) => getComputedStyle(el).opacity), '1');
		});

		// Five seconds in: long enough for a browser that is going to refuse to
		// have refused, and for one that is playing to be seconds into the track.
		await page.waitForTimeout(5000);
		const state = await isPaused(page);
		console.log(`\n${name}: widget five seconds in — ${state}`);

		if (state.startsWith('playing')) {
			await t('the widget really plays', () =>
				assert.ok(Number.parseInt(state.match(/at (\d+)/)[1], 10) > 1000, state),
			);
			await button.click();
			await page.waitForTimeout(2000);
			await t('stopping pauses the widget', async () =>
				assert.match(await isPaused(page), /^paused/),
			);
			await t('stopping resets the button', async () => {
				assert.equal(await button.getAttribute('aria-pressed'), 'false');
				assert.equal(await button.textContent(), 'Play the record');
			});
			await t('mix link hides again', async () => {
				assert.equal(await mix.evaluate((el) => getComputedStyle(el).visibility), 'hidden');
			});
		} else {
			// The browser refused. The record still has to play, truthfully.
			await t('refusal leaves the crackle running', async () => {
				assert.equal(await button.getAttribute('aria-pressed'), 'true');
				assert.equal(await button.textContent(), 'Stop the record');
			});
			await t('refusal hides the mix link', async () => {
				assert.equal(await mix.evaluate((el) => getComputedStyle(el).visibility), 'hidden');
			});
			await button.click();
			await page.waitForTimeout(500);
			await t('the button still stops', async () => {
				assert.equal(await button.getAttribute('aria-pressed'), 'false');
				assert.equal(await button.textContent(), 'Play the record');
			});
		}
	} finally {
		await browser.close();
	}
	console.log(results.join('\n'));
}

stop();
process.exit(0);
