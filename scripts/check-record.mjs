/**
 * Browser checks for the record player on /.
 *
 *   bun run record:check
 *
 * Starts its own dev server unless DEV_URL points at one already. Hits the
 * real SoundCloud widget, so it needs a network.
 *
 * The interesting assertion is the one that branches: Chromium honours the
 * gesture delegated through the iframe's `allow="autoplay"` and plays; WebKit
 * fires PLAY and then pauses at position 0, which is Safari declining. Both
 * are allowed outcomes — what is not allowed is the button lying about which
 * one happened, so each branch checks the state the page is left in.
 */
import assert from 'node:assert/strict';
import { chromium, webkit } from 'playwright';
import { serve } from './dev-server.mjs';

const ENGINES = { chromium, webkit };

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

const { base, stop } = await serve();
const url = `${base}/`;
let failed = false;

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
			failed = true;
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
		await t('there is no SoundCloud link on the table', async () => {
			assert.equal(await page.locator('a[href*="soundcloud.com"]').count(), 0);
		});

		const button = page.locator('button.living-room__hotspot');
		await button.click();

		await t('crackle starts immediately', async () => {
			assert.equal(await button.getAttribute('aria-pressed'), 'true');
		});

		await t('widget loads Bouncyx Seven', async () => {
			const frame = page.locator('.living-room__widget');
			await frame.waitFor({ timeout: 30000 });
			assert.match(await frame.getAttribute('src'), /user2211512\/bouncyx-seven/);
		});

		await t('soundcloud only requested after the click', () => {
			assert.ok(requested.filter((u) => /soundcloud/i.test(u)).length > 0);
		});

		await t('button label flipped', async () => {
			assert.equal(await button.textContent(), 'Stop the record');
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
		} else {
			// The browser refused. The record still has to play, truthfully.
			await t('refusal leaves the crackle running', async () => {
				assert.equal(await button.getAttribute('aria-pressed'), 'true');
				assert.equal(await button.textContent(), 'Stop the record');
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
process.exit(failed ? 1 : 0);
