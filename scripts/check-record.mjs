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
const url = `${base}${process.env.ROOM_PATH ?? '/'}`;
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
		// The scene runs inside the tile on a desktop; wait for it to be live.
		await page.waitForSelector('[data-room][data-live]', { timeout: 30000 });

		await t('nothing soundcloud before click', () => {
			assert.deepEqual(
				requested.filter((u) => /soundcloud/i.test(u)),
				[],
			);
		});
		// The plain-link fallback names the set; with the room live it steps aside.
		await t('there is no visible SoundCloud link on the tile', async () => {
			const links = page.locator('a[href*="soundcloud.com"]');
			for (const link of await links.all()) assert.equal(await link.isVisible(), false);
		});

		// The record player in the scene is the only way to play. Click the scene
		// itself where the record player stands (a canvas click is a real pick,
		// not the hotspot button laid over it): the set starts and the room opens
		// at the record view.
		const button = page.locator('button.living-room__hotspot');
		const spot = await button.boundingBox();
		await page
			.locator('.living-room__canvas')
			.evaluate(
				(canvas, [x, y]) =>
					canvas.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: x, clientY: y })),
				[spot.x + spot.width / 2, spot.y + spot.height / 2],
			);

		await t('the record click opens the room at the record view', async () => {
			await page.locator('[data-room-dialog][open]').waitFor({ timeout: 5000 });
			await page.waitForFunction(
				() => document.querySelector('[data-room]').dataset.view === 'record',
				null,
				{
					timeout: 8000,
				},
			);
		});
		await t('there is no play button on the tile', async () => {
			assert.equal(await page.locator('[data-tile-record], .room-tile__chip').count(), 0);
		});

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

		await t('the opened room shows what is playing', async () => {
			assert.ok(await page.locator('.room-bar__playing').isVisible());
		});
		await t('the tile header shows what is playing, quietly', async () => {
			await page.locator('[data-room-close]').click();
			await page.waitForFunction(() => !document.querySelector('[data-room-dialog]').open);
			assert.ok(await page.locator('.room-tile__playing').isVisible());
			assert.equal(await page.locator('.room-tile').getAttribute('data-playing'), '');
			// Back in the tile, the record view is left behind.
			await page.waitForFunction(
				() => document.querySelector('[data-room]').dataset.view === 'room',
			);
		});

		// The record player is also what stops the set: back in the tile, a click
		// on it plays or stops and opens the room at the record view.
		const stopRecord = async () => {
			if (!(await page.locator('[data-room-dialog]').evaluate((el) => el.open))) {
				const box = await button.boundingBox();
				await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
				await page.locator('[data-room-dialog][open]').waitFor({ timeout: 5000 });
			}
		};

		// Five seconds in: long enough for a browser that is going to refuse to
		// have refused, and for one that is playing to be seconds into the track.
		await page.waitForTimeout(5000);
		const state = await isPaused(page);
		console.log(`\n${name}: widget five seconds in — ${state}`);

		if (state.startsWith('playing')) {
			await t('the widget really plays', () =>
				assert.ok(Number.parseInt(state.match(/at (\d+)/)[1], 10) > 1000, state),
			);
			await stopRecord();
			// The scene runs inside the tile on a desktop; wait for it to be live.
			await page.waitForSelector('[data-room][data-live]', { timeout: 30000 });
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
			await stopRecord();
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
