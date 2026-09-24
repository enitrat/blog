/**
 * Browser checks for the living-room hero on /.
 *
 *   bun run room:check              every engine and viewport
 *   bun run room:check chromium     one engine
 *   bun run room:check --shots      also write frames to /tmp/room-check
 *
 * Starts its own dev server unless DEV_URL points at one already.
 *
 * Boot is read off [data-room][data-live], which the scene publishes for the
 * stylesheet — the same signal shoot-poster.mjs waits on. It is the
 * contract this file asserts against: the budgets it reports are the whole
 * point of the baked architecture, so they are checked rather than remembered.
 */
import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { chromium, webkit } from 'playwright';
import sharp from 'sharp';
import { serve } from './dev-server.mjs';

const OUTPUT = '/tmp/room-check';
const SHOTS = process.argv.includes('--shots');
const METRICS = process.argv.includes('--metrics');
const ENGINES = { chromium, webkit };
const only = process.argv.slice(2).find((arg) => arg in ENGINES);

const VIEWPORTS = [
	{ name: 'desktop', width: 1440, height: 900 },
	{ name: 'laptop', width: 1024, height: 768 },
	// hasTouch drives `pointer: coarse`, which is the gate that keeps phones on
	// the poster. Without it this case silently tests the desktop path.
	{ name: 'phone', width: 390, height: 844, hasTouch: true, isMobile: true },
];

/** Every failure this run saw, so one bad engine does not hide the rest. */
const failures = [];
const check = async (name, fn) => {
	try {
		// Awaited: a rejecting page.evaluate() inside a case must be recorded as a
		// failure, not thrown out of run() with the browser still open.
		await fn();
	} catch (error) {
		failures.push(`${name}: ${error.message}`);
	}
};

async function run(engineName, url) {
	for (const viewport of VIEWPORTS) {
		const browser = await ENGINES[engineName].launch();
		const { name: _name, width, height, ...emulation } = viewport;
		const page = await browser.newPage({ viewport: { width, height }, ...emulation });
		const errors = [];
		const requested = [];
		page.on('request', (request) => requested.push(request.url()));
		page.on('pageerror', (error) => errors.push(error.message));
		page.on('console', (message) => {
			// three warns rather than throws on deprecated APIs, so those must fail
			// the run. The parallel-compile notice is not one: it reports a missing
			// driver extension and headless GL never has it.
			const text = message.text();
			if (text.includes('KHR_parallel_shader_compile')) return;
			// A third-party endpoint failing is not this page's problem — Vercel's
			// analytics script 403s against the dev server. Same-origin failures are
			// caught by the response listener below, which knows the URL.
			if (text.startsWith('Failed to load resource')) return;
			if (message.type() === 'error' || text.includes('THREE.')) errors.push(text);
		});
		page.on('response', (response) => {
			if (response.status() >= 400 && response.url().startsWith(url)) {
				errors.push(`${response.status()} ${response.url()}`);
			}
		});

		const label = `${engineName}/${viewport.name}`;
		try {
			await page.goto(url, { waitUntil: 'load' });

			// The hero must not be able to shift layout: it needs a reserved box
			// before the canvas ever boots.
			await check(`${label} reserves height`, async () => {
				const reserved = await page.locator('.living-room').evaluate((el) => el.clientHeight);
				assert.ok(reserved > 200, `got ${reserved}px`);
			});

			// A <canvas> is never an LCP candidate; the poster is what the metric
			// measures, so it has to be a real eager <img>.
			await check(`${label} poster is eager`, async () => {
				const poster = page.locator('.living-room__poster');
				assert.equal(await poster.count(), 1);
				assert.equal(await poster.getAttribute('loading'), 'eager');
				assert.ok((await poster.getAttribute('alt'))?.length > 20, 'poster needs real alt text');
			});

			// Both controls are real DOM, present and operable before any of the 3D
			// exists — on every device, including the ones that never load it.
			await check(`${label} controls are real`, async () => {
				assert.equal(await page.locator('a.living-room__hotspot[href="#room-library"]').count(), 1);
				assert.equal(await page.locator('button.living-room__hotspot').count(), 1);
				const reachable = await page.evaluate(() =>
					[...document.querySelectorAll('.living-room__hotspot')].every(
						(el) => el.tabIndex >= 0 && el.textContent.trim().length > 0,
					),
				);
				assert.ok(reachable, 'a hotspot is not keyboard reachable or has no label');
			});

			// A desktop viewport with WebGL2 must boot the room; a touch one must not.
			const webgl2 = await page.evaluate(() => {
				try {
					return !!document.createElement('canvas').getContext('webgl2');
				} catch {
					return false;
				}
			});
			const gateOpens = webgl2 && !viewport.hasTouch;

			const booted =
				gateOpens &&
				(await page
					.waitForSelector('[data-room][data-live]', { timeout: 20000 })
					.then(() => true)
					.catch(() => false));

			const metrics = booted
				? await page.evaluate(() => {
						const assets = performance
							.getEntriesByType('resource')
							.filter((entry) => entry.name.includes('.glb') && entry.initiatorType === 'fetch');
						return {
							liveMs: Math.round(performance.getEntriesByName('room-live').at(-1)?.startTime ?? 0),
							lastAssetMs: Math.round(Math.max(0, ...assets.map((entry) => entry.responseEnd))),
							assets: assets.length,
							encodedBytes: assets.reduce((total, entry) => total + entry.encodedBodySize, 0),
						};
					})
				: null;
			if (metrics) {
				await check(`${label} stays within the room asset budget`, () => {
					assert.ok(metrics.encodedBytes <= 11_000_000, `${metrics.encodedBytes} GLB bytes`);
				});
				if (METRICS) {
					console.log(`ROOM_METRICS ${label} ${JSON.stringify(metrics)}`);
				}
			}

			if (viewport.hasTouch) {
				await check(`${label} never fetches three or the GLBs`, () => {
					const heavy = requested.filter((u) => /three|scene|\.glb/i.test(u));
					assert.equal(heavy.length, 0, heavy.join(', '));
				});
			}

			// A desktop viewport with WebGL2 has no excuse: if the gate opened and the
			// room still did not mount, something threw inside it and the catch in
			// LivingRoom.astro swallowed it. That is exactly the failure this check
			// exists for, and letting it pass as "falls back to poster" is how a
			// misaligned mesh header shipped once already.
			if (gateOpens) {
				await check(`${label} mounts the room`, () =>
					assert.equal(booted, true, 'the gate opened but the room never went live'),
				);
			}

			if (booted) {
				// The 3D moves the controls onto their objects; if projection breaks
				// they pile up in one corner and both point at the same thing.
				await check(`${label} hotspots land apart`, async () => {
					const boxes = await page
						.locator('.living-room__hotspot')
						.evaluateAll((els) => els.map((el) => el.getBoundingClientRect()));
					const gap = Math.hypot(boxes[0].x - boxes[1].x, boxes[0].y - boxes[1].y);
					assert.ok(gap > 60, `controls are ${Math.round(gap)}px apart`);
				});

				// WebKit exposes depth fighting between near-coplanar manuscript surfaces.
				if (engineName === 'webkit' && viewport.name === 'desktop') {
					await check(`${label} manuscript surfaces stay stable`, async () => {
						const room = page.locator('[data-room]');
						await room.scrollIntoViewIfNeeded();
						const bounds = await room.boundingBox();
						assert.ok(bounds);
						let darkest = 0;
						for (let i = 0; i < 40; i++) {
							await page.mouse.move(
								bounds.x + bounds.width * (i % 2 ? 0.95 : 0.05),
								bounds.y + bounds.height * (i % 4 < 2 ? 0.05 : 0.95),
							);
							await page.waitForTimeout(16);
							const frame = await page.locator('.living-room__canvas').screenshot();
							const { data, info } = await sharp(frame)
								.extract({ left: 555, top: 325, width: 105, height: 85 })
								.removeAlpha()
								.raw()
								.toBuffer({ resolveWithObject: true });
							let dark = 0;
							for (let pixel = 0; pixel < data.length; pixel += info.channels) {
								if (data[pixel] < 35 && data[pixel + 1] < 35 && data[pixel + 2] < 35) dark++;
							}
							darkest = Math.max(darkest, dark);
						}
						assert.ok(darkest < 900, `${darkest} near-black pixels over the manuscripts`);
					});
				}

				await check(`${label} sofa leaves the bookshelf clear`, async () => {
					await page.locator('[data-anchor="bookshelf"]').click();
					await page.waitForFunction(
						() =>
							document.querySelector('[data-room]')?.getAttribute('data-view') === 'shelf' &&
							!document.querySelector('[data-room]')?.hasAttribute('data-traveling'),
					);
					try {
						const canvas = page.locator('.living-room__canvas');
						const canvasBounds = await canvas.boundingBox();
						const bottomRow = await page
							.locator('.room-library__row:not([hidden])')
							.last()
							.boundingBox();
						assert.ok(canvasBounds && bottomRow);
						const left = Math.max(0, Math.floor(bottomRow.x - canvasBounds.x));
						const top = Math.max(0, Math.floor(bottomRow.y - canvasBounds.y));
						const width = Math.min(canvasBounds.width - left, Math.ceil(bottomRow.width));
						const height = Math.min(canvasBounds.height - top, Math.ceil(bottomRow.height));
						const { data, info } = await sharp(await canvas.screenshot())
							.extract({ left, top, width, height })
							.removeAlpha()
							.raw()
							.toBuffer({ resolveWithObject: true });
						let oxblood = 0;
						for (let pixel = 0; pixel < data.length; pixel += info.channels) {
							const red = data[pixel];
							if (red > 25 && data[pixel + 1] / red < 0.25 && data[pixel + 2] / red < 0.4)
								oxblood++;
						}
						const covered = oxblood / (data.length / info.channels);
						assert.ok(covered < 0.2, `${Math.round(covered * 100)}% of the bottom row is sofa`);
					} finally {
						await page.locator('[data-shelf-exit]').click();
						await page.waitForFunction(
							() =>
								document.querySelector('[data-room]')?.getAttribute('data-view') === 'room' &&
								!document.querySelector('[data-room]')?.hasAttribute('data-traveling'),
						);
					}
				});

				if (SHOTS) {
					await check(`${label} shot`, async () => {
						await page.screenshot({
							path: `${OUTPUT}/${label.replace('/', '-')}.png`,
							clip: await page.locator('.living-room').boundingBox(),
						});
					});
				}
			}

			// Also on the poster path: a page error thrown by the gate script used to
			// slip through here, and every phone viewport takes that path.
			await check(`${label} console clean`, () => assert.deepEqual(errors, []));
		} finally {
			await browser.close();
		}
	}
}

const { base, stop } = await serve();
const url = `${base}/`;
try {
	if (SHOTS) {
		await rm(OUTPUT, { recursive: true, force: true });
		await mkdir(OUTPUT, { recursive: true });
	}
	for (const engine of only ? [only] : Object.keys(ENGINES)) await run(engine, url);
} finally {
	stop();
}

if (failures.length) {
	console.error(`\n${failures.length} failure(s):`);
	for (const failure of failures) console.error(`  - ${failure}`);
	process.exit(1);
}
console.log(`room checks passed${SHOTS ? ` (frames in ${OUTPUT})` : ''}`);
