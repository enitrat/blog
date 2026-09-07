/**
 * Browser checks for the living-room hero on /.
 *
 *   bun run room:check              every engine and viewport
 *   bun run room:check chromium     one engine
 *   bun run room:check --shots      also write frames to /tmp/room-check
 *
 * Starts its own dev server unless ROOM_URL points at one already.
 *
 * Boot is read off [data-room][data-live], which the scene publishes for the
 * stylesheet — the same signal shoot-poster.mjs waits on. It is the
 * contract this file asserts against: the budgets it reports are the whole
 * point of the baked architecture, so they are checked rather than remembered.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, rm } from 'node:fs/promises';
import { chromium, webkit } from 'playwright';

const OUTPUT = '/tmp/room-check';
const SHOTS = process.argv.includes('--shots');
const ENGINES = { chromium, webkit };
const only = process.argv.slice(2).find((arg) => arg in ENGINES);

const VIEWPORTS = [
	{ name: 'desktop', width: 1440, height: 900 },
	{ name: 'laptop', width: 1024, height: 768 },
	// hasTouch drives `pointer: coarse`, which is the gate that keeps phones on
	// the poster. Without it this case silently tests the desktop path.
	{ name: 'phone', width: 390, height: 844, hasTouch: true, isMobile: true },
];

async function serve() {
	if (process.env.ROOM_URL) return { url: process.env.ROOM_URL, stop: () => {} };
	const port = 4340 + Math.floor(Math.random() * 40);
	const child = spawn('npx', ['astro', 'dev', '--port', String(port)], { stdio: 'ignore' });
	const url = `http://localhost:${port}/`;
	for (let i = 0; i < 60; i++) {
		try {
			if ((await fetch(url)).ok) return { url, stop: () => child.kill() };
		} catch {}
		await new Promise((resolve) => setTimeout(resolve, 500));
	}
	child.kill();
	throw new Error('dev server never came up');
}

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

			await check(`${label} hero present`, async () =>
				assert.equal(await page.locator('.living-room').count(), 1),
			);

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

			/* The gate is a pure function of these four inputs, so its decision can
			   be read straight off the page rather than waited out. If this ever
			   drifts from the gate in LivingRoom.astro the mismatch fails loudly
			   below — a booting page would miss its poster assertion — not
			   silently. */
			const gate = await page.evaluate(() => ({
				browsing: location.hash.startsWith('#bookshelf'),
				coarse: matchMedia('(pointer: coarse)').matches,
				memory: navigator.deviceMemory ?? 8,
				saveData: navigator.connection?.saveData === true,
				webgl2: (() => {
					try {
						return !!document.createElement('canvas').getContext('webgl2');
					} catch {
						return false;
					}
				})(),
			}));
			const gateOpens =
				gate.webgl2 && (gate.browsing || (!gate.coarse && gate.memory >= 4 && !gate.saveData));

			const booted =
				gateOpens &&
				(await page
					.waitForSelector('[data-room][data-live]', { timeout: 20000 })
					.then(() => true)
					.catch(() => false));

			if (viewport.hasTouch) {
				await check(`${label} stays on the poster`, () =>
					assert.equal(gateOpens, false, `gate opened with ${JSON.stringify(gate)}`),
				);
				await check(`${label} coarse pointer is the reason`, () => assert.equal(gate.coarse, true));
				await check(`${label} never fetches three or the atlas`, () => {
					const heavy = requested.filter((u) => /three|scene|room\.(webp|bin)/i.test(u));
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

			if (!booted) {
				// A phone or a WebGL-less engine is allowed to stay on the poster, but
				// then the poster and the real controls have to carry the hero alone.
				await check(`${label} falls back to poster`, async () =>
					assert.equal(await page.locator('.living-room__poster').count(), 1),
				);
			} else {
				// The 3D moves the controls onto their objects; if projection breaks
				// they pile up in one corner and both point at the same thing.
				await check(`${label} hotspots land apart`, async () => {
					const boxes = await page
						.locator('.living-room__hotspot')
						.evaluateAll((els) => els.map((el) => el.getBoundingClientRect()));
					const gap = Math.hypot(boxes[0].x - boxes[1].x, boxes[0].y - boxes[1].y);
					assert.ok(gap > 60, `controls are ${Math.round(gap)}px apart`);
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

const { url, stop } = await serve();
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
