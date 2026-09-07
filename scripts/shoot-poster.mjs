/**
 * Render the living room once, headless, and save it as the hero poster.
 *
 *   bun run room:poster
 *
 * The poster is the homepage's LCP element — a <canvas> is never an LCP
 * candidate, so this PNG is what the metric actually measures, and it has to be
 * the scene's own frame rather than an approximation of it. Which means it can
 * only be produced from the scene. Re-run it whenever the room is re-baked.
 *
 * It also has to be right for the devices that never load the 3D at all: on a
 * phone this image *is* the hero, with the two controls annotated over it.
 *
 * Starts its own dev server unless ROOM_URL points at one already.
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import sharp from 'sharp';

const OUT = 'src/assets/living-room.png';
const WIDTH = 1920;
const HEIGHT = 1080;

async function serve() {
	if (process.env.ROOM_URL) return { url: process.env.ROOM_URL, stop: () => {} };
	const port = 4390;
	const child = spawn(
		process.execPath,
		['node_modules/astro/astro.js', 'dev', '--port', String(port)],
		{ stdio: 'ignore' },
	);
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

const { url, stop } = await serve();
const browser = await chromium.launch({ args: ['--use-gl=angle', '--enable-gpu'] });
try {
	/* Tall enough that the hero is not below the fold, and at 2x so the 60rem
	   column comes back at the poster's full 1920 without upscaling. */
	const page = await browser.newPage({
		viewport: { width: 1280, height: 1400 },
		deviceScaleFactor: 2,
	});
	page.on('pageerror', (error) => console.error('page error:', error.message));
	await page.goto(url, { waitUntil: 'load' });
	await page.waitForSelector('[data-room][data-live]', { timeout: 30000 });
	// The dev toolbar floats over the bottom of the viewport and lands in the
	// clip; the poster has to be the room and nothing else.
	await page.addStyleTag({
		content:
			'astro-dev-toolbar, .living-room__hotspots, .living-room__caption { visibility: hidden !important; }',
	});
	// One more frame after the fade, so the capture is the canvas and not a
	// half-opaque canvas over the previous poster.
	await page.waitForTimeout(600);

	// The locator's own screenshot, not a viewport clip: `clip` is measured from
	// the viewport, and the hero sits below the fold on anything but a very tall
	// window — which is how the first poster came out as a corner of the wall.
	const shot = await page.locator('.living-room__canvas').screenshot();
	await sharp(shot)
		.resize(WIDTH, HEIGHT, { fit: 'cover' })
		.png({ compressionLevel: 9 })
		.toFile(OUT);
	console.log(`poster written to ${OUT} (${WIDTH}x${HEIGHT})`);
} finally {
	await browser.close();
	stop();
}
