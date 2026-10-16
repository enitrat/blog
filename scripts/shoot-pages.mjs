/**
 * Full-page screenshots of the main routes at every breakpoint, for looking at a visual change on
 * every consumer, not only the tile that prompted it.
 *
 *   bun run shots                      every route, chromium
 *   bun run shots webkit / /writing    one engine, some routes
 *
 * Starts its own dev server (fresh Vite cache, so no stale CSS), hides the Astro dev toolbar, and
 * stubs WebGL so the room shows its poster (software WebGL makes headless pages crawl; `room:check`
 * covers the room). Writes `<engine>-<route>-<width>-<scheme>.png` to SHOTS_OUT (default /tmp/shots).
 */
import { mkdir } from 'node:fs/promises';
import { chromium, webkit } from 'playwright';
import { serve } from './dev-server.mjs';

const ENGINES = { chromium, webkit };
const ROUTES = ['/', '/writing/', '/reading/', '/work/', '/system/'];
const VIEWPORTS = [
	{ width: 1440, height: 900 },
	{ width: 1024, height: 768 },
	{ width: 390, height: 844, isMobile: true, hasTouch: true },
];
const OUT = process.env.SHOTS_OUT ?? '/tmp/shots';

const args = process.argv.slice(2);
const engine = args.find((a) => a in ENGINES) ?? 'chromium';
const routes = args.filter((a) => a.startsWith('/'));

function stub() {
	const getContext = HTMLCanvasElement.prototype.getContext;
	HTMLCanvasElement.prototype.getContext = function (type, ...a) {
		return /webgl/i.test(type) ? null : getContext.call(this, type, ...a);
	};
	addEventListener('DOMContentLoaded', () => {
		const style = document.createElement('style');
		style.textContent = 'astro-dev-toolbar{display:none!important}';
		document.head.append(style);
	});
}

await mkdir(OUT, { recursive: true });
const server = await serve();
const browser = await ENGINES[engine].launch();
try {
	for (const route of routes.length ? routes : ROUTES) {
		for (const { width, height, ...device } of VIEWPORTS) {
			for (const colorScheme of ['light', 'dark']) {
				const page = await browser.newPage({ viewport: { width, height }, colorScheme, ...device });
				await page.addInitScript(stub);
				await page.goto(server.base + route, { waitUntil: 'networkidle' });
				// past the bench settle entrance
				await page.waitForTimeout(800);
				const name = route.replace(/\//g, '') || 'home';
				const path = `${OUT}/${engine}-${name}-${width}-${colorScheme}.png`;
				await page.screenshot({ path, fullPage: true });
				console.log(path);
				await page.close();
			}
		}
	}
} finally {
	await browser.close();
	server.stop();
}
