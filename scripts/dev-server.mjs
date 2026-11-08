/** One Next dev server for the browser checks, or the caller's own via DEV_URL. */
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';

const freePort = () =>
	new Promise((resolve, reject) => {
		const probe = createServer().listen(0, '127.0.0.1', () => {
			const { port } = probe.address();
			probe.close(() => resolve(port));
		});
		probe.on('error', reject);
	});

/** Resolves to `{ base, stop }`; `base` has no trailing slash. */
export async function serve() {
	if (process.env.DEV_URL) return { base: process.env.DEV_URL.replace(/\/$/, ''), stop: () => {} };
	const port = await freePort();
	const base = `http://127.0.0.1:${port}`;
	// Media under public/_assets and its manifest must exist before the first request.
	await new Promise((resolve, reject) =>
		spawn(process.execPath, ['scripts/sync-assets.mjs'], { stdio: 'ignore' })
			.on('exit', (code) => (code === 0 ? resolve() : reject(new Error('sync-assets failed'))))
			.on('error', reject),
	);
	// Each server writes its own dist dir, so two checks never share a dev cache.
	const child = spawn(
		process.execPath,
		['node_modules/next/dist/bin/next', 'dev', '--hostname', '127.0.0.1', '--port', String(port)],
		{ stdio: 'ignore', env: { ...process.env, NEXT_DIST_DIR: `.next-dev-${port}` } },
	);
	for (let attempt = 0; attempt < 240; attempt++) {
		try {
			if ((await fetch(base, { signal: AbortSignal.timeout(5000) })).ok)
				return { base, stop: () => child.kill() };
		} catch {}
		await new Promise((resolve) => setTimeout(resolve, 500));
	}
	child.kill();
	throw new Error('The dev server did not come up within 120s.');
}
