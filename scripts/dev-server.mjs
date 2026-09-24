/** One Astro dev server for the browser checks, or the caller's own via DEV_URL. */
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
	// A stale Vite dep cache serves 504s that surface as unrelated selector
	// timeouts, so every server starts from a cleared one.
	const child = spawn(
		process.execPath,
		[
			'node_modules/astro/astro.js',
			'dev',
			'--host',
			'127.0.0.1',
			'--port',
			String(port),
			'--force',
		],
		{ stdio: 'ignore' },
	);
	for (let attempt = 0; attempt < 120; attempt++) {
		try {
			if ((await fetch(base, { signal: AbortSignal.timeout(2000) })).ok)
				return { base, stop: () => child.kill() };
		} catch {}
		await new Promise((resolve) => setTimeout(resolve, 500));
	}
	child.kill();
	throw new Error('The dev server did not come up within 60s.');
}
