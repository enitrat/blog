/** A Next server for the browser checks: `serve()` runs `next dev` (or the caller's own via DEV_URL), `serveBuild()` runs `next start` on the last build. */
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
	// Each server writes its own dist dir, so two checks never share a dev cache.
	return start('dev', port, { NEXT_DIST_DIR: `.next-dev-${port}` });
}

/** `next start` on the existing production build; resolves to `{ base, stop }`. */
export async function serveBuild() {
	return start('start', await freePort(), {});
}

async function start(command, port, env) {
	const base = `http://127.0.0.1:${port}`;
	const child = spawn(
		process.execPath,
		['node_modules/next/dist/bin/next', command, '--hostname', '127.0.0.1', '--port', String(port)],
		{ stdio: 'ignore', env: { ...process.env, ...env } },
	);
	for (let attempt = 0; attempt < 240; attempt++) {
		try {
			if ((await fetch(base, { signal: AbortSignal.timeout(5000) })).ok)
				return { base, stop: () => child.kill() };
		} catch {}
		await new Promise((resolve) => setTimeout(resolve, 500));
	}
	child.kill();
	throw new Error(`next ${command} did not come up within 120s.`);
}
