/**
 * Serves a static build folder (default `dist/`) on 127.0.0.1, as Vercel would: `/x/` is
 * `x/index.html`, a miss is `404.html`. `bun scripts/serve-dist.mjs [dir] [port]`.
 * Imported by the browser checks as `serveStatic(dir)`.
 */
import { readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';

const TYPES = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript',
	'.css': 'text/css',
	'.json': 'application/json',
	'.svg': 'image/svg+xml',
	'.png': 'image/png',
	'.jpg': 'image/jpeg',
	'.jpeg': 'image/jpeg',
	'.webp': 'image/webp',
	'.avif': 'image/avif',
	'.woff2': 'font/woff2',
	'.glb': 'model/gltf-binary',
	'.txt': 'text/plain; charset=utf-8',
	'.xml': 'application/xml',
};

async function resolve(dir, url) {
	const path = normalize(decodeURIComponent(new URL(url, 'http://x').pathname)).replace(
		/^(\.\.[/\\])+/,
		'',
	);
	for (const candidate of [path, join(path, 'index.html'), `${path}.html`]) {
		const file = join(dir, candidate);
		try {
			if ((await stat(file)).isFile()) return { file, status: 200 };
		} catch {}
	}
	return { file: join(dir, '404.html'), status: 404 };
}

/** Resolves to `{ base, stop }`; `base` has no trailing slash. */
export async function serveStatic(dir = 'dist', port = 0) {
	const server = createServer(async (req, res) => {
		const { file, status } = await resolve(dir, req.url ?? '/');
		try {
			const body = await readFile(file);
			res.writeHead(status, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
			res.end(body);
		} catch {
			res.writeHead(404).end();
		}
	});
	await new Promise((r) => server.listen(port, '127.0.0.1', r));
	return { base: `http://127.0.0.1:${server.address().port}`, stop: () => server.close() };
}

if (import.meta.main ?? process.argv[1]?.endsWith('serve-dist.mjs')) {
	const { base } = await serveStatic(process.argv[2] ?? 'dist', Number(process.argv[3] ?? 4322));
	console.log(`serving ${process.argv[2] ?? 'dist'} at ${base}`);
}
