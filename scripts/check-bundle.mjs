/**
 * Fail the build if three.js reaches a page's eager module graph.
 *
 * The hero loads three behind a dynamic import so none of its ~170KB gzipped
 * sits on the critical path. That guarantee is one stray top-level `import`
 * away from silently disappearing, and the only symptom is a slower homepage.
 * So it gets asserted rather than remembered.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, posix, relative, resolve, sep } from 'node:path';

const DIST = 'dist';

/**
 * Pages allowed to pull three in eagerly. /bookshelf/ IS a 3D bookcase — there
 * is no poster to fall back to, so deferring three there would only add a round
 * trip. The homepage is the opposite: its hero has a real poster as its LCP
 * element, and three must stay behind the dynamic import. Hence an explicit
 * per-page allowlist rather than a global off switch.
 */
const EAGER_THREE_OK = new Set(['bookshelf/index.html']);

/**
 * Content markers, not filenames: Vite is free to name the chunk carrying three
 * anything at all (it currently lands in `pleiade-paint.*.js`). The license
 * banner survives minification; the renderer's own error prefix is the backup
 * in case a future build strips legal comments.
 */
const THREE_MARKERS = [/Three\.js Authors/, /THREE\.WebGLRenderer/];

/**
 * Every static form Rollup emits: `import"a"`, `import x from"a"`,
 * `import{a}from"a"`, `import*as x from"a"`, `export*from"a"`, `export{a}from"a"`.
 * A dynamic `import("a")` must NOT match — that is the whole point of the check —
 * so the clause between the keyword and `from` may not contain parentheses.
 */
const STATIC_IMPORT =
	/(?:^|[^\w$.])(?:import|export)(?:\s*[^'"()\n;]*?\bfrom)?\s*["']([^"']+)["']/g;

function htmlFiles(dir) {
	return readdirSync(dir).flatMap((name) => {
		const path = join(dir, name);
		if (statSync(path).isDirectory()) return htmlFiles(path);
		return name.endsWith('.html') ? [path] : [];
	});
}

/** Resolve a specifier against the importing file's directory, dist-relative. */
function resolveSpecifier(specifier, fromDir) {
	if (!specifier.startsWith('.') && !specifier.startsWith('/')) return null; // bare or http
	const absolute = specifier.startsWith('/')
		? join(DIST, specifier.slice(1))
		: resolve(fromDir, specifier);
	const relativePath = relative(DIST, absolute);
	return relativePath.startsWith('..') ? null : relativePath.split(sep).join(posix.sep);
}

function read(path) {
	try {
		return readFileSync(join(DIST, path), 'utf8');
	} catch {
		return null;
	}
}

/**
 * Walk the eager graph from one entry. Returns the import chain that reaches
 * three, or null. Chains make a failure debuggable: the offending chunk is
 * usually several hops from the file anyone would think to look at.
 */
function chainToThree(entry, source) {
	const seen = new Set([entry]);
	const queue = [[entry, source, [entry]]];
	while (queue.length) {
		const [path, code, chain] = queue.shift();
		if (code === null) continue;
		if (THREE_MARKERS.some((marker) => marker.test(code))) return chain;
		for (const match of code.matchAll(STATIC_IMPORT)) {
			const next = resolveSpecifier(match[1], dirname(join(DIST, path)));
			if (!next || seen.has(next)) continue;
			seen.add(next);
			queue.push([next, read(next), [...chain, next]]);
		}
	}
	return null;
}

const pages = htmlFiles(DIST);
const offenders = [];
for (const file of pages) {
	const page = relative(DIST, file).split(sep).join(posix.sep);
	const html = readFileSync(file, 'utf8');
	const pageDir = dirname(file);

	const entries = [
		...[...html.matchAll(/<script[^>]+src="([^"]+)"/g)].map((m) => {
			const path = resolveSpecifier(m[1], pageDir);
			return path && [path, read(path)];
		}),
		// Inline module scripts can static-import chunks just as well.
		...[...html.matchAll(/<script[^>]*type="module"[^>]*>([\s\S]*?)<\/script>/g)].map((m) => [
			`${page} (inline script)`,
			m[1],
		]),
	].filter(Boolean);

	for (const [entry, source] of entries) {
		const chain = chainToThree(entry, source);
		if (!chain) continue;
		if (EAGER_THREE_OK.has(page)) continue;
		offenders.push(`${page}\n      entry: ${entry}\n      chain: ${chain.join('\n          -> ')}`);
	}
}

if (offenders.length) {
	console.error('three.js is eagerly loaded — it must stay behind a dynamic import:');
	for (const offender of offenders) console.error(`  - ${offender}`);
	console.error(`\n  Allowed to load three eagerly: ${[...EAGER_THREE_OK].join(', ')}`);
	process.exit(1);
}
console.log(
	`bundle check passed (${pages.length} pages, three stays lazy outside ${[...EAGER_THREE_OK].join(', ')})`,
);
