import 'server-only';
import { existsSync } from 'node:fs';
import type { StaticImageData } from 'next/image';

/** An image as `next/image` takes it, plus its source path for build-time reads (`glowOf`). */
export type Img = StaticImageData & { file: string };

type Load = (path: string) => Promise<{ default: StaticImageData }>;

// The bundler bundles every file a template can match, so each extension is its own literal.
// A template that matches no file fails the build: add a line here when a new format is checked in.
const ASSETS: Record<string, Load> = {
	png: (p) => import(`../assets/${p}.png`),
	jpg: (p) => import(`../assets/${p}.jpg`),
};
const CONTENT: Record<string, Load> = {
	png: (p) => import(`../content/${p}.png`),
	jpg: (p) => import(`../content/${p}.jpg`),
};

/**
 * An image by its path under `src/assets/` (`covers/<isbn>.jpg`), or under `src/content/` with a
 * `content/` prefix; undefined when no such file is checked in.
 */
export async function maybeImage(key: string): Promise<Img | undefined> {
	const inContent = key.startsWith('content/');
	const rel = inContent ? key.slice('content/'.length) : key;
	const file = `src/${inContent ? 'content' : 'assets'}/${rel}`;
	const m = /^(.+)\.(\w+)$/.exec(rel);
	if (!m || !existsSync(file)) return undefined;
	const load = (inContent ? CONTENT : ASSETS)[m[2]];
	if (!load) throw new Error(`${file}: no loader for .${m[2]} in src/lib/media.ts.`);
	return { ...(await load(m[1])).default, file };
}

/** Like `maybeImage`, but the file must exist. */
export async function image(key: string): Promise<Img> {
	const img = await maybeImage(key);
	if (!img) throw new Error(`No image at ${key}.`);
	return img;
}
