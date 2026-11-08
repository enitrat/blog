import 'server-only';
import { type MediaEntry, type MediaKey, media } from '@/generated/media';

export type Img = {
	src: string;
	width: number;
	height: number;
	/** `srcset` of WebP widths. */
	srcSet: string;
	/** Source file, relative to the repo root (build-time reads such as `glowOf`). */
	file: string;
};

const toImg = (key: string, e: MediaEntry): Img => {
	if (!e.width || !e.height || !e.file) throw new Error(`${key} is not an image`);
	return {
		src: e.src,
		width: e.width,
		height: e.height,
		srcSet: (e.srcSet ?? []).map((s) => `${s.src} ${s.width}w`).join(', '),
		file: e.file,
	};
};

/** A published image by its path under `src/assets/` (or `content/...` under `src/content/`). */
export function image(key: MediaKey): Img {
	return toImg(key, media[key]);
}

/** Like `image`, but undefined when nothing is checked in under that key. */
export function maybeImage(key: string): Img | undefined {
	const e = (media as Record<string, MediaEntry>)[key];
	return e ? toImg(key, e) : undefined;
}

/** Every published file whose key starts with `prefix`. */
export function mediaUnder(prefix: string): [string, MediaEntry][] {
	return Object.entries(media).filter(([k]) => k.startsWith(prefix));
}
