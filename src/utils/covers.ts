import 'server-only';
import { type Img, maybeImage } from '@/lib/media';

/** Cover art for an ISBN, from `src/assets/covers/<isbn>.<ext>`; undefined when none is checked in. */
export function coverFor(isbn: string): Img | undefined {
	for (const ext of ['jpg', 'png', 'webp']) {
		const img = maybeImage(`covers/${isbn}.${ext}`);
		if (img) return img;
	}
	return undefined;
}
