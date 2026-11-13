import { SITE_DESCRIPTION } from '@/consts';
import { OG_SIZE, wordmarkOgImage } from '@/lib/og';

export const alt = `MSAUG.DEV: ${SITE_DESCRIPTION}`;
export const size = OG_SIZE;
export const contentType = 'image/png';

export default async function Image() {
	return wordmarkOgImage({ line: SITE_DESCRIPTION });
}
