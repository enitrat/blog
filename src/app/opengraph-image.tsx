import { OG_SIZE, wordmarkOgImage } from '@/lib/og';

export const alt = 'MSAUG.DEV: cool things i do on the internet';
export const size = OG_SIZE;
export const contentType = 'image/png';

export default async function Image() {
	return wordmarkOgImage({ line: 'cool things i do on the internet' });
}
