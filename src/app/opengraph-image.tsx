import { SITE_DESCRIPTION } from '@/consts';
import { OG_SIZE, ogImage } from '@/lib/og';

export const alt = 'Mathieu Saugier';
export const size = OG_SIZE;
export const contentType = 'image/png';

export default async function Image() {
	return ogImage({
		title: 'Mathieu Saugier',
		titleSize: 92,
		kicker: SITE_DESCRIPTION,
		footer: 'msaug.dev',
	});
}
