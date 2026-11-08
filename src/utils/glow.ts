import 'server-only';
import sharp from 'sharp';
import type { Img } from '@/lib/media';

/**
 * A cover's glow colour as `rgb(r g b)`: the average of an 8x8 thumbnail, weighted by saturation so a
 * white margin or black band doesn't wash it out. Build time only: reads the source file.
 */
export async function glowOf(image: Pick<Img, 'file'>): Promise<string> {
	const { data } = await sharp(image.file)
		.resize(8, 8, { fit: 'fill' })
		.removeAlpha()
		.raw()
		.toBuffer({ resolveWithObject: true });
	let r = 0,
		g = 0,
		b = 0,
		w = 0;
	for (let i = 0; i < data.length; i += 3) {
		const max = Math.max(data[i], data[i + 1], data[i + 2]);
		const weight = 0.05 + (max - Math.min(data[i], data[i + 1], data[i + 2])) / 255;
		r += data[i] * weight;
		g += data[i + 1] * weight;
		b += data[i + 2] * weight;
		w += weight;
	}
	return `rgb(${Math.round(r / w)} ${Math.round(g / w)} ${Math.round(b / w)})`;
}
