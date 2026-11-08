/** An 8x8 ordered-dither duotone of an image, drawn on a canvas laid over it. */

/** CSS px per dither dot. */
const CELL = 2;

type Rgb = readonly [number, number, number];

/** Resolves CSS colours, in any syntax a canvas accepts, to RGB. */
export function toRgb(...colors: string[]): Rgb[] {
	const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
	return colors.map((c) => {
		if (!ctx) return [0, 0, 0];
		ctx.clearRect(0, 0, 1, 1);
		ctx.fillStyle = c;
		ctx.fillRect(0, 0, 1, 1);
		const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
		return [r, g, b];
	});
}

const bayer = (n: number): number[][] => {
	let m = [[0]];
	while (m.length < n) {
		const s = m.length;
		m = Array.from({ length: 2 * s }, (_, y) =>
			Array.from(
				{ length: 2 * s },
				(_, x) =>
					4 * m[y % s][x % s] +
					[
						[0, 2],
						[3, 1],
					][+(y >= s)][+(x >= s)],
			),
		);
	}
	return m;
};
const BAYER = bayer(8);

const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** Duotones `px` (RGBA, `w` wide) in place: `dark` takes the shadows, `light` the rest. */
export function dither(px: Uint8ClampedArray, w: number, dark: Rgb, light: Rgb) {
	const n = px.length / 4;
	const L = new Float32Array(n);
	const hist = new Uint32Array(256);
	for (let i = 0; i < n; i++) {
		L[i] = lum(px[i * 4], px[i * 4 + 1], px[i * 4 + 2]);
		hist[L[i] | 0]++;
	}
	const q = (p: number) => {
		let s = 0;
		for (let i = 0; i < 256; i++) {
			s += hist[i];
			if (s >= p * n) return i;
		}
		return 255;
	};
	const lo = q(0.02);
	const hi = Math.max(q(0.98), lo + 8); // stretch contrast so flat photos still dither
	for (let i = 0; i < n; i++) {
		const x = i % w;
		const y = (i - x) / w;
		const v = Math.min(1, Math.max(0, (L[i] - lo) / (hi - lo)));
		const c = v < (BAYER[y & 7][x & 7] + 0.5) / 64 ? dark : light;
		px.set([c[0], c[1], c[2], 255], i * 4);
	}
	return px;
}

/** Paints `img` dithered onto `cv`, cropped like `object-fit: cover`. False when it cannot yet. */
export function paint(img: HTMLImageElement, cv: HTMLCanvasElement, dark: Rgb, light: Rgb) {
	if (!img.complete || !img.naturalWidth || !cv.clientWidth) return false;
	const w = Math.max(1, Math.round(cv.clientWidth / CELL));
	const h = Math.max(1, Math.round(cv.clientHeight / CELL));
	cv.width = w;
	cv.height = h;
	const ctx = cv.getContext('2d', { willReadFrequently: true });
	if (!ctx) return false;
	const r = w / h;
	let [sx, sy, sw, sh] = [0, 0, img.naturalWidth, img.naturalHeight];
	if (sw / sh > r) {
		sw = sh * r;
		sx = (img.naturalWidth - sw) / 2;
	} else {
		sh = sw / r;
		sy = (img.naturalHeight - sh) / 2;
	}
	ctx.drawImage(img, sx, sy, sw, sh, 0, 0, w, h);
	let data: ImageData;
	try {
		data = ctx.getImageData(0, 0, w, h);
	} catch {
		return false; // A cross-origin image taints the canvas: the plain image stays.
	}
	dither(data.data, w, dark, light);
	ctx.putImageData(data, 0, 0);
	return true;
}
