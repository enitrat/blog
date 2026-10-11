// Shared by every DitherImage (and Prose figures): 8x8 ordered-dither duotone on a canvas over the image.
const CELL = 2; // CSS px per dither dot

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

const probe = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
/** Any CSS colour string to [r,g,b]. */
const rgb = (css: string): number[] => {
	if (!probe) return [0, 0, 0];
	probe.clearRect(0, 0, 1, 1);
	// An invalid string is ignored by the canvas; reset first so it falls back to black.
	probe.fillStyle = '#000';
	probe.fillStyle = css.trim() || '#000';
	probe.fillRect(0, 0, 1, 1);
	return [...probe.getImageData(0, 0, 1, 1).data];
};
const lum = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

/** Two semantic colours for a tone; the darker one always takes the shadows, in either theme. */
const pair = (box: HTMLElement, tone: string): number[][] => {
	// Read from the box, not :root, so scoped (inverse) surfaces dither in their own colours.
	const cs = getComputedStyle(box);
	const v = (n: string) => rgb(cs.getPropertyValue(n));
	const [a, b] =
		tone === 'accent' ? [v('--accent'), v('--surface')] : [v('--foreground'), v('--surface')];
	return lum(a) <= lum(b) ? [a, b] : [b, a];
};

function paint(box: HTMLElement) {
	const img = box.querySelector('img');
	const cv = box.querySelector('canvas');
	if (!img || !cv || !img.naturalWidth || !box.clientWidth) return;
	const w = Math.max(1, Math.round(box.clientWidth / CELL));
	const h = Math.max(1, Math.round(box.clientHeight / CELL));
	const src = document.createElement('canvas');
	src.width = w;
	src.height = h;
	const sc = src.getContext('2d', { willReadFrequently: true });
	if (!sc) return;
	// object-fit: cover crop
	const r = w / h;
	let sw = img.naturalWidth;
	let sh = img.naturalHeight;
	let sx = 0;
	let sy = 0;
	if (sw / sh > r) {
		sw = sh * r;
		sx = (img.naturalWidth - sw) / 2;
	} else {
		sh = sw / r;
		sy = (img.naturalHeight - sh) / 2;
	}
	sc.drawImage(img, sx, sy, sw, sh, 0, 0, w, h);
	let px: Uint8ClampedArray;
	try {
		px = sc.getImageData(0, 0, w, h).data;
	} catch {
		return; // tainted: leave the plain image
	}
	const L = new Float32Array(w * h);
	const hist = new Uint32Array(256);
	for (let i = 0; i < L.length; i++) {
		L[i] = lum([px[i * 4], px[i * 4 + 1], px[i * 4 + 2]]);
		hist[L[i] | 0]++;
	}
	const q = (p: number) => {
		let s = 0;
		for (let i = 0; i < 256; i++) {
			s += hist[i];
			if (s >= p * L.length) return i;
		}
		return 255;
	};
	const lo = q(0.02);
	const hi = Math.max(q(0.98), lo + 8); // stretch contrast so flat photos still dither
	const [dark, light] = pair(box, box.dataset.tone ?? 'ink');
	cv.width = w;
	cv.height = h;
	const ctx = cv.getContext('2d');
	if (!ctx) return;
	const out = ctx.createImageData(w, h);
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++) {
			const v = Math.min(1, Math.max(0, (L[y * w + x] - lo) / (hi - lo)));
			const c = v < (BAYER[y & 7][x & 7] + 0.5) / 64 ? dark : light;
			out.data.set([c[0], c[1], c[2], 255], (y * w + x) * 4);
		}
	ctx.putImageData(out, 0, 0);
	box.dataset.ready = '';
}

const boxes = new Set<HTMLElement>();
const repaint = () => {
	for (const b of boxes) paint(b);
};
let timer: number;
const debounced = () => {
	clearTimeout(timer);
	timer = window.setTimeout(repaint, 120);
};
const ro = new ResizeObserver(debounced);

function init(box: HTMLElement) {
	if (boxes.has(box)) return;
	boxes.add(box);
	const img = box.querySelector('img');
	if (!img) return;
	if (box.dataset.reveal === 'hover') {
		// Only interactive images get a button.
		const btn = document.createElement('button');
		btn.type = 'button';
		btn.setAttribute('aria-pressed', 'false');
		btn.addEventListener('click', () =>
			btn.setAttribute('aria-pressed', String(btn.getAttribute('aria-pressed') !== 'true')),
		);
		btn.append(...box.childNodes);
		box.append(btn);
	}
	img.complete ? paint(box) : img.addEventListener('load', () => paint(box));
	ro.observe(box);
}

/** Dither every `[data-dither]` under root (idempotent). */
export function ditherIn(root: ParentNode, selector = '[data-dither]') {
	for (const el of root.querySelectorAll<HTMLElement>(selector)) init(el);
}

/** Wrap bare images (e.g. inside Prose) so they get the same treatment. */
export function wrapImages(root: ParentNode) {
	for (const img of root.querySelectorAll('img')) {
		if (img.parentElement?.tagName === 'PICTURE') continue;
		const box = document.createElement('span');
		box.className = 'dither';
		box.dataset.dither = '';
		box.dataset.tone = 'ink';
		box.dataset.reveal = 'hover';
		box.style.setProperty(
			'--ratio',
			`${img.getAttribute('width')} / ${img.getAttribute('height')}`,
		);
		img.replaceWith(box);
		const cv = document.createElement('canvas');
		cv.setAttribute('aria-hidden', 'true');
		box.append(img, cv);
	}
}

ditherIn(document);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', repaint);
new MutationObserver(repaint).observe(document.documentElement, {
	attributes: true,
	attributeFilter: ['data-theme'],
});
