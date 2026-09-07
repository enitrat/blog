/** Pléiade spine artwork. Fine gold ribbing over the whole leather, a darkened
    label panel carrying the author and title in gold capitals, gold rules at
    head and tail. One SVG per book, authored at the spine's true aspect: the
    plane it is drawn on is as wide as the book is thick, so a fixed-aspect
    drawing would be squeezed on a thin volume and stretched on a fat one. */

/** Nominal drawing height. Every dimension below is in these units; the caller
    rasterises with a density that gives the atlas the pixels it can hold. */
export const ART_HEIGHT = 800;

const GOLD = '#dcc484';
const RULE = '#c8ac6b';
/** Georgia capitals, mean advance per character in em, rounded up. The bake has
    no font metrics, and lettering is fitted rather than condensed: the renderer
    behind `sharp` ignores `textLength`, so a line that does not fit is not
    squeezed, it runs off the leather. Overestimating keeps it on. */
const CAP = 0.83; // includes the letter-spacing added below

const xml = (text) =>
	text.replace(
		/[&<>"']/g,
		(c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c],
	);

const mix = (hex, target, amount) =>
	`#${[0, 1, 2]
		.map((i) => {
			const from = parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16);
			return Math.round(from + (target - from) * amount)
				.toString(16)
				.padStart(2, '0');
		})
		.join('')}`;

/** Greedy fill towards an even character count per line. */
const breakInto = (words, count) => {
	const target = Math.ceil(words.join(' ').length / count);
	const lines = [];
	for (const word of words) {
		const last = lines.at(-1);
		if (last !== undefined && lines.length >= count) lines[lines.length - 1] = `${last} ${word}`;
		else if (last !== undefined && last.length + 1 + word.length <= target)
			lines[lines.length - 1] = `${last} ${word}`;
		else lines.push(word);
	}
	return lines;
};

/** The largest lettering that fits `width`, over as few lines as that size
    allows. A long single word simply sets small, as it does on a real spine. */
const fit = (text, width, ideal, maxLines) => {
	const words = text.toUpperCase().split(/\s+/).filter(Boolean);
	let best = { size: 0, lines: words };
	for (let count = 1; count <= maxLines; count += 1) {
		const lines = breakInto(words, count);
		const longest = Math.max(...lines.map((entry) => entry.length));
		const size = Math.min(ideal, width / (longest * CAP));
		if (size > best.size) best = { size, lines };
		if (lines.length < count) break; // out of words to break
	}
	return best;
};

/**
 * @param {{title: string, author: string, color: string}} book
 * @param {number} aspect spine width over spine height, as the plane is built
 */
export const spineSvg = (book, aspect) => {
	const height = ART_HEIGHT;
	const width = Math.round(height * aspect);
	const hinge = Math.max(1, width * 0.035);
	const inner = width - 2 * hinge - height * 0.012;
	const leather = book.color;

	// Raised bands: a gold hairline over its own shadow, the length of the spine.
	const step = height / 132;
	const bands = [];
	for (let y = height * 0.075; y < height * 0.928; y += step)
		bands.push(
			`<path d="M${hinge.toFixed(1)} ${y.toFixed(1)}h${(width - 2 * hinge).toFixed(1)}"/>`,
		);
	const rules = [height * 0.045, height * 0.056, height * 0.944, height * 0.955]
		.map(
			(y) => `<path d="M${hinge.toFixed(1)} ${y.toFixed(1)}h${(width - 2 * hinge).toFixed(1)}"/>`,
		)
		.join('');

	const title = fit(book.title, inner, height * 0.0205, 5);
	/* One panel, one voice: a long title sets small, and the author comes down
	   with it rather than towering over its own book. */
	const author = fit(book.author, inner, Math.min(height * 0.024, title.size * 1.4), 2);
	const step_a = author.size * 1.5;
	const step_t = title.size * 1.5;
	const gap = Math.max(author.size, title.size) * 1.4;
	const body = author.lines.length * step_a + gap + title.lines.length * step_t - step_t * 0.25;
	/* One plate, at one place, on every volume of the collection: the panels line
	   up across the shelf, and short titles get air rather than a smaller box. */
	const top = height * 0.19;
	const panel = Math.max(body + height * 0.048, height * 0.155);

	const lettering = [];
	let cursor = top + (panel - body) / 2 + author.size * 0.85;
	for (const text of author.lines) {
		lettering.push(`<text y="${cursor.toFixed(1)}">${xml(text)}</text>`);
		cursor += step_a;
	}
	cursor += gap - step_a + title.size * 0.85;
	const titleLines = title.lines.map((text) => {
		const line = `<text y="${cursor.toFixed(1)}">${xml(text)}</text>`;
		cursor += step_t;
		return line;
	});

	const box = `x="${hinge.toFixed(1)}" y="${top.toFixed(1)}" width="${(width - 2 * hinge).toFixed(1)}" height="${panel.toFixed(1)}"`;
	const type = `text-anchor="middle" font-family="Georgia,'Times New Roman',serif" fill="${GOLD}"`;
	return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
<rect width="${width}" height="${height}" fill="${leather}"/>
<g fill="none" stroke="#000" stroke-width="${(height * 0.0013).toFixed(2)}" opacity=".24" transform="translate(0 ${(step * 0.45).toFixed(2)})">${bands.join('')}</g>
<g fill="none" stroke="${GOLD}" stroke-width="${(height * 0.0013).toFixed(2)}" opacity=".38">${bands.join('')}</g>
<g fill="none" stroke="${RULE}" stroke-width="${(height * 0.002).toFixed(2)}" opacity=".8">${rules}</g>
<rect ${box} fill="${mix(leather, 0, 0.38)}"/>
<rect ${box} fill="none" stroke="${RULE}" stroke-width="${(height * 0.0019).toFixed(2)}" opacity=".9"/>
<g ${type} transform="translate(${(width / 2).toFixed(1)} 0)">
<g font-size="${author.size.toFixed(2)}" letter-spacing="${(author.size * 0.04).toFixed(2)}">${lettering.join('')}</g>
<g font-size="${title.size.toFixed(2)}" letter-spacing="${(title.size * 0.04).toFixed(2)}">${titleLines.join('')}</g>
</g>
<rect width="${hinge.toFixed(1)}" height="${height}" fill="#000" opacity=".22"/>
<rect x="${(width - hinge).toFixed(1)}" width="${hinge.toFixed(1)}" height="${height}" fill="#000" opacity=".22"/>
</svg>`;
};
