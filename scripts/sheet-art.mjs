/** Manuscript sheets for the writing desk. One sheet per published piece: its
    title in a copperplate hand across the head of the page, the date beneath,
    then the body scribbled in ink -- lines that wander, a struck-out phrase,
    an afterthought in the margin. The head is what shows when the sheets lie
    fanned on the desk, so the title sits high and reads at arm's length. Drawn
    at A4 and rasterised by the caller. Fonts are the bake machine's own:
    Snell Roundhand ships with macOS, and the fallbacks are ordinary cursives. */

/** Drawing size, in the same units as every dimension below. A4 at 4 per mm. */
export const SHEET = { width: 840, height: 1188 };

const INK = '#1f2a4a';
const PAPER = '#efe6cf';
const HAND = "'Snell Roundhand','Zapfino','Apple Chancery',cursive";

const xml = (text) =>
	text.replace(
		/[&<>"']/g,
		(c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c],
	);

/** A small deterministic generator, so a rebake draws the same page. */
const noise = (seed) => {
	let state = seed * 2654435761 + 1;
	return () => {
		state ^= state << 13;
		state ^= state >>> 17;
		state ^= state << 5;
		return ((state >>> 0) % 10000) / 10000;
	};
};

/** Break a title into at most `max` lines of roughly even length. */
const lines = (title, max) => {
	const words = title.split(/\s+/).filter(Boolean);
	const perLine = Math.ceil(words.join(' ').length / max);
	const result = [];
	for (const word of words) {
		const last = result.at(-1);
		if (last !== undefined && (last.length + 1 + word.length <= perLine || result.length >= max))
			result[result.length - 1] = `${last} ${word}`;
		else result.push(word);
	}
	return result;
};

/** A line of handwriting rendered as ink: a cubic path that rises and falls
    like a baseline with words on it, broken by word gaps. */
const scribble = (random, x, y, width, weight) => {
	const parts = [];
	let cursor = x;
	while (cursor < x + width) {
		const word = Math.min(x + width - cursor, 40 + random() * 110);
		if (word < 24) break;
		const bumps = Math.max(2, Math.round(word / 22));
		let d = `M${cursor.toFixed(1)} ${(y + (random() - 0.5) * 4).toFixed(1)}`;
		for (let i = 0; i < bumps; i += 1) {
			const step = word / bumps;
			// Mostly minims, now and then an ascender or a descender.
			const up = random() < 0.18 ? 20 + random() * 10 : 6 + random() * 9;
			const down = random() < 0.12 ? 8 + random() * 8 : random() * 3;
			d += ` q${(step * (0.25 + random() * 0.2)).toFixed(1)} ${(-up).toFixed(1)} ${(step * 0.6).toFixed(1)} ${(random() * 3 - 1).toFixed(1)} t${(step * 0.4).toFixed(1)} ${down.toFixed(1)}`;
		}
		parts.push(`<path d="${d}" stroke-width="${weight.toFixed(1)}"/>`);
		cursor += word + 14 + random() * 18;
	}
	return parts.join('');
};

/**
 * @param {{title: string, date: string}} piece
 * @param {number} seed sheet index, so no two pages scribble alike
 */
export const sheetSvg = (piece, seed) => {
	const random = noise(seed + 7);
	const { width, height } = SHEET;
	const margin = 78;
	const measure = width - margin * 2;
	// Two lines at most across the head; a long title sets smaller rather than
	// spilling into the body where the fan of the pile would hide it.
	const single = (measure / piece.title.length) * 2.05;
	const heading = single >= 44 ? [piece.title] : lines(piece.title, 2);
	const longest = Math.max(...heading.map((line) => line.length));
	const size = Math.min(58, (measure / longest) * 2.05);
	const leading = size * 1.25;
	const head = [];
	let y = 96 + size * 0.8;
	for (const line of heading) {
		head.push(
			`<text x="${(width / 2).toFixed(1)}" y="${y.toFixed(1)}" font-size="${size.toFixed(1)}">${xml(line)}</text>`,
		);
		y += leading;
	}
	const dateY = y + 8;
	head.push(
		`<text x="${(width - margin).toFixed(1)}" y="${dateY.toFixed(1)}" font-size="26" text-anchor="end" opacity=".8">${xml(piece.date)}</text>`,
	);
	// The body: paragraphs of scribbled lines, a struck phrase, a marginal note.
	const body = [];
	let line = dateY + 70;
	let paragraph = 0;
	while (line < height - 110) {
		const last = random() < 0.22;
		const start = margin + (paragraph === 0 || random() < 0.3 ? 34 : 0);
		const length = last
			? measure * (0.35 + random() * 0.4)
			: measure - (start - margin) - random() * 20;
		body.push(scribble(random, start, line, length, 2.2 + random() * 0.8));
		if (random() < 0.09) {
			const at = start + random() * (length - 120);
			body.push(
				`<path d="M${at.toFixed(1)} ${(line - 6).toFixed(1)}l${(70 + random() * 60).toFixed(1)} ${(random() * 6 - 3).toFixed(1)}" stroke-width="3"/>`,
			);
		}
		line += 40 + random() * 5;
		if (last) {
			line += 26;
			paragraph += 1;
		}
	}
	if (random() < 0.7) {
		const at = dateY + 200 + random() * 500;
		body.push(
			`<g transform="rotate(-90 ${(margin * 0.45).toFixed(1)} ${at.toFixed(1)})">${scribble(random, margin * 0.45 - 90, at, 180, 1.8)}</g>`,
		);
	}
	// A blot where the pen rested.
	const blot =
		random() < 0.5
			? `<ellipse cx="${(margin + measure * random()).toFixed(1)}" cy="${(dateY + 80 + random() * 700).toFixed(1)}" rx="${(4 + random() * 4).toFixed(1)}" ry="${(3 + random() * 3).toFixed(1)}" fill="${INK}" opacity=".7"/>`
			: '';
	return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
<rect width="${width}" height="${height}" fill="${PAPER}"/>
<g font-family="${HAND}" fill="${INK}" text-anchor="middle">${head.join('')}</g>
<g fill="none" stroke="${INK}" stroke-linecap="round" opacity=".82" transform="translate(${(Math.tan((9 * Math.PI) / 180) * (height / 2)).toFixed(1)} 0) skewX(-9)">${body.join('')}</g>
${blot}
</svg>`;
};
