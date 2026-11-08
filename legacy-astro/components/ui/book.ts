/**
 * A hardback lying flat, in the IsoGlyph projection (`iso.ts`): the book's height runs along x,
 * its width along y, its thickness up z. The spine faces front left (the lit `l` face), so a
 * title set on it reads left to right; the page edges face front right.
 */
import { box, P, pts } from './iso';

/** Pages per world unit of thickness, the same for every book, so piles compare honestly. */
export const PAGES_PER_UNIT = 40;
/** The tallest pile, in world units; a longer run of books starts a new pile beside it. */
export const PILE_MAX = 64;

const LEN = 30;
const WIDE = 20;
const BOARD = 0.5;
const INSET = 0.7;
/** Spine type: a fraction of the thickness, capped; thinner spines carry no title. */
const FONT_MAX = 3;
const FONT_MIN = 2;
/** Average Geist glyph advance, in ems, used to cut a title to its spine. */
const ADVANCE = 0.56;

export interface PileBook {
	title: string;
	pages: number;
}

export const thickness = (pages: number) => pages / PAGES_PER_UNIT;

/** Cuts a run of books into piles no taller than `PILE_MAX`; a single taller book stands alone. */
export function piles<T extends PileBook>(books: T[]): T[][] {
	const out: T[][] = [];
	let height = Number.POSITIVE_INFINITY;
	for (const book of books) {
		const t = thickness(book.pages);
		if (height + t > PILE_MAX) {
			out.push([]);
			height = 0;
		}
		out[out.length - 1].push(book);
		height += t;
	}
	return out;
}

const hash = (s: string) => {
	let h = 0;
	for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
	return h;
};

/** Fits `title` to `room` world units at `font`, with an ellipsis when it is cut. */
export function spineTitle(title: string, room: number, font: number) {
	const fits = Math.floor(room / (font * ADVANCE));
	return title.length <= fits ? title : `${title.slice(0, Math.max(0, fits - 1)).trimEnd()}…`;
}

/** Everything one book in a pile draws, bottom of the pile first. */
export function stack(books: PileBook[]) {
	let z = 0;
	return books.map((book) => {
		const h = hash(book.title);
		const t = thickness(book.pages);
		// A hand-made pile: each book sits a little off the one under it, spines roughly flush.
		const dx = ((h % 5) - 2) * 0.6;
		const dy = (((h >> 3) % 3) - 1) * 0.4;
		const len = LEN - (h % 4);
		const [x0, x1, y0, y1] = [dx, dx + len, -WIDE + dy, dy];
		const z0 = z;
		z += t;
		const board = Math.min(BOARD, t * 0.18);
		const font = Math.min(FONT_MAX, t * 0.56);
		const [ex, ey] = P(x0 + 1.6, y1, z0 + t / 2);
		const leaves = Math.min(4, Math.floor(book.pages / 120));
		const short = book.title.replace(/\s*\(.*\)\s*$/, '');
		return {
			title: short,
			tone: h % 3,
			lower: box(x0, y0, x1, y1, z0, z0 + board),
			block: box(x0, y0 + INSET, x1 - INSET, y1, z0 + board, z0 + t - board),
			upper: box(x0, y0, x1, y1, z0 + t - board, z0 + t),
			spine: pts([x0, y1, z0 + t], [x1, y1, z0 + t], [x1, y1, z0], [x0, y1, z0]),
			/** Page-edge hairlines across the fore-edge, one per ~120 pages, edge to edge. */
			leaves: Array.from({ length: leaves }, (_, i) => {
				const zi = z0 + board + ((t - 2 * board) * (i + 1)) / (leaves + 1);
				return pts([x1 - INSET, y0 + INSET, zi], [x1 - INSET, y1, zi]);
			}),
			/** Headbands near both ends of the spine, floor to lid. */
			bands: [x0 + 0.9, x1 - 0.9].map((x) => pts([x, y1, z0], [x, y1, z0 + t])),
			text: font >= FONT_MIN ? spineTitle(short, len - 3.2, font) : undefined,
			font,
			// Text x runs along +x on the spine plane, text y runs down -z.
			textMatrix: `matrix(${Math.SQRT1_2} ${Math.SQRT1_2 / 2} 0 ${Math.sqrt(3) / 2} ${ex} ${ey})`,
		};
	});
}

/** The screen box of a pile: [x, y, width, height] in world-projected units. */
export function frameOf(books: PileBook[], pad = 1.5) {
	const top = books.reduce((z, b) => z + thickness(b.pages), 0);
	const corners = [-2, LEN + 2].flatMap((x) =>
		[-WIDE - 1, 1].flatMap((y) => [P(x, y, 0), P(x, y, top)]),
	);
	const xs = corners.map((c) => c[0]);
	const ys = corners.map((c) => c[1]);
	const [minX, minY] = [Math.min(...xs) - pad, Math.min(...ys) - pad];
	return [minX, minY, Math.max(...xs) + pad - minX, Math.max(...ys) + pad - minY].map(
		(v) => +v.toFixed(2),
	);
}
