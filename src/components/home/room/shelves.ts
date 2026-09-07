/** Every fact about where the bookcase and its books are. Both come from the
 *  bake: `bake-room.mjs` writes them and `room.py` builds the furniture from the
 *  same numbers, so a constant here cannot fall behind the geometry on screen. */

import slots from '../../../assets/room/book-slots.json';
import cabinet from '../../../assets/room/cabinet.json';

export type Slot = (typeof slots)[number];
export type ShelfFrame = {
	view: 'cabinet' | 'row';
	x: number;
	y: number;
	z: number;
	width: number;
	height: number;
};

/** Top shelf first, left to right, the order a reader would take them in. */
export const ordered = [...slots].sort((a, b) => b.row - a.row || a.x - b.x);

export const rows = [...new Set(ordered.map((slot) => slot.row))]
	.sort((a, b) => b - a)
	.map((row) => ({ row, books: ordered.filter((slot) => slot.row === row) }));

/** Named by position in the cabinet, not by the row number the bake happened to
 *  assign: a four-shelf bake names four shelves instead of returning undefined. */
export function rowName(row: number) {
	const index = rows.findIndex((entry) => entry.row === row);
	if (index === 0) return 'Top row';
	if (index === rows.length - 1) return 'Bottom row';
	if (rows.length === 3) return 'Middle row';
	return `Row ${index + 1} from the top`;
}

/** Composition margin around the carcass, so the cabinet does not touch the
 *  edge of the picture. Wider at the sides than at the ends, as composed. */
const MARGIN = { side: 0.15, end: 0.12 };

/** The whole cabinet, plinth to crown: what the first click frames. */
export const CABINET: ShelfFrame = {
	view: 'cabinet',
	x: cabinet.x,
	y: (cabinet.floor + cabinet.ceiling) / 2,
	z: cabinet.front,
	width: cabinet.width + MARGIN.side * 2,
	height: cabinet.ceiling - cabinet.floor + MARGIN.end * 2,
};

const spines = {
	bottom: Math.min(...ordered.map((slot) => slot.y)),
	top: Math.max(...ordered.map((slot) => slot.y + slot.height)),
};

/** The overview ring sits on the books, which fill only part of the carcass. */
export const BOOKSHELF_ANCHOR = {
	x: cabinet.x,
	y: (spines.bottom + spines.top) / 2,
	z: cabinet.front,
};

/** The pick volume for the cabinet body, a little larger than the carcass so
 *  the pointer does not have to be exact. */
const GRAB = 0.03;
export const CABINET_BOX = {
	min: [
		cabinet.x - cabinet.width / 2 - GRAB,
		cabinet.floor - GRAB,
		cabinet.face - cabinet.depth - GRAB,
	],
	max: [cabinet.x + cabinet.width / 2 + GRAB, cabinet.ceiling + GRAB, cabinet.face + GRAB],
} as const;
