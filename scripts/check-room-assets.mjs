/** Check the baked room against the content it must show, without a browser.
 * Runs in `bun run build`: a note, post or book added without the matching bake
 * fails here instead of shipping a room that quietly lacks it. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { englishPieces, notedIsbns } from './room-content.mjs';
import { ROOM_GROUPS } from './room-targets.mjs';

const DIR = 'src/assets/room';
const BUDGET = 11_000_000;
const json = async (name) => JSON.parse(await readFile(`${DIR}/${name}`, 'utf8'));

let bytes = 0;
const nodes = new Set();
for (const group of ROOM_GROUPS) {
	const glb = await readFile(`${DIR}/${group}.glb`);
	bytes += glb.length;
	// GLB: 12-byte header, then the JSON chunk's length, type, and body.
	const gltf = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)).toString());
	for (const node of gltf.nodes ?? []) nodes.add(node.name);
}

const slots = await json('book-slots.json');
const sheets = await json('sheets.json');
const noted = await notedIsbns();
const pieces = await englishPieces();
const shelved = new Set(slots.map((slot) => slot.isbn));
const expected = [
	'spines',
	'Platter',
	'Tonearm',
	'Bookmark',
	...slots.map((slot) => `Body_${slot.isbn}`),
	...[...noted].filter((isbn) => shelved.has(isbn)).map((isbn) => `Cover_${isbn}`),
	...pieces.map((piece) => `Sheet_${piece.slug}`),
];
const missing = expected.filter((name) => !nodes.has(name));
assert.deepEqual(
	missing,
	[],
	`Missing baked nodes; rerun room:bake for them: ${missing.join(', ')}`,
);
assert.deepEqual(
	sheets.map((sheet) => sheet.slug),
	pieces.map((piece) => piece.slug),
	'sheets.json is out of date; run `bun run room:bake --only sheets`.',
);
assert.ok(bytes <= BUDGET, `Room GLBs are ${bytes} bytes; the budget is ${BUDGET}.`);
console.log(`room assets ok: ${nodes.size} nodes, ${(bytes / 1e6).toFixed(2)} MB`);
