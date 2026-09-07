/** Bake original room assets with Blender 4.5; Blender is an offline tool only. */
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import sharp from 'sharp';
import { books } from '../src/booksData.ts';
import { PLEIADE_HEX, pleiadeStyleFor } from '../src/utils/pleiade.ts';

const option = (name, fallback) => {
	const index = process.argv.indexOf(name);
	return index < 0 ? fallback : process.argv[index + 1];
};
const work = process.env.ROOM_WORK ?? (await mkdtemp(join(tmpdir(), 'living-room-')));
const out = resolve(option('--out', 'src/assets/room'));
await mkdir(work, { recursive: true });
const library = books.map((book) => ({
	isbn: book.edition.isbn13,
	title: book.title,
	author: pleiadeStyleFor(book.author).label,
	color: PLEIADE_HEX[pleiadeStyleFor(book.author).color],
	pages: book.edition.pageCount ?? 500,
}));
// The bookcase, in the runtime's coordinates: metres, Y up, the camera down +Z.
// Blender is Z up and its Y is depth, so room.py negates `face` to get back to
// its own axes. This is the only description of the cabinet anywhere: room.py
// builds it from the copy written beside the assets, and the browser frames,
// anchors and picks it from the same file.
const CABINET = {
	x: -1.12, // centre of the carcass
	width: 1.08, // outer width, side panel to side panel
	usable: 0.98, // clear span between the sides, where books may stand
	depth: 0.4,
	face: -1.04, // front plane of the carcass
	front: -1.068, // the plane the spines stand on, recessed behind the face
	floor: 0.02, // top of the plinth
	ceiling: 1.713, // top of the crown
	shelves: [0.18, 0.55, 0.91, 1.27],
	bays: [0.55, 0.91, 1.27], // the shelves the library stands on; 0.18 holds records
};
const leftmost = CABINET.x - CABINET.usable / 2;
const rightmost = CABINET.x + CABINET.usable / 2;

// These slots are consumed by Blender and shipped with its assets. The browser
// never reconstructs positions from the current order of booksData.
const perShelf = Math.ceil(library.length / CABINET.bays.length);
const slots = library.map((book, index) => {
	const row = Math.floor(index / perShelf);
	const start = row * perShelf;
	const widthOf = (entry) => 0.022 + (Math.min(entry.pages, 2200) / 2200) * 0.021;
	const width = widthOf(book);
	const left =
		leftmost + library.slice(start, index).reduce((x, entry) => x + widthOf(entry) + 0.0015, 0);
	if (left + width > rightmost)
		throw new Error('The room bookcase is full. Add shelf space before baking more books.');
	return {
		isbn: book.isbn,
		row,
		x: left + width / 2,
		y: CABINET.bays[row],
		z: CABINET.front,
		width,
		height: 0.237 + (index % 4) * 0.002,
	};
});
// Blender reads these from the work directory; the browser imports the pair
// written beside the GLBs.
const publish = async (directory, pretty) => {
	// Tabs, so the checked-in copies match what `bun run lint` expects.
	const space = pretty ? '\t' : undefined;
	const end = pretty ? '\n' : '';
	await writeFile(join(directory, 'book-slots.json'), JSON.stringify(slots, null, space) + end);
	await writeFile(join(directory, 'cabinet.json'), JSON.stringify(CABINET, null, space) + end);
};
await publish(work, false);
// The browser imports these from the asset directory, so every run writes them
// there, not only --layout-only: a bake that moved the cabinet must not leave
// the runtime framing and picking the shape it had before.
await mkdir(out, { recursive: true });
await publish(out, true);
if (process.argv.includes('--layout-only')) process.exit(0);
await writeFile(join(work, 'books.json'), JSON.stringify(library));
const escape = (text) =>
	text.replace(
		/[&<>"']/g,
		(c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c],
	);
for (const [index, book] of library.entries()) {
	const words = book.title.toUpperCase().split(/\s+/);
	const lines = [];
	for (const word of words) {
		if (lines.length && (lines.at(-1) + ' ' + word).length <= 12)
			lines[lines.length - 1] += ' ' + word;
		else lines.push(word);
	}
	const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="128" height="768"><rect width="128" height="768" fill="${book.color}"/><path d="M5 0v768M123 0v768" stroke="#10100e" opacity=".35" stroke-width="5"/><g fill="#d4bd7d" text-anchor="middle" font-family="Georgia,serif"><text x="64" y="135" font-size="20" textLength="108" lengthAdjust="spacingAndGlyphs">${escape(book.author)}</text>${lines
		.slice(0, 7)
		.map(
			(line, i) =>
				`<text x="64" y="${225 + i * 33}" font-size="14" textLength="${Math.min(108, line.length * 9)}" lengthAdjust="spacingAndGlyphs">${escape(line)}</text>`,
		)
		.join(
			'',
		)}<text x="64" y="640" font-size="12">PLÉIADE</text><text x="64" y="675" font-size="10">GALLIMARD</text></g><g stroke="#c8ac6b" stroke-width="3">${[54, 61, 171, 178, 581, 588, 718, 725].map((y) => `<path d="M8 ${y}h112"/>`).join('')}</g></svg>`;
	/* Rendered well above the 128x768 the SVG is authored at: these jackets are
	   the only type in the room a visitor is meant to actually read, and the
	   bake cannot invent detail the source does not have. */
	await sharp(Buffer.from(svg), { density: 288 })
		.png()
		.toFile(join(work, `book-${index}.png`));
}
// Original geometric artwork, with no invented attribution or borrowed cover art.
for (const [name, width, height] of [
	['print', 720, 840],
	['sleeve', 640, 640],
]) {
	const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 720 840"><rect width="720" height="840" fill="#d7cbb2"/><rect x="70" y="70" width="580" height="640" fill="#3d5358"/><circle cx="360" cy="390" r="240" fill="#ae5940"/><path d="M120 390h480M120 440h480M145 490h430M185 540h350M245 590h230" stroke="#d7cbb2" stroke-width="17"/><circle cx="360" cy="390" r="55" fill="#d7cbb2"/><circle cx="360" cy="390" r="15" fill="#3d5358"/></svg>`;
	await sharp(Buffer.from(svg))
		.png()
		.toFile(join(work, `${name}.png`));
}
console.log(`Blender source and intermediate textures: ${work}`);
const child = spawn(
	process.env.BLENDER ?? 'blender',
	[
		'--background',
		'--factory-startup',
		'--python',
		resolve('scripts/room.py'),
		'--',
		work,
		out,
		option('--samples', '256'),
		option('--size', '2048'),
		...(process.argv.includes('--preview') ? ['--preview'] : []),
		...(process.argv.includes('--spines-only') ? ['--spines-only'] : []),
	],
	{ stdio: 'inherit' },
);
child.on('error', (error) => {
	console.error(`Cannot start Blender: ${error.message}. Set BLENDER to its executable.`);
	process.exitCode = 1;
});
child.on('exit', (code) => {
	process.exitCode = code ?? 1;
});
