/** Bake original room assets with Blender 4.5; Blender is an offline tool only. */
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import sharp from 'sharp';
import { books } from '../src/booksData.ts';
import { PLEIADE_HEX, pleiadeStyleFor } from '../src/utils/pleiade.ts';
import { SHEET, sheetSvg } from './sheet-art.mjs';
import { ART_HEIGHT, coverSvg, spineSvg } from './spine-art.mjs';

const option = (name, fallback) => {
	const index = process.argv.indexOf(name);
	return index < 0 ? fallback : process.argv[index + 1];
};
const work = process.env.ROOM_WORK ?? (await mkdtemp(join(tmpdir(), 'living-room-')));
const out = resolve(option('--out', 'src/assets/room'));
await mkdir(work, { recursive: true });
// Only an annotated volume can be opened, so only its front cover is ever seen
// off the shelf. Baking the other fifty-odd spends the atlas on faces nobody
// can reach. A note's filename is its ISBN, the same contract `notedIsbns()`
// enforces against the shelves at build time; `astro:content` is not available
// out here, so the directory is read directly.
const noted = new Set(
	(await readdir('src/content/notes')).map((file) => file.replace(/\.mdx?$/, '')),
);
const library = books.map((book) => ({
	isbn: book.edition.isbn13,
	noted: noted.has(book.edition.isbn13),
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
// A Pléiade volume is one height, whatever it holds; only its thickness varies,
// with the number of leaves. Both are the same numbers Blender builds from.
const SPINE_HEIGHT = 0.237;
const widthOf = (entry) => 0.017 + (Math.min(entry.pages, 1400) / 1400) * 0.03;
const leftmost = CABINET.x - CABINET.usable / 2;
const rightmost = CABINET.x + CABINET.usable / 2;

// These slots are consumed by Blender and shipped with its assets. The browser
// never reconstructs positions from the current order of booksData.
const perShelf = Math.ceil(library.length / CABINET.bays.length);
const slots = library.map((book, index) => {
	const row = Math.floor(index / perShelf);
	const start = row * perShelf;
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
		height: SPINE_HEIGHT,
	};
});
// The writing lies on the desk, one manuscript per published piece. A piece is
// a directory in the blog collection whose route is its directory name, and
// the English original is what the index lists; `astro:content` is not
// available out here, so the frontmatter is read directly, as the notes are.
const pieces = [];
for (const directory of (await readdir('src/content/blog', { withFileTypes: true })).filter(
	(entry) => entry.isDirectory(),
)) {
	const files = (await readdir(join('src/content/blog', directory.name))).filter((file) =>
		/\.mdx?$/.test(file),
	);
	for (const file of files) {
		const source = await readFile(join('src/content/blog', directory.name, file), 'utf8');
		const front = source.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '';
		const field = (name) =>
			front
				.match(new RegExp(`^${name}:\\s*(.+)$`, 'm'))?.[1]
				.trim()
				.replace(/^(['"])(.*)\1$/, '$2');
		if ((field('lang') ?? 'en') !== 'en') continue;
		const date = new Date(field('pubDate'));
		pieces.push({
			slug: directory.name,
			title: field('title'),
			date: date.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' }),
			time: date.valueOf(),
		});
	}
}
pieces.sort((a, b) => b.time - a.time);
// Two fanned piles on the leather, newest on top and nearest the chair, each
// older sheet pushed a little further from the writer so its head -- where
// the title is written -- shows past the sheet on top of it. Blender's axes:
// the desk runs along y, the writer sits at -x and reads towards +x.
const DESK = {
	top: 0.762,
	x: 1.5,
	piles: [-0.55, -0.88],
	step: 0.05,
	leaves: 0.003,
	sheet: { width: 0.21, height: 0.297 },
};
let seed = 11;
const jitter = (amount) => {
	seed = (seed * 9301 + 49297) % 233280;
	return (seed / 233280 - 0.5) * 2 * amount;
};
const sheets = pieces.map((piece, index) => {
	const perPile = Math.ceil(pieces.length / DESK.piles.length);
	const pile = Math.floor(index / perPile);
	const depth = index % perPile;
	const yaw = jitter(0.1);
	// Blender: x along the sheet's height (its head towards +x), y across it.
	const x = DESK.x + DESK.sheet.height / 2 + depth * DESK.step + jitter(0.006);
	const y = DESK.piles[pile] + jitter(0.02);
	// Each manuscript is a few leaves thick, so the pile steps by real paper.
	const z = DESK.top + (perPile - 1 - depth) * DESK.leaves;
	// The exposed head band, in the runtime's axes (Y up, depth along -Z), so
	// the browser can lay a target over exactly what a reader sees of it.
	const band = depth === 0 ? DESK.sheet.height : DESK.step;
	const corners = [
		[DESK.sheet.height / 2, -DESK.sheet.width / 2],
		[DESK.sheet.height / 2, DESK.sheet.width / 2],
		[DESK.sheet.height / 2 - band, DESK.sheet.width / 2],
		[DESK.sheet.height / 2 - band, -DESK.sheet.width / 2],
	].map(([along, across]) => [
		x + along * Math.cos(yaw) - across * Math.sin(yaw),
		z + DESK.leaves,
		-(y + along * Math.sin(yaw) + across * Math.cos(yaw)),
	]);
	return {
		slug: piece.slug,
		title: piece.title,
		x,
		y: z,
		z: -y,
		yaw,
		band: corners,
		...DESK.sheet,
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
	await writeFile(join(directory, 'sheets.json'), JSON.stringify(sheets, null, space) + end);
};
await publish(work, false);
// The browser imports these from the asset directory, so every run writes them
// there, not only --layout-only: a bake that moved the cabinet must not leave
// the runtime framing and picking the shape it had before.
await mkdir(out, { recursive: true });
await publish(out, true);
if (process.argv.includes('--layout-only')) process.exit(0);
await writeFile(join(work, 'books.json'), JSON.stringify(library));
for (const [index, book] of library.entries()) {
	const slot = slots[index];
	/* The jacket plane Blender builds is `width - .001` by `height - .003`, so
	   the drawing is authored at that exact aspect and nothing on it is
	   squeezed. Rasterised well above its nominal size: these jackets are the
	   only type in the room a visitor is meant to actually read, and neither the
	   bake nor the browser can invent detail the source does not have. */
	const svg = spineSvg(book, (slot.width - 0.001) / (slot.height - 0.003));
	const density =
		96 * Math.max(2, 320 / (ART_HEIGHT * ((slot.width - 0.001) / (slot.height - 0.003))));
	await sharp(Buffer.from(svg), { density })
		.png()
		.toFile(join(work, `book-${index}.png`));
	await sharp(Buffer.from(coverSvg(book)), { density: 192 })
		.png()
		.toFile(join(work, `cover-${index}.png`));
}
for (const [index, piece] of pieces.entries()) {
	// Four pixels per millimetre: a title read from the chair, not a texture.
	await sharp(Buffer.from(sheetSvg(piece, index)), { density: 96 * (1600 / SHEET.height) })
		.png()
		.toFile(join(work, `sheet-${index}.png`));
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
		...(process.argv.includes('--books-only') ? ['--books-only'] : []),
		...(process.argv.includes('--room-only') ? ['--room-only'] : []),
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
