/** Bake original room assets with Blender 4.5; the browser consumes only the exports. */
import { spawn } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import sharp from 'sharp';
import { books } from '../src/booksData.ts';
import { PLEIADE_HEX, pleiadeStyleFor } from '../src/utils/pleiade.ts';
import { englishPieces, notedIsbns } from './room-content.mjs';
import { ROOM_GROUPS, roomTargets } from './room-targets.mjs';
import { SHEET, sheetSvg } from './sheet-art.mjs';
import { ART_HEIGHT, coverSvg, spineSvg } from './spine-art.mjs';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const index = args.indexOf(name);
	return index < 0 ? fallback : args[index + 1];
};
const totalStarted = performance.now();
const timed = (label, started) =>
	console.log(`TIMING ${label} ${((performance.now() - started) / 1000).toFixed(2)}s`);
const targets = roomTargets(args);
const targetSet = new Set(targets);
const preview = args.includes('--preview');
const noPublish = args.includes('--no-publish');
// Checked-in assets are always production quality; tuning is for experiments.
if (!preview && !noPublish && (option('--samples') || option('--size')))
	throw new Error('--samples and --size need --no-publish or --preview.');
// A fresh directory per run, so concurrent bakes never share intermediates.
// Set ROOM_WORK only to open a preview at a known path.
const work = process.env.ROOM_WORK ?? (await mkdtemp(join(tmpdir(), 'living-room-')));
const out = resolve('src/assets/room');
const stage = join(work, 'publish');
await mkdir(work, { recursive: true });
await rm(stage, { recursive: true, force: true });
await mkdir(stage, { recursive: true });
// Only an annotated volume can be opened, so only its front cover is ever seen
// off the shelf. Baking the other fifty-odd spends the atlas on faces nobody
// can reach.
const noted = await notedIsbns();
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
// The writing lies on the desk, one manuscript per published piece.
const pieces = await englishPieces();
// Two fanned piles on the leather, newest on top and nearest the chair, each
// older sheet pushed a little further from the writer so its head -- where
// the title is written -- shows past the sheet on top of it. Blender's axes:
// the desk runs along y, the writer sits at -x and reads towards +x.
const DESK = {
	// The desk's centre and the underside of its top; room.py builds it here.
	centre: [1.72, -0.55, 0.725],
	top: 0.762, // the leather writing surface
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
// Tabs, so the checked-in copies match what `bun run lint` expects.
const manifests = { 'book-slots.json': slots, 'cabinet.json': CABINET, 'sheets.json': sheets };
// Blender alone needs the desk; the browser finds it through the sheets.
await writeFile(join(work, 'desk.json'), JSON.stringify(DESK));
for (const [name, value] of Object.entries(manifests)) {
	const text = `${JSON.stringify(value, null, '\t')}\n`;
	await writeFile(join(work, name), text);
	await writeFile(join(stage, name), text);
}

const sameFile = async (left, right) => {
	try {
		return (await readFile(left)).equals(await readFile(right));
	} catch {
		return false;
	}
};
/** The embedded images of a GLB, in glTF order. */
const glbImages = async (path) => {
	const glb = await readFile(path);
	const length = glb.readUInt32LE(12);
	const gltf = JSON.parse(glb.subarray(20, 20 + length).toString());
	const bin = glb.subarray(20 + length + 8);
	return (gltf.images ?? []).map((image) => {
		const view = gltf.bufferViews[image.bufferView];
		return bin.subarray(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength);
	});
};
// Copied only after every selected GLB validated; git is the rollback.
const install = async (names) => {
	for (const name of names) await copyFile(join(stage, name), join(out, name));
};

if (targets.length !== ROOM_GROUPS.length) {
	for (const name of ['book-slots.json', 'cabinet.json']) {
		if (!(await sameFile(join(stage, name), join(out, name)))) {
			throw new Error(`${name} changed. Run a full bake so geometry and manifests stay in sync.`);
		}
	}
}
await writeFile(join(work, 'books.json'), JSON.stringify(library));
const artworkStarted = performance.now();
// Groups baked in isolation never see the room's other artwork, so they skip it.
const fullArtwork =
	preview || targets.some((name) => !['books', 'covers', 'bookmark', 'sheets'].includes(name));
const raster = (svg, density, name) =>
	sharp(Buffer.from(svg), { density }).png().toFile(join(work, name));
await Promise.all([
	...library.flatMap((book, index) => {
		const slot = slots[index];
		/* The jacket plane Blender builds is `width - .001` by `height - .003`, so
		   the drawing is authored at that exact aspect and nothing on it is
		   squeezed. Rasterised well above its nominal size: these jackets are the
		   only type in the room a visitor is meant to actually read, and neither
		   the bake nor the browser can invent detail the source does not have. */
		const aspect = (slot.width - 0.001) / (slot.height - 0.003);
		return [
			fullArtwork &&
				raster(
					spineSvg(book, aspect),
					96 * Math.max(2, 320 / (ART_HEIGHT * aspect)),
					`book-${index}.png`,
				),
			book.noted &&
				(fullArtwork || targetSet.has('covers')) &&
				raster(coverSvg(book), 192, `cover-${index}.png`),
		];
	}),
	...pieces.map(
		(piece, index) =>
			(fullArtwork || targetSet.has('sheets')) &&
			// Four pixels per millimetre: a title read from the chair, not a texture.
			raster(sheetSvg(piece, index), 96 * (1600 / SHEET.height), `sheet-${index}.png`),
	),
	// Original geometric artwork, with no invented attribution or borrowed cover art.
	...[
		['print', 720, 840],
		['sleeve', 640, 640],
	].map(
		([name, width, height]) =>
			fullArtwork &&
			raster(
				`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 720 840"><rect width="720" height="840" fill="#d7cbb2"/><rect x="70" y="70" width="580" height="640" fill="#3d5358"/><circle cx="360" cy="390" r="240" fill="#ae5940"/><path d="M120 390h480M120 440h480M145 490h430M185 540h350M245 590h230" stroke="#d7cbb2" stroke-width="17"/><circle cx="360" cy="390" r="55" fill="#d7cbb2"/><circle cx="360" cy="390" r="15" fill="#3d5358"/></svg>`,
				72,
				`${name}.png`,
			),
	),
]);
timed('artwork', artworkStarted);
console.log(`Blender source and intermediate textures: ${work}`);
const run = (command, commandArgs, silent = false) =>
	new Promise((resolveRun, rejectRun) => {
		const output = [];
		const child = spawn(command, commandArgs, {
			stdio: silent ? ['ignore', 'pipe', 'pipe'] : 'inherit',
		});
		if (silent) {
			child.stdout.on('data', (chunk) => output.push(chunk));
			child.stderr.on('data', (chunk) => output.push(chunk));
		}
		child.on('error', (error) => rejectRun(error));
		child.on('exit', (code) => {
			if (code === 0) resolveRun();
			else rejectRun(new Error(`${command} exited ${code}\n${Buffer.concat(output).toString()}`));
		});
	});

const blenderStarted = performance.now();
const blender = process.env.BLENDER ?? 'blender';
try {
	await run(blender, [
		'--background',
		'--factory-startup',
		// Without it, a Python exception in room.py still exits 0.
		'--python-exit-code',
		'1',
		'--python',
		resolve('scripts/room.py'),
		'--',
		work,
		stage,
		option('--samples', '256'),
		option('--size', '2048'),
		'--only',
		targets.join(','),
		...(preview ? ['--preview'] : []),
		...(!fullArtwork ? ['--minimal-art'] : []),
	]);
} catch (error) {
	if (error.code === 'ENOENT') throw new Error(`Cannot find Blender at ${blender}. Set BLENDER.`);
	throw error;
}
timed('blender', blenderStarted);

if (preview) {
	timed('total', totalStarted);
	process.exit(0);
}

// Meshopt shrinks the geometry of the static groups. Its quantization moves
// node origins, so the groups the browser turns, lifts or places are left alone.
const compressStarted = performance.now();
await Promise.all(
	targets
		.filter((name) => ['shell', 'furniture', 'objects'].includes(name))
		.map((name) => {
			const glb = join(stage, `${name}.glb`);
			return run('bunx', ['@gltf-transform/cli@4.5.0', 'meshopt', glb, glb], true);
		}),
);
timed('meshopt', compressStarted);
const validationStarted = performance.now();
for (const name of targets) {
	await run('bunx', ['@gltf-transform/cli@4.5.0', 'validate', join(stage, `${name}.glb`)], true);
}
timed('validation', validationStarted);

if (noPublish) {
	// Say how far each staged atlas moved from the checked-in one, so a
	// refactor can show it changed nothing a visitor would see.
	for (const name of targets) {
		const [staged, current] = await Promise.all(
			[join(stage, `${name}.glb`), join(out, `${name}.glb`)].map(glbImages),
		);
		for (const [index, image] of staged.entries()) {
			const [a, b] = await Promise.all(
				[image, current[index]].map(
					(buffer) => buffer && sharp(buffer).raw().toBuffer({ resolveWithObject: true }),
				),
			);
			if (!b || a.info.width !== b.info.width || a.info.height !== b.info.height) {
				console.log(`DIFF ${name}[${index}] new image size or count`);
				continue;
			}
			let sum = 0;
			let changed = 0;
			for (let i = 0; i < a.data.length; i += 1) {
				const delta = Math.abs(a.data[i] - b.data[i]);
				sum += delta;
				if (delta > 16) changed += 1;
			}
			console.log(
				`DIFF ${name}[${index}] mean ${(sum / a.data.length).toFixed(2)}/255, ${((100 * changed) / a.data.length).toFixed(2)}% of channels off by >16`,
			);
		}
	}
	console.log(`Validated assets remain staged in ${stage}`);
	timed('total', totalStarted);
	process.exit(0);
}

const manifestNames =
	targets.length === ROOM_GROUPS.length
		? ['book-slots.json', 'cabinet.json', 'sheets.json']
		: targetSet.has('sheets')
			? ['sheets.json']
			: [];
await install([...targets.map((name) => `${name}.glb`), ...manifestNames]);
timed('total', totalStarted);
