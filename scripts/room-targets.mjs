import { readFile } from 'node:fs/promises';

/** The room's bake groups. Each one is baked into, and shipped as, `<group>.glb`. */
export const ROOM_GROUPS = [
	'shell',
	'furniture',
	'objects',
	'moving',
	'spines',
	'books',
	'covers',
	'bookmark',
	'sheets',
];

/** Groups baked with the rest of the room hidden: surfaces the browser moves
 * must not keep the shadow of where they stood. */
export const ISOLATED_GROUPS = ['books', 'covers', 'bookmark', 'sheets'];

/** Groups compressed with Meshopt. Its quantization moves node origins, so only
 * the groups whose nodes the browser never turns, lifts, or places qualify. */
export const COMPRESSED_GROUPS = ['shell', 'furniture', 'objects'];

/** The most the room's GLBs may weigh together. */
export const GLB_BUDGET = 11_000_000;

/** A GLB's JSON chunk and binary chunk. */
export async function readGlb(path) {
	const glb = await readFile(path);
	const length = glb.readUInt32LE(12);
	return {
		gltf: JSON.parse(glb.subarray(20, 20 + length).toString()),
		bin: glb.subarray(20 + length + 8),
		bytes: glb.length,
	};
}

/** `--only a,b` selects groups in bake order; no flag selects all of them. */
export function roomTargets(args) {
	const only = args.indexOf('--only');
	if (only < 0) return ROOM_GROUPS;
	const requested = (args[only + 1] ?? '').split(',').filter(Boolean);
	const unknown = requested.filter((name) => !ROOM_GROUPS.includes(name));
	if (!requested.length || unknown.length)
		throw new Error(`--only takes a comma-separated list of: ${ROOM_GROUPS.join(', ')}`);
	return ROOM_GROUPS.filter((name) => requested.includes(name));
}
