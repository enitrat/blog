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
