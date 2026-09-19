import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

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

const LEGACY_TARGETS = new Map([
	['--room-only', ROOM_GROUPS.filter((name) => !['books', 'covers', 'bookmark'].includes(name))],
	['--books-only', ['books', 'covers']],
	['--spines-only', ['spines']],
]);

export function roomTargets(args) {
	const only = args.indexOf('--only');
	const legacy = [...LEGACY_TARGETS].filter(([flag]) => args.includes(flag));
	if (only >= 0 && legacy.length)
		throw new Error('Use --only or a legacy partial-bake flag, not both.');
	if (legacy.length > 1) throw new Error('Use only one partial-bake flag.');

	let requested = ROOM_GROUPS;
	if (only >= 0) {
		const value = args[only + 1];
		if (!value || value.startsWith('--'))
			throw new Error('--only requires a comma-separated group list.');
		requested = value.split(',').filter(Boolean);
	} else if (legacy.length) {
		requested = legacy[0][1];
	}

	const unknown = requested.filter((name) => !ROOM_GROUPS.includes(name));
	if (unknown.length) throw new Error(`Unknown room group: ${unknown.join(', ')}`);
	const selected = new Set(requested);
	if (!selected.size) throw new Error('Select at least one room group.');
	return ROOM_GROUPS.filter((name) => selected.has(name));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	assert.deepEqual(roomTargets([]), ROOM_GROUPS);
	assert.deepEqual(roomTargets(['--only', 'sheets,covers']), ['covers', 'sheets']);
	assert.deepEqual(roomTargets(['--books-only']), ['books', 'covers']);
	assert.throws(() => roomTargets(['--only', 'nope']), /Unknown room group/);
	assert.throws(() => roomTargets(['--only', 'sheets', '--room-only']), /not both/);
	console.log('room target checks passed');
}
