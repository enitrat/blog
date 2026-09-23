/** Inspect selected room assets, then verify the shipped browser scene. */
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { roomTargets } from './room-targets.mjs';

const args = process.argv.slice(2);
const targets = roomTargets(args);
const engine = args.includes('webkit') ? 'webkit' : 'chromium';

const run = (command, commandArgs) =>
	new Promise((resolve, reject) => {
		const child = spawn(command, commandArgs, { stdio: 'inherit' });
		child.on('error', reject);
		child.on('exit', (code) =>
			code === 0 ? resolve() : reject(new Error(`${command} exited ${code}`)),
		);
	});

for (const target of targets) {
	await run('bunx', [
		'@gltf-transform/cli',
		'inspect',
		join('src/assets/room', `${target}.glb`),
		'--format=md',
	]);
}

await run('bun', ['scripts/check-room.mjs', engine, '--shots', '--metrics']);
