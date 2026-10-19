import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const KIT = join(dirname(fileURLToPath(import.meta.url)), 'kit');

/** The kit as plain source (exports stripped), to inline into the standalone page. */
export const kitScript = () =>
	readFileSync(join(KIT, 'iso-kit.mjs'), 'utf8').replace(/^export /gm, '');
