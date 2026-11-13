import { readFileSync } from 'node:fs';
import { Glob } from 'bun';

/** Every source file the site ships, tests excluded. */
export const sourceFiles = [...new Glob('src/**/*.{ts,tsx}').scanSync('.')].filter(
	(file) => !file.endsWith('.test.ts'),
);

/** `file:line: text` for each source line matching `pattern`, in files `skip` lets through. */
export function offenders(pattern: RegExp, skip: (file: string) => boolean = () => false) {
	return sourceFiles
		.filter((file) => !skip(file))
		.flatMap((file) =>
			readFileSync(file, 'utf8')
				.split('\n')
				.flatMap((line, i) => (pattern.test(line) ? [`${file}:${i + 1}: ${line.trim()}`] : [])),
		);
}
