import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import sharp from 'sharp';
import { books } from '../src/booksData';

const DIR = 'src/assets/covers';
const CONCURRENCY = 4;
const MAX_HEIGHT = 600;
const MIN_SIDE = 50;

type Outcome = 'fetched' | 'skipped' | 'missing';

/** Open Library answers 404 when it has no cover; a placeholder pixel counts as missing too. */
async function download(isbn: string): Promise<Buffer | undefined> {
	const url = `https://covers.openlibrary.org/b/isbn/${isbn}-L.jpg?default=false`;
	for (let attempt = 0; ; attempt++) {
		try {
			const res = await fetch(url);
			if (res.status === 404) return undefined;
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			return Buffer.from(await res.arrayBuffer());
		} catch (err) {
			if (attempt >= 1) throw err;
		}
	}
}

async function fetchCover(isbn: string): Promise<Outcome> {
	const file = `${DIR}/${isbn}.jpg`;
	if (existsSync(file)) return 'skipped';
	const raw = await download(isbn);
	if (!raw) return 'missing';
	const { width = 0, height = 0 } = await sharp(raw).metadata();
	if (width < MIN_SIDE || height < MIN_SIDE) return 'missing';
	await sharp(raw)
		.resize({ height: MAX_HEIGHT, withoutEnlargement: true })
		.jpeg({ quality: 82, mozjpeg: true })
		.toFile(file);
	return 'fetched';
}

await mkdir(DIR, { recursive: true });
const queue = [...new Map(books.map((b) => [b.edition.isbn13, b])).values()];
const counts: Record<Outcome, number> = { fetched: 0, skipped: 0, missing: 0 };
const missing: string[] = [];
const failed: string[] = [];

await Promise.all(
	Array.from({ length: CONCURRENCY }, async () => {
		for (let book = queue.shift(); book; book = queue.shift()) {
			try {
				const outcome = await fetchCover(book.edition.isbn13);
				counts[outcome]++;
				if (outcome === 'missing') missing.push(`${book.title} (${book.edition.isbn13})`);
			} catch (err) {
				failed.push(`${book.title} (${book.edition.isbn13}): ${err}`);
			}
		}
	}),
);

console.log(`fetched ${counts.fetched}, skipped ${counts.skipped}, missing ${counts.missing}`);
for (const m of missing.sort()) console.log(`  missing: ${m}`);
for (const f of failed) console.log(`  failed: ${f}`);
if (failed.length) process.exit(1);
