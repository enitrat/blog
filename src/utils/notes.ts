import 'server-only';
import { books } from '@/booksData';
import { getNotes } from '@/lib/content';

/**
 * The ISBNs of the volumes that carry a note.
 *
 * A note's filename is its book's ISBN-13. A typo would shelve a ribbon on a
 * book that cannot be opened, or quietly drop a written note — so mismatches
 * fail the build rather than the visitor's click.
 */
export function notedIsbns(): Set<string> {
	const shelved = new Set(books.map((book) => book.edition.isbn13));
	const notes = getNotes();
	for (const note of notes) {
		if (!shelved.has(note.id))
			throw new Error(`src/content/notes/${note.id}.md matches no book on the shelves.`);
	}
	return new Set(notes.map((note) => note.id));
}
