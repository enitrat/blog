import type { Book } from '../booksData';

const MONTH_YEAR = new Intl.DateTimeFormat('en-GB', { month: 'long', year: 'numeric' });

/** The reading status, in one line. */
export function readingLine(book: Book): string {
	if (book.status === 'reading') return 'Reading now';
	if (book.status === 'want-to-read') return 'On the reading list';
	if (book.readingPeriod === 'before-2020') return 'Read before 2020';
	return book.dateFinished ? `Finished ${MONTH_YEAR.format(book.dateFinished)}` : 'Finished';
}

/**
 * Everything the archive knows about this book, in reading order. The shelf
 * slip and the record page inside an opened book print the same line.
 */
export function recordFacts(book: Book): string {
	return [
		`${book.edition.pageCount} pages`,
		readingLine(book),
		book.rating === null ? undefined : `rated ${book.rating} of 5`,
	]
		.filter(Boolean)
		.join(' · ');
}
