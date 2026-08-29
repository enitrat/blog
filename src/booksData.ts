export type BookStatus = 'reading' | 'finished' | 'want-to-read';

export interface BookEdition {
	isbn13: string;
	pageCount: number;
	metadataSource: string;
}

export interface Book {
	title: string;
	author: string;
	edition: BookEdition;
	status: BookStatus;
	dateStarted: Date | null;
	dateFinished: Date | null;
	/** Out of 5; null while unrated. */
	rating: number | null;
	/** Base slug of the blog piece this book is the subject of. */
	writingSlug?: string;
}

export const books: Book[] = [
	{
		title: 'Le Roi de fer (Les Rois maudits, tome 1)',
		author: 'Maurice Druon',
		edition: {
			isbn13: '9782253011019',
			pageCount: 256,
			metadataSource:
				'https://www.hachette.fr/livre/le-roi-de-fer-les-rois-maudits-tome-1-9782253011019/',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2026-08-01'),
		rating: null,
	},
	{
		title: 'White Nights',
		author: 'Fyodor Dostoevsky',
		edition: {
			isbn13: '9780241252086',
			pageCount: 112,
			metadataSource:
				'https://www.penguin.co.uk/books/292382/white-nights-by-dostoyevsky-fyodor/9780241252086',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2026-08-01'),
		rating: null,
	},
	{
		title: 'East of Eden',
		author: 'John Steinbeck',
		edition: {
			isbn13: '9780143129486',
			pageCount: 608,
			metadataSource:
				'https://www.penguinrandomhouse.com/books/541274/east-of-eden-by-john-steinbeck/9780143129486/',
		},
		status: 'reading',
		dateStarted: new Date('2026-08-01'),
		dateFinished: null,
		rating: null,
	},
	{
		title: 'Le Prince',
		author: 'Niccolò Machiavelli',
		edition: {
			isbn13: '9782070392469',
			pageCount: 338,
			metadataSource: 'https://openlibrary.org/books/OL8838838M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2026-07-01'),
		rating: null,
	},
	{
		title: 'Magnifica humanitas',
		author: 'Pope Leo XIV',
		edition: {
			isbn13: '9781565487543',
			pageCount: 192,
			metadataSource:
				'https://www.dymocks.com.au/magnifica-humanitas-by-pope-leo-xiv-9781565487543',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2026-06-01'),
		rating: null,
	},
	{
		title: 'Notes from Underground',
		author: 'Fyodor Dostoevsky',
		edition: {
			isbn13: '9780099140115',
			pageCount: 176,
			metadataSource:
				'https://www.penguin.co.uk/books/354004/notes-from-underground-by-dostoevsky-fyodor/9780099140115',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2026-03-01'),
		rating: null,
	},
	{
		title: 'Principles for Dealing with the Changing World Order',
		author: 'Ray Dalio',
		edition: {
			isbn13: '9781982160272',
			pageCount: 576,
			metadataSource:
				'https://www.simonandschuster.com/books/Principles-for-Dealing-with-the-Changing-World-Order/Ray-Dalio/Principles/9781982160272',
		},
		status: 'reading',
		dateStarted: new Date('2025-11-09'),
		dateFinished: null,
		rating: null,
	},
	{
		title: 'Fooled by Randomness',
		author: 'Nassim Nicholas Taleb',
		edition: {
			isbn13: '9780812975215',
			pageCount: 316,
			metadataSource: 'https://openlibrary.org/books/OL3425559M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2026-02-01'),
		rating: null,
	},
	{
		title: 'Laughable Love',
		author: 'Milan Kundera',
		edition: {
			isbn13: '9780140096910',
			pageCount: 240,
			metadataSource: 'https://openlibrary.org/books/OL21363471M.json',
		},
		status: 'finished',
		dateStarted: new Date('2025-12-26'),
		dateFinished: new Date('2026-01-14'),
		rating: 4.2,
	},
	{
		title: 'The Unbearable Lightness of Being',
		author: 'Milan Kundera',
		edition: {
			isbn13: '9780060912529',
			pageCount: 314,
			metadataSource: 'https://openlibrary.org/books/OL32150408M.json',
		},
		status: 'finished',
		dateStarted: new Date('2025-11-09'),
		dateFinished: new Date('2025-11-25'),
		rating: 4.4,
	},
	{
		title: 'Kafka on the Shore',
		author: 'Haruki Murakami',
		edition: {
			isbn13: '9781400043668',
			pageCount: 448,
			metadataSource: 'https://openlibrary.org/books/OL8363016M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2025-06-15'),
		rating: 4.2,
	},
	{
		title: 'Norwegian Wood',
		author: 'Haruki Murakami',
		edition: {
			isbn13: '9780099448822',
			pageCount: 389,
			metadataSource: 'https://openlibrary.org/books/OL24240971M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2025-03-10'),
		rating: 3.9,
	},
	{
		title: 'The Brothers Karamazov',
		author: 'Fyodor Dostoevsky',
		edition: {
			isbn13: '9780374528379',
			pageCount: 824,
			metadataSource: 'https://openlibrary.org/books/OL30521039M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2025-01-20'),
		rating: 4,
	},
	{
		title: 'Brave New World',
		author: 'Aldous Huxley',
		edition: {
			isbn13: '9780099518471',
			pageCount: 288,
			metadataSource: 'https://openlibrary.org/books/OL27270734M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2024-09-15'),
		rating: 4.1,
	},
	{
		title: 'The Idiot',
		author: 'Fyodor Dostoevsky',
		edition: {
			isbn13: '9780140447927',
			pageCount: 732,
			metadataSource: 'https://openlibrary.org/books/OL3434283M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2024-05-20'),
		rating: 4.5,
	},
	{
		title: 'Crime and Punishment',
		author: 'Fyodor Dostoevsky',
		edition: {
			isbn13: '9780140449136',
			pageCount: 671,
			metadataSource: 'https://openlibrary.org/books/OL23140636M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2023-10-12'),
		rating: 3.7,
	},
	{
		title: 'The Red and the Black',
		author: 'Stendhal',
		edition: {
			isbn13: '9780812972078',
			pageCount: 560,
			metadataSource: 'https://openlibrary.org/books/OL8021000M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2023-06-08'),
		rating: 4.0,
	},
	{
		title: 'Froth on the Daydream',
		author: 'Boris Vian',
		edition: {
			isbn13: '9780140030754',
			pageCount: 188,
			metadataSource: 'https://openlibrary.org/books/OL17308727M.json',
		},
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2024-12-01'),
		rating: 3,
	},
];
