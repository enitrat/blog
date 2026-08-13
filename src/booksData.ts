export type BookStatus = 'reading' | 'finished' | 'want-to-read';

export interface Book {
	title: string;
	author: string;
	status: BookStatus;
	dateStarted: Date | null;
	dateFinished: Date | null;
	/** Out of 5; null while unrated. */
	rating: number | null;
}

export const books: Book[] = [
	{
		title: 'Principles for Dealing with the Changing World Order',
		author: 'Ray Dalio',
		status: 'reading',
		dateStarted: new Date('2025-11-09'),
		dateFinished: null,
		rating: null,
	},
	{
		// TODO(msaug): confirm the real dates — the original entry said started
		// 2026-02-29 (not a real date) and finished with no finish date.
		title: 'Fooled by Randomness',
		author: 'Nassim Nicholas Taleb',
		status: 'finished',
		dateStarted: new Date('2026-03-01'),
		dateFinished: null,
		rating: null,
	},
	{
		title: 'Laughable Love',
		author: 'Milan Kundera',
		status: 'finished',
		dateStarted: new Date('2025-12-26'),
		dateFinished: new Date('2026-01-14'),
		rating: 4.2,
	},
	{
		title: 'The Unbearable Lightness of Being',
		author: 'Milan Kundera',
		status: 'finished',
		dateStarted: new Date('2025-11-09'),
		dateFinished: new Date('2025-11-25'),
		rating: 4.4,
	},
	{
		title: 'Kafka on the Shore',
		author: 'Haruki Murakami',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2025-06-15'),
		rating: 4.2,
	},
	{
		title: 'Norwegian Wood',
		author: 'Haruki Murakami',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2025-03-10'),
		rating: 3.9,
	},
	{
		title: 'The Brothers Karamazov',
		author: 'Fyodor Dostoevsky',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2025-01-20'),
		rating: 4,
	},
	{
		title: 'Brave New World',
		author: 'Aldous Huxley',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2024-09-15'),
		rating: 4.1,
	},
	{
		title: 'The Idiot',
		author: 'Fyodor Dostoevsky',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2024-05-20'),
		rating: 4.5,
	},
	{
		title: 'Crime and Punishment',
		author: 'Fyodor Dostoevsky',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2023-10-12'),
		rating: 3.7,
	},
	{
		title: 'The Red and the Black',
		author: 'Stendhal',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2023-06-08'),
		rating: 4.0,
	},
	{
		title: 'Froth on the Daydream',
		author: 'Boris Vian',
		status: 'finished',
		dateStarted: null,
		dateFinished: new Date('2024-12-01'),
		rating: 3,
	},
];
