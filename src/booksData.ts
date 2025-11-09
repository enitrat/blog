export const books = [
	// Currently Reading
	{
		title: "The Unbearable Lightness of Being",
		author: "Milan Kundera",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780061148521-M.jpg",
		status: "reading" as const,
		dateStarted: new Date("2025-11-09"),
		dateFinished: null,
		rating: null
	},
	{
		title: "Fooled by Randomness",
		author: "Nassim Nicholas Taleb",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780812975215-M.jpg",
		status: "reading" as const,
		dateStarted: new Date("2025-11-09"),
		dateFinished: null,
		rating: null
	},
	{
		title: "Principles for Dealing with the Changing World Order",
		author: "Ray Dalio",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9781982160272-M.jpg",
		status: "reading" as const,
		dateStarted: new Date("2025-11-09"),
		dateFinished: null,
		rating: null
	},
	// Finished Reading
	{
		title: "Kafka on the Shore",
		author: "Haruki Murakami",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9781400079278-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2025-06-15"),
		dateStarted: null,
		rating: 4.2
	},
	{
		title: "Norwegian Wood",
		author: "Haruki Murakami",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780375704024-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2025-03-10"),
		dateStarted: null,
		rating: 3.9
	},
	{
		title: "The Brothers Karamazov",
		author: "Fyodor Dostoevsky",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780374528379-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2025-01-20"),
		dateStarted: null,
		rating: 4
	},
	{
		title: "Brave New World",
		author: "Aldous Huxley",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780060850524-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2024-09-15"),
		dateStarted: null,
		rating: 4.1
	},
	{
		title: "The Idiot",
		author: "Fyodor Dostoevsky",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780375702242-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2024-05-20"),
		dateStarted: null,
		rating: 4.5
	},
	{
		title: "Crime and Punishment",
		author: "Fyodor Dostoevsky",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780143058144-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2023-10-12"),
		dateStarted: null,
		rating: 3.7
	},
	{
		title: "The Red and the Black",
		author: "Stendhal",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9780140447644-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2023-06-08"),
		dateStarted: null,
		rating: 4.0
	},
	{
		title: "Froth on the Daydream",
		author: "Boris Vian",
		coverUrl: "https://covers.openlibrary.org/b/isbn/9781841959016-M.jpg",
		status: "finished" as const,
		dateFinished: new Date("2024-12-01"),
		dateStarted: null,
		rating: 3
	},
];

export type Book = typeof books[number];
