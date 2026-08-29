export type PleiadePeriod =
	| 'antiquity'
	| 'medieval'
	| 'sixteenth'
	| 'seventeenth'
	| 'eighteenth'
	| 'nineteenth'
	| 'modern'
	| 'spiritual';

export interface PleiadeAuthorStyle {
	label: string;
	period: PleiadePeriod;
}

const authorStyles: Record<string, PleiadeAuthorStyle> = {
	'Aldous Huxley': { label: 'Aldous Huxley', period: 'seventeenth' },
	'Boris Vian': { label: 'Boris Vian', period: 'sixteenth' },
	'Fyodor Dostoevsky': { label: 'Dostoïevski', period: 'nineteenth' },
	'Haruki Murakami': { label: 'Murakami', period: 'antiquity' },
	'John Steinbeck': { label: 'John Steinbeck', period: 'modern' },
	'Maurice Druon': { label: 'Maurice Druon', period: 'modern' },
	'Milan Kundera': { label: 'Kundera', period: 'medieval' },
	'Nassim Nicholas Taleb': { label: 'Taleb', period: 'eighteenth' },
	'Niccolò Machiavelli': { label: 'Machiavel', period: 'sixteenth' },
	'Pope Leo XIV': { label: 'Léon XIV', period: 'spiritual' },
	'Ray Dalio': { label: 'Ray Dalio', period: 'modern' },
	Stendhal: { label: 'Stendhal', period: 'nineteenth' },
};

export interface PageRange {
	min: number;
	max: number;
}

export const pageRangeFor = (books: { edition: { pageCount: number } }[]): PageRange => {
	if (books.length === 0) return { min: 0, max: 0 };
	const counts = books.map((book) => book.edition.pageCount);
	return { min: Math.min(...counts), max: Math.max(...counts) };
};

export const pleiadeStyleFor = (author: string): PleiadeAuthorStyle => {
	const style = authorStyles[author];
	if (!style) throw new Error(`Missing Pléiade spine style for author: ${author}`);
	return style;
};

export const spineWidthFor = (
	pageCount: number,
	pageRange: PageRange,
	variant: 'showcase' | 'extended',
): number => {
	const minWidth = variant === 'extended' ? 58 : 42;
	const widthRange = variant === 'extended' ? 60 : 48;
	if (pageRange.max <= pageRange.min) return minWidth + Math.round(widthRange / 2);
	return Math.round(
		minWidth + ((pageCount - pageRange.min) / (pageRange.max - pageRange.min)) * widthRange,
	);
};
