export type PleiadeColor =
	| 'green'
	| 'violet'
	| 'corinthe'
	| 'red'
	| 'blue'
	| 'emerald'
	| 'havane'
	| 'grey';

/** The eight Pléiade leather colours. Canonical: the CSS spines and the
    canvas-painted 3D spines must not drift apart. */
export const PLEIADE_HEX: Record<PleiadeColor, string> = {
	green: '#416653',
	violet: '#574363',
	corinthe: '#70483f',
	red: '#813b34',
	blue: '#315a78',
	emerald: '#2f6652',
	havane: '#6a4a35',
	grey: '#54534f',
};

export interface PleiadeAuthorStyle {
	label: string;
	color: PleiadeColor;
}

export const PLEIADE_AUTHOR_COLORS: Record<string, PleiadeAuthorStyle> = {
	'Aldous Huxley': { label: 'Aldous Huxley', color: 'red' },
	'Boris Vian': { label: 'Boris Vian', color: 'corinthe' },
	'Fyodor Dostoevsky': { label: 'Dostoïevski', color: 'emerald' },
	'Haruki Murakami': { label: 'Murakami', color: 'green' },
	'John Steinbeck': { label: 'John Steinbeck', color: 'havane' },
	'Maurice Druon': { label: 'Maurice Druon', color: 'havane' },
	'Milan Kundera': { label: 'Kundera', color: 'violet' },
	'Nassim Nicholas Taleb': { label: 'Taleb', color: 'blue' },
	'Niccolò Machiavelli': { label: 'Machiavel', color: 'corinthe' },
	'Pope Leo XIV': { label: 'Léon XIV', color: 'grey' },
	'Ray Dalio': { label: 'Ray Dalio', color: 'havane' },
	Stendhal: { label: 'Stendhal', color: 'emerald' },
	'George R. R. Martin': { label: 'George R. R. Martin', color: 'violet' },
	'Alexandre Dumas': { label: 'Alexandre Dumas', color: 'green' },
	'Frank Herbert': { label: 'Frank Herbert', color: 'corinthe' },
	'Isaac Asimov': { label: 'Isaac Asimov', color: 'red' },
	'Alain Damasio': { label: 'Alain Damasio', color: 'blue' },
	'Albert Camus': { label: 'Albert Camus', color: 'emerald' },
	'Franz Kafka': { label: 'Franz Kafka', color: 'havane' },
	'J. R. R. Tolkien': { label: 'J. R. R. Tolkien', color: 'grey' },
	'Honoré de Balzac': { label: 'Honoré de Balzac', color: 'green' },
	'Émile Zola': { label: 'Émile Zola', color: 'emerald' },
	'Wajdi Mouawad': { label: 'Wajdi Mouawad', color: 'corinthe' },
	'George Orwell': { label: 'George Orwell', color: 'blue' },
	'J. K. Rowling': { label: 'J. K. Rowling', color: 'violet' },
	'Liu Cixin': { label: 'Liu Cixin', color: 'red' },
	'Suzanne Collins': { label: 'Suzanne Collins', color: 'grey' },
	'Stefan Zweig': { label: 'Stefan Zweig', color: 'corinthe' },
	'Pierre Lemaitre': { label: 'Pierre Lemaitre', color: 'red' },
	Vercors: { label: 'Vercors', color: 'grey' },
	'Stephen King': { label: 'Stephen King', color: 'violet' },
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
	const style = PLEIADE_AUTHOR_COLORS[author];
	if (!style) throw new Error(`Missing Pléiade spine style for author: ${author}`);
	return style;
};

/** Where this book sits between the thinnest and the thickest on the shelves, 0..1. */
export const thicknessRatioFor = (pageCount: number, pageRange: PageRange): number =>
	pageRange.max <= pageRange.min
		? 0.5
		: (pageCount - pageRange.min) / (pageRange.max - pageRange.min);

export const spineWidthFor = (
	pageCount: number,
	pageRange: PageRange,
	variant: 'showcase' | 'extended',
): number => {
	const minWidth = variant === 'extended' ? 58 : 42;
	const widthRange = variant === 'extended' ? 60 : 48;
	return Math.round(minWidth + thicknessRatioFor(pageCount, pageRange) * widthRange);
};
