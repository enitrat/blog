import 'server-only';
import { BookOpenIcon } from 'lucide-react';
import Link from 'next/link';
import { books } from '@/booksData';
import { Card, CardHeader } from '@/components/ui/card';
import { findImage } from '@/lib/media';
import { BookStack } from './book-stack';

const stamp = (d: Date | null) => d?.getTime() ?? 0;
const short = (t: string) => t.replace(/\s*\(.*\)\s*$/, '');

/**
 * The books being read now as a pile of covers; hover fans them. The title link covers the card.
 * With no current book it shows the last one finished.
 */
export async function ReadingTile({ className }: { className?: string }) {
	const current = books
		.filter((b) => b.status === 'reading')
		.sort((a, b) => stamp(b.dateStarted) - stamp(a.dateStarted));
	const last = books
		.filter((b) => b.status === 'finished')
		.sort((a, b) => stamp(b.dateFinished) - stamp(a.dateFinished))[0];
	const shown = (current.length ? current : last ? [last] : []).slice(0, 3);
	if (shown.length === 0) return null;
	const covers = await Promise.all(
		shown.map(async (book) => ({
			id: book.edition.isbn13,
			cover: await findImage(`covers/${book.edition.isbn13}`),
		})),
	);
	const front = shown[0];
	return (
		<Card index={3} lift labelledBy="reading-title" className={className}>
			<CardHeader
				id="reading-title"
				icon={<BookOpenIcon />}
				label={current.length ? 'Reading' : 'Last read'}
			>
				{current.length > 1 && (
					<span className="font-mono text-xs tracking-[-0.01em] whitespace-nowrap">
						+{current.length - 1} more
					</span>
				)}
			</CardHeader>
			<BookStack books={covers} />
			<div className="min-w-0 pt-3">
				<h3 className="truncate text-sm leading-[1.25] font-semibold">
					<Link
						href="/reading/"
						className="rounded-sm outline-none after:absolute after:inset-0 focus-visible:ring-2 focus-visible:ring-ring"
					>
						{short(front.title)}
					</Link>
				</h3>
				<p className="truncate text-meta text-faint">{front.author}</p>
			</div>
		</Card>
	);
}
