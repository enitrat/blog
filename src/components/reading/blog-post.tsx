import type { Root } from 'hast';
import { Label, Link, ListRow, ListRows, Page, PageTitle } from '@/ds';
import type { BlogData } from '@/lib/content';
import type { InShortItem } from '@/utils/inShort';
import { InShort } from './in-short';
import { Prose } from './prose';

type Adjacent = { href: string; title: string };

const t = {
	min: 'min read',
	updated: 'Updated',
	onward: 'Keep reading',
	all: 'All writing',
};

const formatDate = (date: Date) =>
	date.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: 'numeric' });

/**
 * The essay page: a mono meta line, the title, the build-time "In short" block, then the prose.
 * No banner, no sidebar, nothing sticky. Onward links close it.
 */
export function BlogPost({
	data: { title, description, pubDate, updatedDate },
	tree,
	inShort = [],
	minutes,
	newer,
	older,
	related = [],
}: {
	data: BlogData;
	/** The rendered markdown. */
	tree: Root;
	/** Items from `inShort(entry, headings)`; sections with no hook are left out. */
	inShort?: InShortItem[];
	/** Estimated reading time. */
	minutes?: number;
	/** Chronological neighbours; `related` replaces them when present. */
	newer?: Adjacent;
	older?: Adjacent;
	related?: Adjacent[];
}) {
	const onward =
		related.length > 0
			? related.slice(0, 2)
			: [...(older ? [older] : []), ...(newer ? [newer] : [])];
	const jsonLd = {
		'@context': 'https://schema.org',
		'@type': 'BlogPosting',
		headline: title,
		description,
		datePublished: pubDate.toISOString(),
		dateModified: (updatedDate ?? pubDate).toISOString(),
	};
	return (
		<Page width="measure" as="article">
			<script
				type="application/ld+json"
				// biome-ignore lint/security/noDangerouslySetInnerHtml: JSON-LD of build-time frontmatter
				dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, '\\u003c') }}
			/>
			<PageTitle
				kicker={
					<Label separated>
						<span>Mathieu Saugier</span>
						<time dateTime={pubDate.toISOString()}>{formatDate(pubDate)}</time>
						{minutes && (
							<span>
								{minutes} {t.min}
							</span>
						)}
						{updatedDate && (
							<span>
								{t.updated}{' '}
								<time dateTime={updatedDate.toISOString()}>{formatDate(updatedDate)}</time>
							</span>
						)}
					</Label>
				}
			>
				{title}
			</PageTitle>

			{inShort.length > 0 && <InShort items={inShort} />}

			<Prose tree={tree} backLinks={inShort.length > 0} />

			<footer className="grid gap-8">
				{onward.length > 0 && (
					<nav aria-labelledby="onward">
						<h2 id="onward" className="mb-3 text-ui font-medium text-muted-foreground">
							{t.onward}
						</h2>
						<ListRows>
							{onward.map((item) => (
								<ListRow key={item.href} href={item.href} title={item.title} />
							))}
						</ListRows>
					</nav>
				)}
				<p className="text-ui">
					<Link href="/writing" variant="quiet">
						{t.all}
					</Link>
				</p>
			</footer>
		</Page>
	);
}
