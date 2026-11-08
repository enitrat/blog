import type { Root } from 'hast';
import { Eyebrow } from '@/components/ui/eyebrow';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';
import { TextLink } from '@/components/ui/text-link';
import type { BlogData } from '@/lib/content';
import type { InShortItem } from '@/lib/in-short';
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
 * The essay page: a meta line, the title, the build-time "In short" block, then the prose.
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
	/** Items from `inShort(tree, hooks)`; sections with no hook are left out. */
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
		<Page width="text" as="article">
			<script
				type="application/ld+json"
				// biome-ignore lint/security/noDangerouslySetInnerHtml: JSON-LD of build-time frontmatter
				dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, '\\u003c') }}
			/>
			<PageHeader
				kicker={
					<Eyebrow
						as="p"
						className="flex-wrap gap-x-1.5 tabular-nums [&>:not(:first-child)]:before:me-1.5 [&>:not(:first-child)]:before:content-['·']"
					>
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
					</Eyebrow>
				}
			>
				{title}
			</PageHeader>

			{inShort.length > 0 && <InShort items={inShort} />}

			<Prose tree={tree} backLinks={inShort.length > 0} />

			<footer className="flex flex-col gap-8">
				{onward.length > 0 && (
					<nav aria-label={t.onward} className="flex flex-col gap-2">
						<Eyebrow as="h2">{t.onward}</Eyebrow>
						<ListRows>
							{onward.map((item) => (
								<ListRow key={item.href} href={item.href} title={item.title} />
							))}
						</ListRows>
					</nav>
				)}
				<p>
					<TextLink href="/writing">{t.all}</TextLink>
				</p>
			</footer>
		</Page>
	);
}
