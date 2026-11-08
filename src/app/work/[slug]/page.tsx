import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { Prose } from '@/components/reading/prose';
import { Eyebrow } from '@/components/ui/eyebrow';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';
import { TextLink } from '@/components/ui/text-link';
import { getReports, workUrl } from '@/lib/content';
import { renderMarkdown } from '@/lib/markdown';
import { pageMeta } from '@/lib/meta';

type Params = { params: Promise<{ slug: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
	return getReports().map((report) => ({ slug: report.id }));
}

const reportOf = async ({ params }: Params) => {
	const { slug } = await params;
	return getReports().find((r) => r.id === slug) ?? notFound();
};

export async function generateMetadata(props: Params): Promise<Metadata> {
	const report = await reportOf(props);
	return pageMeta({
		title: report.data.title,
		description: report.data.summary,
		path: workUrl(report),
	});
}

export default async function ReportPage(props: Params) {
	const report = await reportOf(props);
	const tree = await renderMarkdown(report.body, report.dir);
	const { title, summary, role, period, site } = report.data;
	const meta = [role, period].filter(Boolean).join(' · ');
	const siblings = getReports()
		.filter((r) => r.id !== report.id)
		.slice(0, 2);
	return (
		<Page width="text" as="article">
			<PageHeader kicker={meta ? <Eyebrow as="p">{meta}</Eyebrow> : undefined} lead={summary}>
				{title}
			</PageHeader>
			{site && (
				<p>
					<TextLink href={site} icon="external">
						Visit website
					</TextLink>
				</p>
			)}

			<Prose tree={tree} />

			<footer className="flex flex-col gap-6">
				{siblings.length > 0 && (
					<nav aria-label="More work" className="flex flex-col gap-2">
						<Eyebrow as="h2">More work</Eyebrow>
						<ListRows>
							{siblings.map((s) => (
								<ListRow
									key={s.id}
									href={workUrl(s)}
									date={s.data.role}
									title={s.data.title}
									description={s.data.summary}
								/>
							))}
						</ListRows>
					</nav>
				)}
				<p className="flex gap-6">
					<TextLink href="/work" icon="arrow">
						All work
					</TextLink>
					<TextLink href="/writing" icon="arrow">
						Writing
					</TextLink>
				</p>
			</footer>
		</Page>
	);
}
