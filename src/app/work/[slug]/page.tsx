import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { Prose } from '@/components/reading/prose';
import { Label, Link, ListRow, ListRows, Page, PageTitle } from '@/ds';
import { getReports } from '@/lib/content';
import { renderMarkdown } from '@/lib/markdown';
import { pageMeta } from '@/lib/meta';
import { getWorkIndex, workUrl } from '@/utils/work';

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
	const { tree } = await renderMarkdown(report.body, report.dir);
	const { title, summary, role, period, site } = report.data;
	const meta = [role, period].filter(Boolean).join(' · ');
	const siblings = getWorkIndex()
		.filter((r) => r.id !== report.id)
		.slice(0, 2);
	return (
		<Page width="measure" as="article">
			<PageTitle kicker={meta || undefined} lead={summary}>
				{title}
			</PageTitle>
			{site && (
				<p>
					<Link variant="external" href={site}>
						Visit website
					</Link>
				</p>
			)}

			<Prose tree={tree} />

			<footer className="grid gap-6">
				{siblings.length > 0 && (
					<nav aria-label="More work">
						<Label as="h2" space="sm">
							More work
						</Label>
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
					<Link variant="arrow" href="/work">
						All work
					</Link>
					<Link variant="arrow" href="/writing">
						Writing
					</Link>
				</p>
			</footer>
		</Page>
	);
}
