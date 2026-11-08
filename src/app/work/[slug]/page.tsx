import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { Prose } from '@/components/reading/prose';
import { Eyebrow } from '@/components/ui/eyebrow';
import { Page } from '@/components/ui/page';
import { TextLink } from '@/components/ui/text-link';
import { WorkLogo } from '@/components/work/work-logo';
import { WorkRow } from '@/components/work/work-row';
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
			<header className="flex flex-col gap-5">
				<WorkLogo id={report.id} title={title} size="header" />
				<div className="flex flex-col gap-2">
					{meta && <Eyebrow as="p">{meta}</Eyebrow>}
					<h1 className="font-display text-4xl text-balance text-foreground sm:text-5xl">
						{title}
					</h1>
					{summary && (
						<p className="text-sm leading-relaxed text-pretty text-muted-foreground">{summary}</p>
					)}
				</div>
				{site && (
					<p>
						<TextLink href={site} icon="external">
							Visit website
						</TextLink>
					</p>
				)}
			</header>

			<Prose tree={tree} />

			<footer className="flex flex-col gap-6">
				{siblings.length > 0 && (
					<nav aria-label="More work" className="flex flex-col gap-2">
						<Eyebrow as="h2">More work</Eyebrow>
						<ul className="flex flex-col">
							{siblings.map((s) => (
								<WorkRow
									key={s.id}
									href={workUrl(s)}
									logo={<WorkLogo id={s.id} title={s.data.title} />}
									title={s.data.title}
									role={s.data.role}
									summary={s.data.summary}
								/>
							))}
						</ul>
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
