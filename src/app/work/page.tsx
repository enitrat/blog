import { Eyebrow } from '@/components/ui/eyebrow';
import { Page, PageHeader } from '@/components/ui/page';
import { WorkLogo } from '@/components/work/work-logo';
import { WorkRow } from '@/components/work/work-row';
import { getReports, workUrl } from '@/lib/content';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Work',
	description: 'What I have built and worked on.',
	path: '/work/',
});

export default function WorkPage() {
	const reports = getReports();
	// Same two groups as the homepage index. Each section id is the fragment the homepage links
	// to, so `/work/#projects` lands on the group it names.
	const groups = [
		{ id: 'work', label: 'Work', reports: reports.filter((r) => r.data.kind === 'work') },
		{
			id: 'projects',
			label: 'Projects',
			reports: reports.filter((r) => r.data.kind === 'project'),
		},
	].filter((group) => group.reports.length > 0);
	return (
		<Page>
			<PageHeader>Work</PageHeader>
			<div className="flex flex-col gap-rhythm">
				{groups.map((group) => (
					<section
						key={group.id}
						id={group.id}
						aria-label={group.label}
						className="flex scroll-mt-anchor flex-col gap-2"
					>
						<Eyebrow as="h2">{group.label}</Eyebrow>
						<ul className="flex flex-col">
							{group.reports.map((r) => (
								<WorkRow
									key={r.id}
									href={workUrl(r)}
									logo={<WorkLogo id={r.id} title={r.data.title} />}
									title={r.data.title}
									role={r.data.role}
									period={r.data.period ?? 'Project'}
									summary={r.data.summary}
								/>
							))}
						</ul>
					</section>
				))}
			</div>
		</Page>
	);
}
