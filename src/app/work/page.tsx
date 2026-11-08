import { Eyebrow } from '@/components/ui/eyebrow';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';
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
			{groups.map((group) => (
				<section
					key={group.id}
					id={group.id}
					aria-label={group.label}
					className="flex scroll-mt-6 flex-col gap-2"
				>
					<Eyebrow as="h2">{group.label}</Eyebrow>
					<ListRows>
						{group.reports.map((r) => (
							<ListRow
								key={r.id}
								href={workUrl(r)}
								date={r.data.period ?? 'Project'}
								title={r.data.title}
								description={r.data.summary}
							/>
						))}
					</ListRows>
				</section>
			))}
		</Page>
	);
}
