import { Label, ListRow, ListRows, Page, PageTitle } from '@/ds';
import { pageMeta } from '@/lib/meta';
import { getWorkIndex, workUrl } from '@/utils/work';

export const metadata = pageMeta({
	title: 'Work',
	description: 'What I have built and worked on.',
	path: '/work/',
});

export default function WorkPage() {
	const reports = getWorkIndex();
	// Same two groups as the homepage index. Each heading id is the fragment the homepage links
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
			<PageTitle>Work</PageTitle>
			{groups.map((group) => (
				<section key={group.id} aria-labelledby={group.id}>
					<Label as="h2" id={group.id} space="sm">
						{group.label}
					</Label>
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
