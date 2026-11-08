import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';

export const metadata = { title: 'Not found', description: 'This page does not exist.' };

const destinations = [
	{ href: '/', date: 'Home', title: 'Mathieu Saugier' },
	{ href: '/work/', date: 'Work', title: 'What I have worked on' },
	{ href: '/writing/', date: 'Writing', title: 'Writing' },
	{ href: '/about/', date: 'About', title: 'About' },
];

export default function NotFound() {
	return (
		<Page>
			<PageHeader lead="This address doesn't name anything here. Try one of these four pages:">
				Not found
			</PageHeader>
			<ListRows>
				{destinations.map((d) => (
					<ListRow key={d.href} {...d} />
				))}
			</ListRows>
		</Page>
	);
}
