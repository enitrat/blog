import { ListRow, ListRows, Page, PageTitle } from '@/ds';

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
			<PageTitle lead="This address doesn't name anything here. Try one of these four pages:">
				Not found
			</PageTitle>
			<ListRows>
				{destinations.map((d) => (
					<ListRow key={d.href} {...d} />
				))}
			</ListRows>
		</Page>
	);
}
