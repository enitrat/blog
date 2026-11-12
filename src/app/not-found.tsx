import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';

export const metadata = { title: 'Not found', description: 'This page does not exist.' };

const destinations = [
	{ href: '/', title: 'Home' },
	{ href: '/writing/', title: 'Writing' },
	{ href: '/reading/', title: 'Reading' },
	{ href: '/work/', title: 'Work' },
	{ href: '/about/', title: 'About' },
];

export default function NotFound() {
	return (
		<Page>
			<PageHeader lead="This address doesn’t name anything here. Try one of these pages:">
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
