import { type CollectionEntry, getCollection } from 'astro:content';

type Report = CollectionEntry<'work'>;

export const workUrl = (report: Pick<Report, 'id'>) => `/work/${report.id}/`;

export async function getWorkIndex(): Promise<Report[]> {
	const reports = await getCollection('work');
	return reports.sort((a, b) => a.data.order - b.data.order);
}
