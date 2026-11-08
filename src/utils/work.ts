import 'server-only';
import { getReports, type Report } from '@/lib/content';

export const workUrl = (report: Pick<Report, 'id'>) => `/work/${report.id}/`;

export function getWorkIndex(): Report[] {
	return [...getReports()].sort((a, b) => a.data.order - b.data.order);
}
