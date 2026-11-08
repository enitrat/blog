import 'server-only';
import { ArrowUpRightIcon, BriefcaseIcon } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { getReports, workUrl } from '@/lib/content';
import { maybeImage } from '@/lib/media';
import { cn } from '@/lib/utils';
import { Card, CardHeader } from '../card';

/** "October 2025 – present" becomes "2025–present", "August 2023 – October 2025" "2023–25". */
function years(period: string) {
	const [from, to] = period.match(/\d{4}|present/g) ?? [];
	if (!from) return period;
	if (!to) return from;
	return `${from}–${to.slice(0, 2) === from.slice(0, 2) ? to.slice(2) : to}`;
}

const ROW = 'flex min-h-0 flex-1 border-t border-border first:border-t-0';
const LINK =
	'-mx-2 flex min-w-0 flex-1 items-center gap-2.5 rounded-media px-2 outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring';
const LOGO = 'size-5 flex-none rounded-md bg-card object-cover';

/** Jobs as rows (logo, company, role, years), then the projects in one row. */
export async function WorkTile({ className }: { className?: string }) {
	const reports = await Promise.all(
		getReports().map(async (r) => ({ ...r, logo: await maybeImage(`logos/${r.id}.png`) })),
	);
	if (reports.length === 0) return null;
	const jobs = reports.filter((r) => r.data.kind === 'work');
	const projects = reports.filter((r) => r.data.kind === 'project');
	return (
		<Card index={4} labelledBy="work-title" className={className}>
			<CardHeader id="work-title" icon={<BriefcaseIcon />} label="Work">
				<Link
					href="/work/"
					className="inline-flex items-center gap-0.5 rounded-sm outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
				>
					All work
					<ArrowUpRightIcon aria-hidden className="size-3.5" />
				</Link>
			</CardHeader>
			<ul className="mt-3 flex min-h-0 flex-1 flex-col">
				{jobs.map((r) => (
					<li key={r.id} className={ROW}>
						<Link href={workUrl(r)} className={LINK}>
							{r.logo && <Image src={r.logo} alt="" sizes="20px" className={LOGO} />}
							<h3 className="flex-none text-sm font-semibold">{r.data.title}</h3>
							<span className="min-w-0 flex-1 truncate text-[0.8125rem] text-faint">
								{r.data.role}
							</span>
							{r.data.period && (
								<span className="flex-none font-mono text-xs tracking-[-0.01em] text-faint tabular-nums">
									{years(r.data.period)}
								</span>
							)}
						</Link>
					</li>
				))}
				{projects.length > 0 && (
					<li className={ROW}>
						<Link href="/work/" className={LINK}>
							<span className="flex flex-none">
								{projects.map(
									(r, i) =>
										r.logo && (
											<Image
												key={r.id}
												src={r.logo}
												alt=""
												sizes="20px"
												className={cn(LOGO, 'ring-2 ring-card', i > 0 && '-ml-1.5')}
											/>
										),
								)}
							</span>
							<h3 className="flex-none text-sm font-semibold">Projects</h3>
							<span className="min-w-0 flex-1 truncate text-[0.8125rem] text-faint">
								{projects.map((r) => r.data.title).join(', ')}
							</span>
						</Link>
					</li>
				)}
			</ul>
		</Card>
	);
}
