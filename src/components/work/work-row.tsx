import { ArrowRightIcon } from 'lucide-react';
import Link from 'next/link';
import type { ReactNode } from 'react';

/**
 * One work entry: logo, title, role, period and summary. Hover or keyboard focus tints the row,
 * grows the logo a little and slides the arrow in; both are CSS on the link's own states, so a
 * focused row wakes exactly as a hovered one does.
 */
export function WorkRow({
	href,
	logo,
	title,
	role,
	period,
	summary,
}: {
	href: string;
	logo: ReactNode;
	title: string;
	role?: string;
	period?: string;
	summary?: string;
}) {
	const meta = [role, period].filter(Boolean).join(' · ');
	return (
		<li className="border-t border-border first:border-t-0">
			<Link
				href={href}
				className="group -mx-3 flex items-start gap-4 rounded-row px-3 py-4 transition-colors hover:bg-muted"
			>
				<span className="flex flex-none transition-transform duration-200 ease-out group-hover:scale-[1.08] group-focus-visible:scale-[1.08]">
					{logo}
				</span>
				<span className="min-w-0 flex-1">
					<h3 className="type-subtitle text-balance text-foreground">{title}</h3>
					{meta && <span className="mt-0.5 block type-meta text-faint">{meta}</span>}
					{summary && (
						<span className="mt-1.5 block type-body text-pretty text-muted-foreground">
							{summary}
						</span>
					)}
				</span>
				<ArrowRightIcon
					aria-hidden
					className="mt-3 hidden size-4 flex-none -translate-x-1 text-faint opacity-0 transition-[opacity,translate] duration-200 ease-out group-hover:translate-x-0 group-hover:opacity-100 group-focus-visible:translate-x-0 group-focus-visible:opacity-100 sm:block"
				/>
			</Link>
		</li>
	);
}
