import { Eyebrow } from '@/components/ui/eyebrow';
import { IN_SHORT_LABEL, type InShortItem } from '@/lib/in-short';

/**
 * The build-time skim block: a numbered list of section titles, each with its first sentence and
 * a link to its `#id`. Its own id, `in-short`, is the target of the back links Prose adds.
 */
export function InShort({ items }: { items: InShortItem[] }) {
	return (
		<nav
			id="in-short"
			aria-label={IN_SHORT_LABEL}
			className="scroll-mt-6 border-b border-border pb-8"
		>
			<Eyebrow as="h2">{IN_SHORT_LABEL}</Eyebrow>
			<ol className="mt-4 flex flex-col">
				{items.map((item, i) => (
					<li key={item.id} className="grid grid-cols-[1.75rem_1fr]">
						<span aria-hidden className="pt-2.5 font-mono text-xs text-faint tabular-nums">
							{String(i + 1).padStart(2, '0')}
						</span>
						<a
							href={`#${item.id}`}
							className="group flex flex-col rounded-md py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
						>
							<strong className="font-semibold text-foreground underline decoration-transparent underline-offset-[22%] transition-colors group-hover:decoration-primary">
								{item.title}
							</strong>
							{item.hook && (
								<span className="text-pretty text-muted-foreground transition-colors group-hover:text-foreground">
									{item.hook}
								</span>
							)}
						</a>
					</li>
				))}
			</ol>
		</nav>
	);
}
