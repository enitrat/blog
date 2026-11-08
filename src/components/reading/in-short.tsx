import { Label } from '@/ds';
import { IN_SHORT_LABEL, type InShortItem } from '@/utils/inShort';

/**
 * The build-time skim block: a numbered list of section titles, each with its first sentence and
 * a link to its `#id`. Its own id, `in-short`, is the target of the back links Prose adds.
 */
export function InShort({ items }: { items: InShortItem[] }) {
	return (
		<nav id="in-short" data-slot="in-short" className="in-short" aria-label={IN_SHORT_LABEL}>
			<Label as="h2">{IN_SHORT_LABEL}</Label>
			<ol>
				{items.map((item) => (
					<li key={item.id}>
						<a href={`#${item.id}`}>
							<strong>{item.title}</strong>
							{item.hook && <span>{item.hook}</span>}
						</a>
					</li>
				))}
			</ol>
		</nav>
	);
}
