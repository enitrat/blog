import Image, { type StaticImageData } from 'next/image';
import Link from 'next/link';

/**
 * One essay: date, title and description, and the banner as a thumbnail when the post has one. A
 * row without a banner gives its text the room: no empty slot holds a place for a missing image.
 * Hover tints the row and slowly pushes the banner in.
 */
export function WritingRow({
	href,
	date,
	datetime,
	title,
	description,
	banner,
}: {
	href: string;
	date: string;
	datetime: string;
	title: string;
	description: string;
	banner?: StaticImageData;
}) {
	return (
		<li className="border-t border-border first:border-t-0">
			<Link
				href={href}
				className="group -mx-3 flex flex-col gap-3 rounded-row px-3 py-5 transition-colors hover:bg-muted sm:flex-row sm:items-center sm:gap-8"
			>
				<time dateTime={datetime} className="type-meta text-faint sm:w-24 sm:flex-none">
					{date}
				</time>
				<div className="min-w-0 flex-1">
					<h2 className="type-subtitle text-balance text-foreground">{title}</h2>
					<p className="mt-1.5 type-body text-pretty text-muted-foreground">{description}</p>
				</div>
				{banner && (
					<div className="aspect-[16/10] overflow-hidden rounded-media bg-muted shadow-card sm:aspect-auto sm:h-35 sm:w-56 sm:flex-none">
						<Image
							src={banner}
							alt=""
							sizes="(min-width: 40rem) 224px, 100vw"
							className="size-full object-cover transition-transform duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-[1.05]"
						/>
					</div>
				)}
			</Link>
		</li>
	);
}
