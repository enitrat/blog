'use client';

import { Accordion as AccordionPrimitive } from '@base-ui/react/accordion';
import { PlusIcon } from 'lucide-react';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

function Accordion({ className, ...props }: AccordionPrimitive.Root.Props) {
	return (
		<AccordionPrimitive.Root
			data-slot="accordion"
			className={cn('flex flex-col', className)}
			{...props}
		/>
	);
}

/** One hairline-separated section, like the rows of every inner page. */
function AccordionItem({ className, ...props }: AccordionPrimitive.Item.Props) {
	return (
		<AccordionPrimitive.Item
			data-slot="accordion-item"
			className={cn('border-t border-border last:border-b', className)}
			{...props}
		/>
	);
}

/** The row that opens a section: a mono number, the title, a one-line summary, and a plus. */
function AccordionTrigger({
	number,
	summary,
	className,
	children,
	...props
}: AccordionPrimitive.Trigger.Props & { number?: string; summary?: ReactNode }) {
	return (
		<AccordionPrimitive.Header className="flex">
			<AccordionPrimitive.Trigger
				data-slot="accordion-trigger"
				className={cn(
					'group/trigger -mx-3 flex flex-1 cursor-pointer items-baseline gap-4 rounded-row px-3 py-5 text-start outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring',
					className,
				)}
				{...props}
			>
				{number && <span className="w-6 flex-none type-meta text-faint">{number}</span>}
				<span className="type-title text-foreground">{children}</span>
				{summary && (
					<span className="ms-auto hidden truncate type-caption text-faint sm:block">
						{summary}
					</span>
				)}
				<PlusIcon
					aria-hidden
					className={cn(
						'size-4 flex-none self-center text-faint transition-transform duration-200 ease-out group-hover/trigger:text-foreground group-data-panel-open/trigger:rotate-45 motion-reduce:transition-none',
						!summary && 'ms-auto',
					)}
				/>
			</AccordionPrimitive.Trigger>
		</AccordionPrimitive.Header>
	);
}

/** Opens to its measured height; under reduced motion it opens in place. */
function AccordionPanel({ className, children, ...props }: AccordionPrimitive.Panel.Props) {
	return (
		<AccordionPrimitive.Panel
			data-slot="accordion-panel"
			className="h-(--accordion-panel-height) overflow-hidden transition-[height] duration-250 ease-out data-ending-style:h-0 data-starting-style:h-0 motion-reduce:transition-none"
			{...props}
		>
			<div className={cn('pt-1 pb-10', className)}>{children}</div>
		</AccordionPrimitive.Panel>
	);
}

export { Accordion, AccordionItem, AccordionPanel, AccordionTrigger };
