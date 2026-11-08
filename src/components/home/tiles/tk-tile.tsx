import { Tile } from '@/components/ui/tile';

/** A visible placeholder for a card Mathieu has yet to fill. Search the source for `[TK]`. */
export function TkTile({ what }: { what: string }) {
	return (
		<Tile size="compact" label={`[TK] Placeholder — ${what}`} labelAs="h2">
			<div className="size-full rounded-md border border-dashed border-gray-200 dark:border-white/10" />
		</Tile>
	);
}
