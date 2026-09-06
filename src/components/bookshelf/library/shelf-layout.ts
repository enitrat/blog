/**
 * Split an ordered run of spines into shelves of roughly equal *width*.
 *
 * Balancing by count leaves a shelf visibly short, because a spine is as thick
 * as its book. So binary-search the narrowest shelf every spine still fits
 * into, then pack greedily to that width — the partition with the least width
 * left over on any one shelf.
 */
export function packShelves<T>(items: T[], spanOf: (item: T) => number, rows: number): T[][] {
	if (rows < 1 || items.length === 0) return [];
	const spans = items.map(spanOf);
	const total = spans.reduce((sum, value) => sum + value, 0);

	const shelvesNeeded = (limit: number) => {
		let used = 1;
		let filled = 0;
		for (const span of spans) {
			if (filled + span > limit + EPSILON) {
				used++;
				filled = 0;
			}
			filled += span;
		}
		return used;
	};

	let low = Math.max(Math.max(...spans), total / rows);
	let high = total;
	for (let step = 0; step < 40; step++) {
		const mid = (low + high) / 2;
		if (shelvesNeeded(mid) <= rows) high = mid;
		else low = mid;
	}

	const shelves: T[][] = [];
	let current: T[] = [];
	let filled = 0;
	for (const [index, item] of items.entries()) {
		// Break when the shelf is full, or early when the items left would not
		// otherwise reach every remaining shelf — no stub row at the bottom.
		const full = current.length > 0 && filled + spans[index] > high + EPSILON;
		const rationing = current.length > 0 && items.length - index < rows - shelves.length;
		if (full || rationing) {
			shelves.push(current);
			current = [];
			filled = 0;
		}
		current.push(item);
		filled += spans[index];
	}
	if (current.length > 0) shelves.push(current);
	return shelves;
}

const EPSILON = 1e-9;
