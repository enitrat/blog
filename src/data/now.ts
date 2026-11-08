/**
 * What holds Mathieu's attention now. Written by hand, never generated.
 * One entry per item (keep it to 1-4): `{ what: 'Learning to sail', since: '2026-09' }`.
 * `since` is the month it started, 'YYYY-MM'. While the list is empty the Now tile is hidden.
 */
export interface NowItem {
	what: string;
	since: `${number}-${number}`;
}

export const now: NowItem[] = [];
