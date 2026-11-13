/**
 * Mathieu's lifeline, oldest first, shown in home's sidebar. Written by hand, in his words.
 * The gap before each entry grows with the years since the one before, so keep `year` exact.
 */
export interface LifelineEntry {
	year: number;
	text: string;
}

export const lifeline: LifelineEntry[] = [
	{ year: 2001, text: 'born' },
	{ year: 2012, text: 'market making linen cloth in the world of warcraft auction house' },
	{ year: 2017, text: 'scare my mom mining ethereum on the family computer' },
	{ year: 2021, text: 'learn how to code to do market making on NFTs' },
	{ year: 2024, text: 'graduate with a master’s thesis on making Ethereum ZK-Provable' },
	{ year: 2026, text: 'reinvent my job every quarter as AI advances' },
];
