import { describe, expect, test } from 'bun:test';
import { smartQuotes } from './typeset';

describe('smartQuotes', () => {
	test('apostrophes curl', () => {
		expect(smartQuotes("What's the Future for Junior Engineers?")).toBe(
			'What’s the Future for Junior Engineers?',
		);
		expect(smartQuotes("engineers' tools")).toBe('engineers’ tools');
	});

	test('quotes open and close', () => {
		expect(smartQuotes('the "software factory"')).toBe('the “software factory”');
		expect(smartQuotes("a 'quoted' word")).toBe('a ‘quoted’ word');
	});

	test('text without quotes is unchanged', () => {
		expect(smartQuotes('Using DSPy in Production')).toBe('Using DSPy in Production');
	});
});
