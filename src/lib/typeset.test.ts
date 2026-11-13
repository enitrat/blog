import { describe, expect, test } from 'bun:test';
import { bindNumbers, smartQuotes, typeset } from './typeset';

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

describe('bindNumbers', () => {
	test('a number stays with the word before it', () => {
		expect(bindNumbers('Under the hood of Cairo 1.0: Exploring Sierra, Part 3')).toBe(
			'Under the hood of Cairo 1.0: Exploring Sierra, Part 3',
		);
	});

	test('text without numbers is unchanged', () => {
		expect(bindNumbers('Using DSPy in Production')).toBe('Using DSPy in Production');
	});
});

test('typeset curls quotes and binds numbers', () => {
	expect(typeset('Cairo 1.0\'s "Sierra"')).toBe('Cairo 1.0’s “Sierra”');
});
