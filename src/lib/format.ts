/**
 * How the site writes dates, counts and ratings: one format per kind, so a date reads the same on
 * every page. Dates are calendar dates stored at UTC midnight, so they format in UTC.
 */
const DAY = new Intl.DateTimeFormat('en-GB', {
	day: 'numeric',
	month: 'short',
	year: 'numeric',
	timeZone: 'UTC',
});
const DAY_LONG = new Intl.DateTimeFormat('en-GB', {
	day: 'numeric',
	month: 'long',
	year: 'numeric',
	timeZone: 'UTC',
});
const MONTH = new Intl.DateTimeFormat('en-GB', { month: 'short', timeZone: 'UTC' });
const MONTH_YEAR = new Intl.DateTimeFormat('en-GB', {
	month: 'long',
	year: 'numeric',
	timeZone: 'UTC',
});
const COUNT = new Intl.NumberFormat('en-GB');

/** "11 Apr 2026": lists, meta lines, tiles. */
export const formatDate = (date: Date) => DAY.format(date);
/** "11 April 2026": where the date is the headline, as on the share card. */
export const formatDateLong = (date: Date) => DAY_LONG.format(date);
/** "Aug": a row inside a year that already names the year. */
export const formatMonth = (date: Date) => MONTH.format(date);
/** "August 2026". */
export const formatMonthYear = (date: Date) => MONTH_YEAR.format(date);
/** "24,897". */
export const formatCount = (n: number) => COUNT.format(n);
/** "4.0": always one decimal, so a column of ratings lines up. */
export const formatRating = (rating: number) => rating.toFixed(1);
/** "2026-04-11", for `<time dateTime>`. */
export const isoDate = (date: Date) => date.toISOString().slice(0, 10);
