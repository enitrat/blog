/** A title without its trailing "(series, tome)", as every list and tile shows it. */
export const shortTitle = (title: string) => title.replace(/\s*\(.*\)\s*$/, '');
