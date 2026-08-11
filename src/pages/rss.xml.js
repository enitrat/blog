import { getCollection } from 'astro:content';
import rss from '@astrojs/rss';
import { SITE_DESCRIPTION, SITE_TITLE } from '../consts';

export async function GET(context) {
	// A translation is reachable from the article itself, never a second item.
	const posts = (await getCollection('blog')).filter((post) => post.data.lang === 'en');
	return rss({
		title: SITE_TITLE,
		description: SITE_DESCRIPTION,
		site: context.site,
		customData: '<language>en-gb</language>',
		items: posts.map((post) => ({
			...post.data,
			link: `/writing/${post.id}/`,
		})),
	});
}
