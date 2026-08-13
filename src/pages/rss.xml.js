import rss from '@astrojs/rss';
import { SITE_DESCRIPTION, SITE_TITLE } from '../consts';
import { getWritingIndex, postUrl } from '../utils/writing';

export async function GET(context) {
	const posts = await getWritingIndex();
	return rss({
		title: SITE_TITLE,
		description: SITE_DESCRIPTION,
		site: context.site,
		customData: '<language>en-gb</language>',
		items: posts.map((post) => ({
			...post.data,
			link: postUrl(post),
		})),
	});
}
