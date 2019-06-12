# Necessary Libraries
import flickrapi
import urllib

# Flickr Credentials
api_key='628d66f2f9f6d6ac9dedd2d4a079a204'
api_secret='58179c9b212b486b'


flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)

Keyward = 585585

photos = flickr.photo.search()
    (text = Keyward, tag_mode = 'all', tags=Keyward, extras = 'url_c', per_page = 10, sort='relevance')
urls = []


for i, photo in enumerate(photos):
    print(i)

    url = photo.get('url_c')
    urls.append(url)

    # get 50 urls
    if i > 50:
        break

print(urls)


