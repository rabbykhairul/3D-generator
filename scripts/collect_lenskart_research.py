"""Collect a bounded, local research cache of public Lenskart product views.

This does not label components or make the photos suitable for model training.
It does not log in, bypass access controls, or retry rate limits. Record source
and product identity so later annotations can be split by physical product.
"""

import argparse
import hashlib
from html import unescape
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATTERN = re.compile(r'https?:[/]{2}[^\s"<>]+?\.(?:jpe?g|png|webp)(?:\?[^\s"<>]*)?', re.I)
PRODUCT_CODE = re.compile(r'(?i)(?:vc-s\d+(?:-c\d+)?|jj-s\d+-c\d+|l[rb]-e\d+|lk-s\w+)')


def lenskart_host(url):
    host = (urlparse(url).hostname or '').lower()
    return host == 'lenskart.com' or host.endswith('.lenskart.com')


def image_url(url):
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    return (lenskart_host(url) and (host.startswith('static') or host.startswith('cdn'))
            and re.search(r'\.(?:jpe?g|png|webp)$', parsed.path, re.I)
            and '/catalog/product/' in parsed.path)


class ProductPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []
        self.json_ld = []
        self._in_json_ld = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
        if tag == 'img':
            self.images.extend(attrs[key] for key in ('src', 'data-src', 'data-original')
                               if attrs.get(key))
            for key in ('srcset', 'data-srcset'):
                if attrs.get(key):
                    self.images.extend(part.strip().split(' ')[0]
                                       for part in attrs[key].split(',') if part.strip())
        if tag == 'meta' and attrs.get('property') in ('og:image', 'og:image:url'):
            if attrs.get('content'):
                self.images.append(attrs['content'])
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self._in_json_ld = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self._in_json_ld = False

    def handle_data(self, data):
        if self._in_json_ld:
            self.json_ld.append(data)


def product_links(page_url, html):
    parser = ProductPageParser()
    parser.feed(html)
    urls = []
    for href in parser.links:
        url = urljoin(page_url, unescape(href))
        if lenskart_host(url) and urlparse(url).path.endswith('.html'):
            urls.append(url.split('#', 1)[0])
    return list(dict.fromkeys(urls))


def product_images(page_url, html):
    parser = ProductPageParser()
    parser.feed(html)
    candidates = list(parser.images)
    for block in parser.json_ld:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        queue = [data]
        while queue:
            node = queue.pop()
            if isinstance(node, list):
                queue.extend(node)
            elif isinstance(node, dict):
                for key, value in node.items():
                    if key == 'image':
                        if isinstance(value, str):
                            candidates.append(value)
                        elif isinstance(value, list):
                            candidates.extend(v for v in value if isinstance(v, str))
                    elif isinstance(value, (dict, list)):
                        queue.append(value)
    clean = unescape(html).replace('\\/', '/')
    candidates.extend(IMAGE_PATTERN.findall(clean))
    urls = []
    for candidate in candidates:
        url = urljoin(page_url, unescape(candidate).replace('\\/', '/'))
        if image_url(url):
            urls.append(url)
    return list(dict.fromkeys(urls))


def fetch(url, timeout=30):
    with urlopen(Request(url, headers={'User-Agent': 'EyewearResearchPilot/0.1'}),
                 timeout=timeout) as response:
        length = response.headers.get('Content-Length')
        if length and int(length) > 12_000_000:
            raise ValueError('Response exceeds 12 MB limit')
        payload = response.read(12_000_001)
    if len(payload) > 12_000_000:
        raise ValueError('Response exceeds 12 MB limit')
    return payload


def collect(product_urls, output, max_products=20, max_images=8, delay=1.0,
            fetcher=fetch):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'collection.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {
        'schema': 1, 'source': 'lenskart-public-research', 'products': []}
    existing = {item['page_url'] for item in manifest['products']
                if item.get('status') == 'collected'}
    image_hashes = {photo['sha256'] for item in manifest['products']
                    for photo in item.get('photos', [])}
    for page_url in list(dict.fromkeys(product_urls))[:max_products]:
        if not lenskart_host(page_url) or not urlparse(page_url).path.endswith('.html'):
            raise ValueError(f'Not a Lenskart product URL: {page_url}')
        if page_url in existing:
            continue
        manifest['products'] = [item for item in manifest['products']
                                if item['page_url'] != page_url]
        product_code = PRODUCT_CODE.search(urlparse(page_url).path)
        product_id = (product_code.group(0).lower() if product_code else
                      hashlib.sha256(page_url.encode()).hexdigest()[:16])
        record = {'product_id': product_id, 'page_url': page_url,
                  'photos': [], 'labels_reviewed': False, 'rights_reviewed': False}
        try:
            markup = fetcher(page_url).decode('utf-8', errors='replace')
            urls = product_images(page_url, markup)[:max_images]
            for image_source in urls:
                time.sleep(delay)
                data = fetcher(image_source)
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                digest = hashlib.sha256(data).hexdigest()
                if digest in image_hashes:
                    continue
                ext = Path(urlparse(image_source).path).suffix.lower()
                folder = output / product_id
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f'{len(record["photos"]):02d}-{digest[:12]}{ext}'
                path.write_bytes(data)
                record['photos'].append({'path': str(path.resolve()),
                    'source_url': image_source, 'sha256': digest, 'view': 'unknown'})
                image_hashes.add(digest)
            record['status'] = 'collected' if record['photos'] else 'no-product-photos'
        except (HTTPError, URLError, OSError, ValueError) as error:
            record['status'] = 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
        manifest['products'].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        time.sleep(delay)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listing-url', action='append', default=[])
    parser.add_argument('--product-url', action='append', default=[])
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/eyewear-training/lenskart-research')
    parser.add_argument('--max-products', type=int, default=20)
    parser.add_argument('--max-images', type=int, default=8)
    parser.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.max_products <= 500 or not 1 <= args.max_images <= 20 or args.delay < .5:
        parser.error('Use 1–500 products, 1–20 photos each, and at least 0.5 s delay')
    urls = list(args.product_url)
    for listing in args.listing_url:
        if not lenskart_host(listing):
            parser.error(f'Not a Lenskart listing URL: {listing}')
        urls.extend(product_links(listing, fetch(listing).decode('utf-8', errors='replace')))
    if not urls:
        parser.error('Provide --product-url or --listing-url')
    manifest = collect(urls, args.output, args.max_products, args.max_images,
                       args.delay)
    counts = {status: sum(p['status'] == status for p in manifest['products'])
              for status in ('collected', 'no-product-photos', 'failed')}
    print(json.dumps({'manifest': str((args.output / 'collection.json').resolve()),
                      'products': len(manifest['products']), 'status': counts}, indent=2))
    if counts['collected'] == 0:
        raise SystemExit('No product photos were collected; inspect the manifest errors')


if __name__ == '__main__':
    main()
