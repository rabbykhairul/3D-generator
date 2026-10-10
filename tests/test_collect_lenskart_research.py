import io
import json
from urllib.error import URLError

from PIL import Image

from scripts.collect_lenskart_research import collect, product_images, product_links


def test_extracts_product_links_and_distinct_catalog_images():
    listing = '<a href="/vc-s19329-gold-sunglasses.html">Pair</a><a href="https://other.example/a.html">Other</a>'
    assert product_links('https://www.lenskart.com/sunglasses', listing) == [
        'https://www.lenskart.com/vc-s19329-gold-sunglasses.html']
    page = '''<meta property="og:image" content="https://static5.lenskart.com/media/catalog/product/front.jpg">
    <script type="application/ld+json">{"@type":"Product","image":["https://static5.lenskart.com/media/catalog/product/front.jpg","https://static5.lenskart.com/media/catalog/product/side.webp"]}</script>
    <img src="https://static5.lenskart.com/media/banner.jpg">'''
    assert product_images('https://www.lenskart.com/vc-s19329-gold-sunglasses.html', page) == [
        'https://static5.lenskart.com/media/catalog/product/front.jpg',
        'https://static5.lenskart.com/media/catalog/product/side.webp']


def test_collection_preserves_provenance_and_does_not_claim_labels(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.collect_lenskart_research.time.sleep', lambda _: None)
    page_url = 'https://www.lenskart.com/vc-s19329-gold-sunglasses.html'
    image_url = 'https://static5.lenskart.com/media/catalog/product/front.jpg'
    image = io.BytesIO()
    Image.new('RGB', (100, 100), 'red').save(image, format='JPEG')
    payloads = {page_url: f'<img src="{image_url}">'.encode(), image_url: image.getvalue()}
    manifest = collect([page_url], tmp_path, fetcher=lambda url: payloads[url])
    product = manifest['products'][0]
    assert product['product_id'] == 'vc-s19329'
    assert product['status'] == 'collected'
    assert product['labels_reviewed'] is False and product['rights_reviewed'] is False
    assert product['photos'][0]['source_url'] == image_url
    assert product['photos'][0]['view'] == 'unknown'
    assert json.loads((tmp_path / 'collection.json').read_text()) == manifest
    assert collect([page_url], tmp_path, fetcher=lambda url: payloads[url]) == manifest


def test_failed_page_can_be_retried(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.collect_lenskart_research.time.sleep', lambda _: None)
    page_url = 'https://www.lenskart.com/vc-s19329-gold-sunglasses.html'
    def failed(_):
        raise URLError('offline')
    assert collect([page_url], tmp_path, fetcher=failed)['products'][0]['status'] == 'failed'
    second = collect([page_url], tmp_path, fetcher=lambda _: b'<title>Product</title>')
    assert len(second['products']) == 1
    assert second['products'][0]['status'] == 'no-product-photos'
