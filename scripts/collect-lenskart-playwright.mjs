/** Resumable product-photo research collector using Chrome network responses. */
import { chromium } from '@playwright/test';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';

const root = path.resolve(import.meta.dirname, '..');
const options = { productUrls: [], listingUrls: [], sitemapUrls: [],
  output: path.join(root, 'artifacts/eyewear-training/lenskart-research'),
  maxProducts: 0, maxImages: 8, cdpUrl: null };

for (let i = 2; i < process.argv.length; i++) {
  const flag = process.argv[i];
  const value = process.argv[++i];
  if (!value) throw new Error(`Missing value for ${flag}`);
  if (flag === '--product-url') options.productUrls.push(value);
  else if (flag === '--listing-url') options.listingUrls.push(value);
  else if (flag === '--sitemap-url') options.sitemapUrls.push(value);
  else if (flag === '--output') options.output = path.resolve(value);
  else if (flag === '--max-products') options.maxProducts = Number(value);
  else if (flag === '--max-images') options.maxImages = Number(value);
  else if (flag === '--cdp-url') options.cdpUrl = value;
  else throw new Error(`Unknown option: ${flag}`);
}
if (!Number.isInteger(options.maxProducts) || options.maxProducts < 0 || options.maxProducts > 100000 ||
    !Number.isInteger(options.maxImages) || options.maxImages < 1 || options.maxImages > 20) {
  throw new Error('Use 0–100000 products (0 means all discovered) and 1–20 images per product');
}
if (!options.productUrls.length && !options.listingUrls.length && !options.sitemapUrls.length) {
  options.sitemapUrls.push('https://www.lenskart.com/sitemap.xml');
}

const codePattern = /(?:[a-z]{1,5}-[es]\d{3,}(?:-c\d+)?|\d{6,})/i;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

function isLenskart(url) {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host === 'lenskart.com' || host.endsWith('.lenskart.com');
  } catch { return false; }
}

function isProductImage(url) {
  try {
    const parsed = new URL(url);
    return isLenskart(url) &&
      parsed.pathname.includes('/catalog/product/') &&
      /\.(jpe?g|png|webp)$/i.test(parsed.pathname);
  } catch { return false; }
}

function isImage(bytes) {
  return (bytes.length > 12 &&
    ((bytes[0] === 0xff && bytes[1] === 0xd8) ||
      (bytes[0] === 0x89 && bytes.subarray(1, 4).toString() === 'PNG') ||
      (bytes.subarray(0, 4).toString() === 'RIFF' && bytes.subarray(8, 12).toString() === 'WEBP')));
}

async function collectImageUrls(page) {
  const raw = await page.evaluate(() => {
    const urls = [];
    for (const image of document.images) {
      for (const value of [image.currentSrc, image.src, image.dataset.src, image.dataset.original]) {
        if (value) urls.push(value);
      }
      for (const item of (image.getAttribute('srcset') || '').split(',')) {
        const url = item.trim().split(/\s+/)[0];
        if (url) urls.push(url);
      }
    }
    for (const meta of document.querySelectorAll('meta[property="og:image"]')) {
      if (meta.content) urls.push(meta.content);
    }
    return urls;
  });
  const markup = await page.content();
  raw.push(...(markup.replaceAll('\\/', '/').match(/https?:\/\/[^\s"<>]+?\.(?:jpe?g|png|webp)(?:\?[^\s"<>]*)?/gi) || []));
  return [...new Set(raw.map(value => {
    try { return new URL(value.replaceAll('&amp;', '&'), page.url()).href; }
    catch { return null; }
  }).filter(value => value && isProductImage(value)))];
}

function decodeXml(value) {
  return value.replaceAll('&amp;', '&').replaceAll('&quot;', '"')
    .replaceAll('&apos;', "'").replaceAll('&lt;', '<').replaceAll('&gt;', '>');
}

async function collectSitemapUrls(context, starts) {
  const queue = [...starts];
  const visited = new Set();
  const products = new Set();
  while (queue.length) {
    const url = queue.shift();
    if (visited.has(url)) continue;
    if (!isLenskart(url) || !/\.xml(?:\.gz)?(?:$|\?)/i.test(url)) continue;
    if (visited.size >= 300) throw new Error('Sitemap limit reached; catalog discovery incomplete');
    visited.add(url);
    const response = await context.request.get(url, { timeout: 30000 });
    if (!response?.ok()) throw new Error(`Sitemap HTTP ${response?.status()}: ${url}`);
    let body = Buffer.from(await response.body());
    if (body[0] === 0x1f && body[1] === 0x8b) body = gunzipSync(body);
    for (const match of body.toString('utf8').matchAll(/<loc(?:\s[^>]*)?>([\s\S]*?)<\/loc>/gi)) {
      const candidate = decodeXml(match[1].trim());
      if (!isLenskart(candidate)) continue;
      if (/\.xml(?:\.gz)?(?:$|\?)/i.test(candidate)) queue.push(candidate);
      else if (new URL(candidate).pathname.endsWith('.html') &&
        codePattern.test(new URL(candidate).pathname)) products.add(candidate);
    }
    await sleep(600);
  }
  console.log(`Discovered ${products.size} HTML pages from ${visited.size} sitemaps`);
  return [...products];
}

function productMatchScore(imageUrl, productCode, productName) {
  const imagePath = decodeURIComponent(new URL(imageUrl).pathname).toLowerCase()
    .replace(/[^a-z0-9]/g, '');
  const code = productCode.toLowerCase().replace(/[^a-z0-9]/g, '');
  if (code.length >= 5 && imagePath.includes(code)) return 100;
  const tokens = `${productName} ${productCode}`.toLowerCase().split(/[^a-z0-9]+/)
    .filter(token => token.length >= 4 && !/^(full|frame|glasses|eyeglasses|sunglasses|size|with|online)$/.test(token));
  const matches = [...new Set(tokens)].filter(token => imagePath.includes(token)).length;
  return matches >= 2 ? matches : 0;
}

function productIdFromUrl(url) {
  return path.posix.basename(new URL(url).pathname, '.html').toLowerCase()
    .replace(/[^a-z0-9_-]/g, '-');
}

async function captureProductImages(page, url) {
  const captures = new Map();
  const pending = new Set();
  const onResponse = response => {
    if (!isProductImage(response.url()) || !response.ok()) return;
    const task = (async () => {
      const headers = await response.allHeaders();
      const size = Number(headers['content-length'] || 0);
      if (size > 12_000_000) return;
      const bytes = Buffer.from(await response.body());
      if (bytes.length <= 12_000_000 && isImage(bytes)) captures.set(response.url(), bytes);
    })().catch(() => {});
    pending.add(task);
    task.finally(() => pending.delete(task));
  };
  page.on('response', onResponse);
  try {
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    if (!response?.ok()) throw new Error(`Product HTTP ${response?.status()}`);
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1000);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(600);
    await Promise.allSettled([...pending]);
    return captures;
  } finally {
    page.off('response', onResponse);
  }
}

async function main() {
  await fs.mkdir(options.output, { recursive: true });
  const manifestPath = path.join(options.output, 'collection.json');
  let manifest;
  try { manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8')); }
  catch { manifest = { schema: 1, source: 'lenskart-public-research', products: [] }; }
  const browser = options.cdpUrl ? await chromium.connectOverCDP(options.cdpUrl) :
    await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const urls = [...options.productUrls];
  try {
    urls.push(...await collectSitemapUrls(context, options.sitemapUrls));
    for (const listing of options.listingUrls) {
      if (!isLenskart(listing)) throw new Error(`Invalid listing host: ${listing}`);
      await page.goto(listing, { waitUntil: 'domcontentloaded', timeout: 30000 });
      const links = await page.locator('a[href]').evaluateAll(nodes => nodes.map(a => a.href));
      urls.push(...links.filter(url => isLenskart(url) && codePattern.test(new URL(url).pathname)
        && new URL(url).pathname.endsWith('.html')));
      await sleep(600);
    }
    const seenHashes = new Set(manifest.products.flatMap(product =>
      (product.photos || []).map(photo => photo.sha256)));
    const selectedUrls = [...new Set(urls)];
    const limitedUrls = options.maxProducts ? selectedUrls.slice(0, options.maxProducts) : selectedUrls;
    console.log(`Visiting ${limitedUrls.length} product candidates`);
    for (const url of limitedUrls) {
      if (!isLenskart(url) || !new URL(url).pathname.endsWith('.html')) continue;
      if (manifest.products.some(product => product.page_url === url && product.status === 'collected')) continue;
      manifest.products = manifest.products.filter(product => product.page_url !== url);
      const code = new URL(url).pathname.match(codePattern)?.[0].toLowerCase() || digest(Buffer.from(url)).slice(0, 16);
      const productId = productIdFromUrl(url);
      const record = { product_id: productId, model_code: code, page_url: url, photos: [],
        labels_reviewed: false, rights_reviewed: false };
      try {
        const captured = await captureProductImages(page, url);
        const name = await page.locator('h1').first().textContent({ timeout: 3000 }).catch(() => null);
        record.product_name = name?.trim() || await page.title();
        const domUrls = await collectImageUrls(page);
        const imageUrls = [...new Set([...captured.keys(), ...domUrls])]
          .map(imageUrl => ({ url: imageUrl, score: productMatchScore(imageUrl, code, record.product_name) }))
          .filter(item => item.score > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, options.maxImages);
        for (const image of imageUrls) {
          const imageUrl = image.url;
          let bytes = captured.get(imageUrl);
          let captureSource = 'chrome-network';
          if (!bytes) {
            await sleep(600);
            const imageResponse = await context.request.get(imageUrl, { timeout: 30000 });
            if (!imageResponse.ok()) continue;
            bytes = Buffer.from(await imageResponse.body());
            captureSource = 'page-url-fallback';
          }
          if (bytes.length > 12_000_000 || !isImage(bytes)) continue;
          const hash = digest(bytes);
          if (seenHashes.has(hash)) continue;
          const folder = path.join(options.output, productId);
          await fs.mkdir(folder, { recursive: true });
          const ext = path.extname(new URL(imageUrl).pathname).toLowerCase();
          const destination = path.join(folder, `${String(record.photos.length).padStart(2, '0')}-${hash.slice(0, 12)}${ext}`);
          await fs.writeFile(destination, bytes);
          record.photos.push({ path: destination, source_url: imageUrl,
            sha256: hash, view: 'unknown', capture_source: captureSource,
            name_match_score: image.score });
          seenHashes.add(hash);
        }
        record.status = record.photos.length ? 'collected' : 'no-product-photos';
      } catch (error) {
        record.status = 'failed';
        record.error = String(error).slice(0, 500);
      }
      manifest.products.push(record);
      await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
      console.log(`${record.product_id}: ${record.status}, ${record.photos.length} photos`);
      await sleep(600);
    }
  } finally {
    await context.close();
    if (!options.cdpUrl) await browser.close();
  }
  const collected = manifest.products.filter(product => product.status === 'collected').length;
  if (!collected) process.exitCode = 2;
  console.log(JSON.stringify({ manifest: manifestPath, collected_products: collected }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}

export { isProductImage, productMatchScore, productIdFromUrl, codePattern };
