import test from 'node:test';
import assert from 'node:assert/strict';
import { isProductImage, productMatchScore, productIdFromUrl, codePattern } from '../scripts/collect-lenskart-playwright.mjs';

test('accepts product images and rejects unrelated assets', () => {
  assert.equal(isProductImage('https://static.lenskart.com/media/catalog/product/v/c/vc-e1234.jpg'), true);
  assert.equal(isProductImage('https://static.lenskart.com/media/catalog/product/v/c/vc-e1234.webp'), true);
  assert.equal(isProductImage('https://static.lenskart.com/media/banner.jpg'), false);
  assert.equal(isProductImage('https://example.com/media/catalog/product/vc-e1234.jpg'), false);
});

test('gives separate color variants separate product IDs', () => {
  const blaze = productIdFromUrl('https://www.lenskart.com/lenskart-hustlr-la-e15417-n-blaze-eyeglasses.html');
  const flame = productIdFromUrl('https://www.lenskart.com/lenskart-hustlr-la-e15417-n-flame-eyeglasses.html');
  assert.notEqual(blaze, flame);
});

test('matches SKU or product name to image path', () => {
  assert.match('vincent-chase-vc-e1234.html', codePattern);
  assert.equal(productMatchScore(
    'https://static.lenskart.com/media/catalog/product/vc-e1234/front.jpg',
    'vc-e1234', 'Vincent Chase Gold Sunglasses'), 100);
  assert.equal(productMatchScore(
    'https://static.lenskart.com/media/catalog/product/other-product.jpg',
    'vc-e1234', 'Vincent Chase Gold Sunglasses'), 0);
});
