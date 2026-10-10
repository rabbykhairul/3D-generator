import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from '@playwright/test';

const mode = process.argv[2];
assert.ok(['candidate-initial', 'candidate', 'after'].includes(mode), 'Usage: node scripts/surface-comparison-sheet.mjs candidate-initial|candidate|after');
const root = path.resolve('artifacts/surface-fix');
const inputDir = mode === 'candidate-initial' ? path.join(root, 'candidate-initial') : root;
const prefix = mode === 'candidate-initial' ? 'candidate' : mode;
const before = JSON.parse(await fs.readFile(path.join(root, 'baseline-result.json'), 'utf8'));
const after = JSON.parse(await fs.readFile(path.join(inputDir, `${prefix}-result.json`), 'utf8'));
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 2200 }, deviceScaleFactor: 1 });
  for (const angle of ['side', 'threequarter', 'front']) {
    const rows = [];
    for (const [display, label] of [['photo', 'Photo colors'], ['clay', 'Neutral clay'], ['silhouette', 'Silhouette']]) {
      const beforeName = `baseline-${angle}-closeup-${display}.png`;
      const afterName = `${prefix}-${angle}-closeup-${display}.png`;
      const beforePng = await fs.readFile(path.join(root, beforeName));
      const afterPng = await fs.readFile(path.join(inputDir, afterName));
      rows.push(`<div class="row"><h2>${label}</h2><div class="images"><img src="data:image/png;base64,${beforePng.toString('base64')}" alt="Before ${label}"><img src="data:image/png;base64,${afterPng.toString('base64')}" alt="After ${label}"></div></div>`);
    }
    await page.setContent(`<!doctype html><html><head><style>
      * { box-sizing: border-box; } body { margin: 0; width: 1440px; padding: 18px 30px 30px; background: #fff; font: 17px system-ui, sans-serif; color: #1b2a25; }
      header { display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin-bottom: 10px; }
      header div { padding: 9px 12px; background: #e9efeb; border: 1px solid #ccd9d0; font-weight: 700; }
      header div + div { border-left: 0; } header small { display: block; font-size: 11px; font-weight: 400; color: #486052; }
      h1 { font-size: 22px; margin: 0 0 12px; } h2 { font-size: 15px; margin: 13px 0 6px; }
      .images { display: grid; grid-template-columns: 690px 690px; }
      img { display: block; width: 690px; height: 650px; border: 1px solid #d5dfd9; }
      img + img { border-left: 0; }
    </style></head><body><h1>${angle === 'side' ? 'Right side arm transition' : angle === 'front' ? 'Front frame and lower rims' : 'Three-quarter frame'} · close-up</h1>
      <header><div>Before<small>Job ${before.jobId} · ${before.modelSha256.slice(0, 12)}</small></div><div>${mode === 'after' ? 'After' : 'Offline candidate'}<small>${after.jobId} · ${after.modelSha256.slice(0, 12)}</small></div></header>
      ${rows.join('')}</body></html>`);
    await page.locator('img').last().evaluate(image => image.decode());
    const output = path.join(root, `comparison-${mode}-${angle}-closeup.png`);
    await page.screenshot({ path: output, fullPage: true });
    console.log(output);
  }
} finally {
  await browser.close();
}
