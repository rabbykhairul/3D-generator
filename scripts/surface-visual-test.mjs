import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from '@playwright/test';

const base = process.env.SURFACE_TEST_BASE || 'http://127.0.0.1:5188';
const projectId = '04863342af864ae0a4cabb8854273c49';
const originalJobId = 'a4fbcb83ead2497b937bea132c8229e1';
const sourcePhotoId = '194e7d371e31448fadc30af0f17eeb14';
const evidenceDir = path.resolve('artifacts/surface-fix');
const mode = process.argv[2];
assert.ok(['baseline', 'candidate', 'after'].includes(mode), 'Usage: node scripts/surface-visual-test.mjs baseline|candidate|after');
await fs.mkdir(evidenceDir, { recursive: true });

const response = await fetch(`${base}/api/projects/${projectId}`);
assert.equal(response.status, 200, 'Project API is available');
const live = await response.json();
let project;
let modelBytes;
if (mode === 'baseline') {
  const artifactDir = path.resolve('data/artifacts', originalJobId);
  modelBytes = await fs.readFile(path.join(artifactDir, 'model.glb'));
  const metadata = JSON.parse(await fs.readFile(path.join(artifactDir, 'metadata.json'), 'utf8'));
  project = {
    ...live,
    revision: metadata.input_revision,
    stale: false,
    artifact: {
      job_id: originalJobId,
      revision: metadata.input_revision,
      model_url: `/api/artifacts/${originalJobId}/model.glb`,
      metadata_url: `/api/artifacts/${originalJobId}/metadata.json`,
      metadata,
    },
  };
  await fs.writeFile(path.join(evidenceDir, 'baseline-project.json'), JSON.stringify(project, null, 2));
  await fs.writeFile(path.join(evidenceDir, 'baseline-model.glb'), modelBytes);
  await fs.copyFile(path.resolve('data/projects', projectId, 'photos', `${sourcePhotoId}.png`), path.join(evidenceDir, 'source-right-reference.png'));
} else if (mode === 'candidate') {
  const candidateDir = path.join(evidenceDir, 'candidate');
  modelBytes = await fs.readFile(path.join(candidateDir, 'model.glb'));
  const pinnedBaseline = JSON.parse(await fs.readFile(path.join(evidenceDir, 'baseline-project.json'), 'utf8'));
  let metadata = pinnedBaseline.artifact.metadata;
  try { metadata = JSON.parse(await fs.readFile(path.join(candidateDir, 'metadata.json'), 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  project = {
    ...pinnedBaseline,
    artifact: {
      ...pinnedBaseline.artifact,
      job_id: 'candidate-local',
      model_url: '/api/artifacts/candidate-local/model.glb',
      metadata_url: '/api/artifacts/candidate-local/metadata.json',
      metadata,
    },
  };
} else {
  const expectedJobId = process.env.SURFACE_AFTER_JOB_ID;
  assert.ok(expectedJobId, 'Set SURFACE_AFTER_JOB_ID to the confirmed corrected job before after capture');
  assert.ok(live.artifact?.job_id, 'Regenerated model exists');
  assert.equal(live.artifact.job_id, expectedJobId, 'Live artifact is the confirmed corrected job');
  assert.notEqual(live.artifact.job_id, originalJobId, 'After capture requires a new artifact');
  assert.equal(live.stale, false, 'Regenerated model is current');
  project = live;
  const glbResponse = await fetch(`${base}${project.artifact.model_url}`);
  assert.equal(glbResponse.status, 200, 'Exported GLB is available');
  modelBytes = Buffer.from(await glbResponse.arrayBuffer());
  await fs.writeFile(path.join(evidenceDir, 'after-model.glb'), modelBytes);
}
assert.equal(modelBytes.toString('ascii', 0, 4), 'glTF', 'Pinned asset is a GLB');
const modelSha256 = createHash('sha256').update(modelBytes).digest('hex');

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('requestfailed', request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
try {
  // Pin both JSON and GLB bytes. A new job cannot silently alter the baseline.
  await page.route(`${base}/api/projects/${projectId}`, route => route.fulfill({ json: project }));
  await page.route(`${base}${project.artifact.model_url}`, route => route.fulfill({ body: modelBytes, contentType: 'model/gltf-binary' }));
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.locator('.project-nav').filter({ hasText: projectId.slice(0, 6) }).first().click();
  await page.getByRole('tab', { name: /Review \/ AR/ }).click();
  const canvas = page.getByLabel('Interactive three-dimensional eyewear model');
  await canvas.waitFor();
  const display = page.getByLabel('Surface display');
  const captures = [];
  for (const [value, label] of [['photo', 'photo'], ['clay', 'clay'], ['silhouette', 'silhouette']]) {
    await display.selectOption(value);
    // React recreates the WebGL scene for each display; wait past the previous
    // frame so a stale metrics element cannot make the capture race the loader.
    await page.waitForTimeout(1000);
    await page.locator('.mv-metrics dd').first().waitFor({ timeout: 30_000 });
    await page.locator('.mv-stage-status').waitFor({ state: 'hidden' });
    for (const [angle, button] of [['side', 'Other side'], ['threequarter', 'Orbit'], ['front', 'Front']]) {
      await page.getByRole('button', { name: button, exact: true }).click();
      await page.waitForTimeout(350);
      const normal = `${mode}-${angle}-${label}.png`;
      await canvas.screenshot({ path: path.join(evidenceDir, normal) });
      captures.push(normal);
      const box = await canvas.boundingBox();
      assert.ok(box);
      await page.mouse.move(box.x + box.width * 0.52, box.y + box.height * 0.52);
      await page.mouse.wheel(0, -780);
      await page.waitForTimeout(450);
      const closeup = `${mode}-${angle}-closeup-${label}.png`;
      await canvas.screenshot({ path: path.join(evidenceDir, closeup) });
      captures.push(closeup);
    }
  }
  assert.deepEqual(errors, [], 'Browser has no page or request errors');
  const result = { mode, projectId, jobId: project.artifact.job_id, modelSha256, referencePhotoId: sourcePhotoId, captures, errors, capturedAt: new Date().toISOString() };
  await fs.writeFile(path.join(evidenceDir, `${mode}-result.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
