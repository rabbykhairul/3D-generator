import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from '@playwright/test';

const base = process.env.ASSEMBLY_TEST_BASE || 'http://127.0.0.1:5188';
const projectId = '04863342af864ae0a4cabb8854273c49';
const originalJobId = '676144e72de843759e3ee5563c7f31e1';
const mode = process.argv[2] || 'baseline';
assert.ok(['baseline', 'after', 'angles'].includes(mode), 'Usage: node scripts/assembly-visual-test.mjs baseline|after|angles');
const evidenceDir = path.resolve('artifacts/assembly-fix');
const snapshotPath = path.join(evidenceDir, 'baseline-project.json');
await fs.mkdir(evidenceDir, { recursive: true });

if (mode === 'angles') {
  const pinned = JSON.parse(await fs.readFile(snapshotPath, 'utf8'));
  assert.equal(pinned.artifact.job_id, originalJobId);
  const liveResponse = await fetch(`${base}/api/projects/${projectId}`);
  assert.equal(liveResponse.status, 200);
  const live = await liveResponse.json();
  assert.notEqual(live.artifact?.job_id, originalJobId, 'Current project has the regenerated GLB');
  const angleResults = [];
  for (const [label, project] of [['baseline', pinned], ['after', live]]) {
    const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
    try {
      if (label === 'baseline') {
        await page.route(`${base}/api/projects/${projectId}`, route => route.fulfill({ json: project }));
      }
      await page.goto(base, { waitUntil: 'domcontentloaded' });
      await page.locator('.project-nav').filter({ hasText: projectId.slice(0, 6) }).first().click();
      await page.getByRole('tab', { name: /Review \/ AR/ }).click();
      const canvas = page.getByLabel('Interactive three-dimensional eyewear model');
      await canvas.waitFor();
      await page.locator('.mv-metrics dd').first().waitFor({ timeout: 30_000 });
      await page.locator('.mv-stage-status').waitFor({ state: 'hidden' });
      await page.getByRole('button', { name: 'Front', exact: true }).click();
      const box = await canvas.boundingBox();
      assert.ok(box);
      const startX = box.x + box.width / 2;
      const startY = box.y + box.height / 2;
      // OrbitControls uses the canvas height for horizontal rotation. A 100 px
      // drag on this 649 px canvas moves the camera roughly 55 degrees.
      await page.mouse.move(startX, startY);
      await page.mouse.down();
      await page.mouse.move(startX + 100, startY, { steps: 12 });
      await page.mouse.up();
      await page.waitForTimeout(900);
      await canvas.screenshot({ path: path.join(evidenceDir, `${label}-near-side.png`) });
      await page.getByRole('button', { name: 'Top', exact: true }).click();
      await page.waitForTimeout(500);
      await canvas.screenshot({ path: path.join(evidenceDir, `${label}-top.png`) });
      assert.deepEqual(errors, []);
      angleResults.push({ label, jobId: project.artifact.job_id, canvas: { width: box.width, height: box.height }, dragPixels: 100, errors });
    } finally {
      await browser.close();
    }
  }
  await fs.writeFile(path.join(evidenceDir, 'angles-result.json'), JSON.stringify(angleResults, null, 2));
  console.log(JSON.stringify(angleResults, null, 2));
  process.exit(0);
}

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
const errors = [];
const failedRequests = [];
page.on('pageerror', error => errors.push(error.message));
page.on('requestfailed', request => failedRequests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText}`));

function glbSummary(buffer) {
  assert.equal(buffer.toString('ascii', 0, 4), 'glTF', 'Download is a GLB');
  assert.equal(buffer.readUInt32LE(4), 2, 'GLB version');
  assert.equal(buffer.readUInt32LE(8), buffer.length, 'GLB length');
  assert.equal(buffer.toString('ascii', 16, 20), 'JSON', 'GLB JSON chunk');
  const gltf = JSON.parse(buffer.toString('utf8', 20, 20 + buffer.readUInt32LE(12)));
  const names = (gltf.meshes || []).map(mesh => mesh.name);
  const mappedMaterials = (gltf.materials || []).filter(material => material.pbrMetallicRoughness?.baseColorTexture).length;
  return {
    bytes: buffer.length,
    meshNames: names,
    meshCount: names.length,
    materialCount: gltf.materials?.length || 0,
    mappedMaterials,
    textureCount: gltf.textures?.length || 0,
    imageCount: gltf.images?.length || 0,
    nodes: gltf.nodes?.length || 0,
  };
}

try {
  let project;
  if (mode === 'baseline') {
    const response = await page.request.get(`${base}/api/projects/${projectId}`);
    assert.equal(response.status(), 200);
    project = await response.json();
    assert.equal(project.artifact?.job_id, originalJobId, 'Capture the original asset before regeneration');
    await fs.writeFile(snapshotPath, JSON.stringify(project));
  } else {
    const response = await page.request.get(`${base}/api/projects/${projectId}`);
    assert.equal(response.status(), 200);
    project = await response.json();
    assert.ok(project.artifact?.job_id, 'Regenerated artifact exists');
    assert.notEqual(project.artifact.job_id, originalJobId, 'After capture uses a regenerated artifact');
    assert.equal(project.stale, false, 'Regenerated artifact is current');
  }

  const artifact = project.artifact;
  const modelResponse = await page.request.get(`${base}${artifact.model_url}`);
  assert.equal(modelResponse.status(), 200, 'Exported GLB is available');
  const glb = await modelResponse.body();
  const glbInfo = glbSummary(glb);
  assert.ok(glbInfo.mappedMaterials > 0 && glbInfo.textureCount > 0, 'GLB retains texture maps');
  await fs.writeFile(path.join(evidenceDir, `${mode}-model.glb`), glb);

  if (mode === 'baseline') {
    // The project may be regenerated while this script runs. Pin the viewer's
    // project fetch to the captured response so its model URL stays stable.
    await page.route(`${base}/api/projects/${projectId}`, route => route.fulfill({ json: project }));
  }
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.locator('.project-nav').filter({ hasText: projectId.slice(0, 6) }).first().click();
  await page.getByRole('tab', { name: /Review \/ AR/ }).click();
  const canvas = page.getByLabel('Interactive three-dimensional eyewear model');
  await canvas.waitFor();
  await page.locator('.mv-metrics dd').first().waitFor({ timeout: 30_000 });
  await page.locator('.mv-stage-status').waitFor({ state: 'hidden' });
  assert.ok((await page.locator('.mv-parts label').count()) >= 7, 'Named front, lens, hinge and arm parts load');
  const views = [
    ['front', 'Front'],
    ['side', 'Side'],
    ['threequarter', 'Orbit'],
  ];
  for (const [name, button] of views) {
    await page.getByRole('button', { name: button, exact: true }).click();
    await page.waitForTimeout(350);
    await canvas.screenshot({ path: path.join(evidenceDir, `${mode}-${name}.png`) });
  }
  const dimensions = await page.locator('.mv-metrics dd').allTextContents();
  const exportLinks = await page.locator('.artifact-links a').evaluateAll(links => links.map(link => ({ text: link.textContent?.trim(), href: link.getAttribute('href') })));
  assert.ok(exportLinks.some(link => link.href === artifact.model_url), 'GLB export link points to the displayed asset');
  assert.ok(exportLinks.some(link => link.href === artifact.metadata_url), 'Metadata export link exists');
  const metadataResponse = await page.request.get(`${base}${artifact.metadata_url}`);
  assert.equal(metadataResponse.status(), 200, 'Metadata export is available');
  assert.equal((await metadataResponse.json()).model_file, 'model.glb');
  assert.deepEqual(errors, [], 'No page errors');
  assert.deepEqual(failedRequests, [], 'No failed requests');

  const result = { mode, projectId, jobId: artifact.job_id, revision: artifact.revision, modelUrl: artifact.model_url, glb: glbInfo, dimensions, exportLinks, errors, failedRequests, capturedAt: new Date().toISOString() };
  await fs.writeFile(path.join(evidenceDir, `${mode}-result.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
