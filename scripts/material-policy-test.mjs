import assert from 'node:assert/strict';
import { chromium } from '@playwright/test';

const base = process.env.MATERIAL_TEST_BASE || 'http://127.0.0.1:5188';
const projectId = '04863342af864ae0a4cabb8854273c49';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage();
  const response = await page.request.get(`${base}/api/projects/${projectId}`);
  assert.equal(response.status(), 200);
  const project = await response.json();
  assert.ok(project.artifact?.model_url, 'Current project has an exported GLB');
  await page.goto(base);
  const result = await page.evaluate(async modelUrl => {
    const { testPhotoMaterialPolicy } = await import('/src/viewer/materialPolicy.test.ts');
    return testPhotoMaterialPolicy(modelUrl);
  }, project.artifact.model_url);
  console.log('Photo material policy passed:', JSON.stringify(result));
  console.log('Pose check rotates the actual GLB under an AR-style parent; it does not simulate changing FaceLandmarker output.');
} finally {
  await browser.close();
}
