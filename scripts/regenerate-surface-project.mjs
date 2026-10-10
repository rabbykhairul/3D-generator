// Deliberate refresh of the merchant project after reviewing
// artifacts/surface-fix/side-contours.png and the current front suggestion.
// Never run this as an unattended regression test.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '@playwright/test';

const TARGET = '04863342af864ae0a4cabb8854273c49';
const SIDE_PHOTO = '194e7d371e31448fadc30af0f17eeb14';
const FRONT_PHOTO = '15be4a47102742ffaaccb9f3be214e07';
const SOURCE = 'single-visible-arm-smooth-v4';
const [id, revisionArg] = process.argv.slice(2);
assert.equal(id, TARGET, 'Pass the reviewed merchant project ID exactly');
assert.match(revisionArg || '', /^\d+$/, 'Usage: node scripts/regenerate-surface-project.mjs ID EXPECTED_REVISION');
const expectedRevision = Number(revisionArg);
const base = 'http://127.0.0.1:5188';

const browser = await chromium.launch({channel:'chrome', headless:true,
  args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport:{width:1440,height:1100}});
const project = async () => {
  const response = await page.request.get(`${base}/api/projects/${id}`);
  assert.equal(response.status(), 200, 'Project must be available in the local studio');
  return response.json();
};
const sameInputs = (actual, before) => {
  assert.deepEqual(actual.photos, before.photos, 'Photo evidence must remain unchanged');
  assert.deepEqual(actual.material, before.material, 'Material settings must remain unchanged');
  assert.deepEqual(actual.measurements, before.measurements, 'Physical measurements must remain unchanged');
  assert.equal(actual.category, before.category, 'Frame construction must remain unchanged');
};

try {
  const before = await project();
  assert.equal(before.revision, expectedRevision, 'Stop if the project changed since contour review');
  assert.equal(before.artifact?.metadata?.engine_version, 3, 'Expected the existing version 3 artifact');
  assert.ok(before.photos.some(photo => photo.id === SIDE_PHOTO && photo.view === 'right' && photo.role === 'fitting'));
  assert.ok(before.photos.some(photo => photo.id === FRONT_PHOTO && photo.view === 'front' && photo.role === 'fitting'));
  assert.equal(before.annotations.find(a => a.part === 'temple-right')?.source, 'merchant-reviewed-side');
  assert.equal(before.annotations.find(a => a.part === 'temple-left')?.source, 'explicit-mirrored-side-assumption');

  // Extraction suggests geometry; it does not approve or change this project.
  const sideResponse = await page.request.post(`${base}/api/projects/${id}/photos/${SIDE_PHOTO}/extract`);
  assert.equal(sideResponse.status(), 200, 'New side extractor must be served');
  const sideSuggestion = (await sideResponse.json()).annotations.find(a => a.part === 'temple-right');
  assert.equal(sideSuggestion?.source, SOURCE, 'Restart backend with the corrected side extractor');
  assert.equal(sideSuggestion.approved, false);
  assert.ok(sideSuggestion.points.length >= 20 && sideSuggestion.points.length < 100,
    'Unexpected side contour; review before approving');
  assert.equal((await project()).revision, expectedRevision, 'Read-only extraction must not mutate the project');

  await page.goto(base);
  await page.locator('.project-nav').filter({hasText:id.slice(0, 6)}).first().click();
  await page.getByRole('button', {name:/generate model/i}).first().click();
  const frontPreview = page.getByLabel('Detected outlines on current front photo');
  await frontPreview.locator('polygon').first().waitFor();
  await frontPreview.screenshot({path:'artifacts/surface-fix/front-review-ui.png'});
  assert.equal((await project()).revision, expectedRevision, 'Stop if project changed before front approval');
  await page.getByRole('button', {name:'Approve outlines & generate draft', exact:true}).click();

  // App sends the user to the pending side review after saving the front.
  await page.getByRole('img', {name:'right arm preview'}).locator('polygon').waitFor();
  const afterFront = await project();
  assert.equal(afterFront.revision, expectedRevision + 1, 'Front approval must be the only first mutation');
  sameInputs(afterFront, before);
  for (const part of ['temple-right', 'temple-left']) {
    assert.deepEqual(afterFront.annotations.find(a => a.part === part),
      before.annotations.find(a => a.part === part), 'Front approval must preserve the old side evidence');
  }
  const frontParts = afterFront.annotations.filter(a => a.photo_id === FRONT_PHOTO);
  assert.ok(frontParts.some(a => a.part === 'front' && a.approved && a.source.includes('shadow-aware-v4')));
  const sidePreview = page.getByRole('img', {name:'right arm preview'});
  await sidePreview.screenshot({path:'artifacts/surface-fix/side-review-ui.png'});

  // Preserve the existing explicit symmetry decision. This project has no
  // independent left-side photo; mirroring is an assumption, not new evidence.
  const mirror = page.getByRole('checkbox', {name:/Also mirror this reviewed shape/});
  assert.equal(await mirror.isChecked(), false);
  assert.equal(afterFront.annotations.find(a => a.part === 'temple-left')?.source,
    'explicit-mirrored-side-assumption');
  await mirror.check();
  assert.equal((await project()).revision, expectedRevision + 1, 'Checkbox must not save before approval');
  await page.getByRole('button', {name:'Approve arm & regenerate', exact:true}).click();
  await page.getByText('Model generated', {exact:true}).waitFor({timeout:120000});

  const after = await project();
  assert.equal(after.revision, expectedRevision + 2, 'Only the front and side approvals may bump revision');
  sameInputs(after, before);
  assert.equal(after.stale, false);
  assert.equal(after.artifact?.revision, after.revision);
  assert.equal(after.artifact?.metadata?.engine_version, 4);
  assert.notEqual(after.artifact?.job_id, before.artifact?.job_id);
  assert.deepEqual(after.annotations.filter(a => a.photo_id !== FRONT_PHOTO && a.photo_id !== SIDE_PHOTO),
    before.annotations.filter(a => a.photo_id !== FRONT_PHOTO && a.photo_id !== SIDE_PHOTO));
  const right = after.annotations.find(a => a.part === 'temple-right');
  const left = after.annotations.find(a => a.part === 'temple-left');
  assert.equal(right?.source, `merchant-reviewed-side:${SOURCE}`);
  assert.equal(left?.source, `explicit-mirrored-side-assumption:${SOURCE}`);
  assert.deepEqual(right.points, sideSuggestion.points, 'Approved right contour must be the reviewed v4 suggestion');
  assert.deepEqual(left.points, right.points, 'Left arm must carry the explicit mirrored outline');
  assert.equal(right.approved, true);
  assert.equal(left.approved, true);

  await fs.writeFile(`artifacts/surface-fix/regeneration-${after.revision}-result.json`, JSON.stringify({
    projectId:id, beforeRevision:before.revision, afterRevision:after.revision,
    oldJob:before.artifact.job_id, newJob:after.artifact.job_id,
    source:SOURCE, sidePoints:right.points.length, mirror:'explicit assumption from existing project',
    checks:['current front analyzed and approved in UI','v4 right contour reviewed and approved in UI',
      'existing explicit mirror decision retained','photos, materials, measurements and category unchanged',
      'version 4 GLB generated from final revision']
  }, null, 2));
  console.log(`Regenerated reviewed surface project: revision ${after.revision}, job ${after.artifact.job_id}`);
} finally {
  await browser.close();
}
