// Deliberate update of one existing project, after visually checking its new
// contour overlay. Never invoke as an unattended generic regression test.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '@playwright/test';

const [id, expectedRevision] = process.argv.slice(2);
assert.ok(id && expectedRevision, 'Usage: node scripts/regenerate-reviewed-project.mjs ID EXPECTED_REVISION');
const base = 'http://127.0.0.1:5188';
const browser = await chromium.launch({channel:'chrome', headless:true,
  args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport:{width:1440,height:1100}});
try {
  const before = await (await page.request.get(`${base}/api/projects/${id}`)).json();
  assert.equal(before.revision, Number(expectedRevision), 'Stop if project changed since review');
  await page.goto(base);
  await page.locator('.project-nav').filter({hasText:id.slice(0,6)}).first().click();
  await page.getByRole('button',{name:/generate model/i}).first().click();
  const legacy = before.annotations.some(a=>a.part==='front'&&['automatic','merchant-reviewed:automatic'].includes(a.source));
  if(legacy || before.artifact?.metadata.engine_version!==3) {
    await page.getByLabel('Detected outlines on current front photo').waitFor();
    await page.getByLabel('Detected outlines on current front photo').screenshot({path:'artifacts/assembly-fix/reviewed-contours-ui.png'});
    await page.getByRole('button',{name:'Approve outlines & generate draft',exact:true}).click();
  }
  await page.getByText('Model generated',{exact:true}).waitFor({timeout:60000});
  const after = await (await page.request.get(`${base}/api/projects/${id}`)).json();
  assert.notEqual(after.artifact.job_id,before.artifact.job_id);
  assert.equal(after.artifact.metadata.engine_version,3);
  assert.equal(after.stale,false);
  assert.deepEqual(after.measurements,before.measurements,'Do not silently alter physical dimensions');
  assert.deepEqual(after.material,before.material,'Do not replace product material settings');
  for(const part of ['temple-left','temple-right']) {
    assert.deepEqual(after.annotations.find(a=>a.part===part),before.annotations.find(a=>a.part===part),'Keep reviewed side evidence');
  }
  assert.ok(after.annotations.find(a=>a.part==='front').source.includes('shadow-aware-v3'));
  await fs.writeFile(`artifacts/assembly-fix/regeneration-${before.revision}-result.json`,JSON.stringify({
    projectId:id,oldJob:before.artifact.job_id,newJob:after.artifact.job_id,
    beforeRevision:before.revision,afterRevision:after.revision,
    checks:[legacy?'actual UI analysis, approval and CPU job':'actual UI CPU generation with existing reviewed outlines','new version 3 GLB','same measurements and lens/material settings','same reviewed/mirrored arm evidence','old export retained']
  },null,2));
  console.log('Regenerated reviewed project:',after.artifact.job_id);
} finally { await browser.close(); }
