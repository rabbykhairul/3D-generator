import {chromium,request} from '@playwright/test';
import assert from 'node:assert/strict';

// Inspect saved development photos without approving or changing the project.
const ids=(process.env.SAM2_FRONT_PROJECTS??'').split(',').filter(Boolean);
assert.ok(ids.length,'Set SAM2_FRONT_PROJECTS to saved development project IDs');
const api=process.env.SAM2_FRONT_API??'http://127.0.0.1:8018';
const frontend=process.env.SAM2_FRONT_FRONTEND??'http://127.0.0.1:5188';
const http=await request.newContext({baseURL:api});
const browser=await chromium.launch({channel:'chrome',headless:true});
try{
 for(const id of ids){
  const before=await (await http.get(`/api/projects/${id}`)).json();
  const photo=before.photos.find(p=>p.view==='front'&&p.role==='fitting');
  assert.ok(photo,'A front fitting photo is required');
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(frontend);
  await page.locator('.project-nav').filter({hasText:id.slice(0,6)}).click();
  const panel=page.locator('div.quick-draft');
  await panel.getByLabel('Front photo analysis',{exact:true}).selectOption('sam2');
  const pending=page.waitForResponse(r=>r.url().includes(`/photos/${photo.id}/extract?backend=sam2`)&&r.request().method()==='POST',{timeout:150000});
  await panel.getByRole('button',{name:'Analyze current front photo',exact:true}).click();
  const response=await pending;assert.equal(response.status(),200);
  const result=await response.json();
  assert.equal(result.hybrid_detection.inference_ran,true);
  assert.equal(result.hybrid_detection.proposals_used,true);
  assert.equal(result.detection_report.status,'review-required');
  assert.deepEqual(result.detection_report.missing_parts,[]);
  assert.ok(result.annotations.every(a=>!a.approved));
  assert.ok(['front','lens-left','lens-right'].every(part=>result.annotations.some(a=>a.part===part)));
  await panel.getByText('AI assisted lens boundaries used · experimental',{exact:true}).waitFor();
  assert.equal(await panel.locator('polygon').count()>=3,true);
  await panel.screenshot({path:`artifacts/sam2/front-${id.slice(0,6)}-ui.png`});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false);
  assert.deepEqual(errors,[]);
  const after=await (await http.get(`/api/projects/${id}`)).json();
  assert.deepEqual(after,before,'Analysis must leave saved outlines and approvals unchanged');
  console.log(`Real SAM front ${id.slice(0,6)}: both optical windows, UI, mobile layout and unchanged project passed.`);
  await page.close();
 }
}finally{await browser.close();await http.dispose();}
