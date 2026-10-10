// Run against a separate API/data directory to leave merchant projects intact.
import {chromium,request} from '@playwright/test';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';

const frontend=process.env.HYBRID_TEST_FRONTEND??'http://127.0.0.1:5188';
const api=process.env.HYBRID_TEST_API??'http://127.0.0.1:18018';
const manifest=JSON.parse(await fs.readFile('artifacts/segmenter-evaluation/medium.json','utf8'));
const photo=process.env.HYBRID_TEST_IMAGE?{path:process.env.HYBRID_TEST_IMAGE}:manifest.cases.find(c=>c.view==='front');
assert.ok(photo,'Saved front test photograph is required');
const http=await request.newContext({baseURL:api});
const created=await http.post('/api/projects',{data:{name:'Hybrid browser smoke',category:'opaque-full-rim'}});
assert.equal(created.status(),201);
const project=await created.json();
const uploaded=await http.post(`/api/projects/${project.id}/photos`,{multipart:{view:'front',role:'fitting',file:{name:'front.png',mimeType:'image/png',buffer:await fs.readFile(photo.path)}}});
assert.equal(uploaded.status(),200);
const browser=await chromium.launch({channel:'chrome',headless:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',async route=>{
  const original=new URL(route.request().url());
  const response=await route.fetch({url:api+original.pathname+original.search,timeout:60000});
  await route.fulfill({response});
 });
 await page.goto(frontend);
 const selector=page.getByLabel('Front photo analysis',{exact:true});
 await selector.waitFor();
 assert.equal(await selector.inputValue(),'sam2','SAM 2.1 should be selected on first load');
 const analysis=page.waitForResponse(r=>r.url().includes('/extract?backend=sam2')&&r.request().method()==='POST',{timeout:120000});
 await page.getByRole('button',{name:'Analyze current front photo',exact:true}).click();
 const response=await analysis;
 assert.equal(response.status(),200);
 const result=await response.json();
 assert.equal(result.hybrid_detection.inference_ran,true);
 assert.equal(result.hybrid_detection.proposal_backend,'sam2.1-hiera-small');
 assert.equal(result.hybrid_detection.proposals_used,true);
 await page.getByText('AI assisted lens boundaries used · experimental',{exact:true}).waitFor();
 assert.ok(result.annotations.every(a=>!a.approved));
 await page.locator('div.quick-draft').screenshot({path:'artifacts/sam2-ui-desktop.png'});
 await selector.selectOption('heuristics');
 assert.equal(await page.getByLabel('Detected outlines on current front photo').count(),0,'Changing backend clears stale suggestions');
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false);
 await page.screenshot({path:'artifacts/sam2-ui-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);
 const stored=await (await http.get(`/api/projects/${project.id}`)).json();
 assert.deepEqual(stored.annotations,[]);
 console.log('Hybrid UI: real CPU inference, provenance, stale-result reset and mobile checks passed.');
}finally{await browser.close();await http.dispose();}
