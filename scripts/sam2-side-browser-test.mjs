import {chromium,request} from '@playwright/test';
import assert from 'node:assert/strict';

// Run against a saved development upload; analysis saves diagnostics only.
// Never clicks approval, changes annotations or starts generation.
const projectId=process.env.SAM2_SIDE_PROJECT;
assert.ok(projectId,'Set SAM2_SIDE_PROJECT to a saved development project');
const api=process.env.SAM2_SIDE_API??'http://127.0.0.1:8018';
const frontend=process.env.SAM2_SIDE_FRONTEND??'http://127.0.0.1:5188';
const http=await request.newContext({baseURL:api});
const before=await (await http.get(`/api/projects/${projectId}`)).json();
const photo=before.photos.find(p=>p.view==='right'&&p.role==='fitting');
assert.ok(photo,'A right fitting photo is required');
const browser=await chromium.launch({channel:'chrome',headless:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto(frontend);
 await page.locator('.project-nav').filter({hasText:projectId.slice(0,6)}).click();
 const panel=page.locator('section.quick-draft').filter({has:page.getByRole('heading',{name:'Reconstruct the right arm',exact:true})});
 const responsePromise=page.waitForResponse(r=>r.url().includes(`/photos/${photo.id}/extract?backend=sam2`)&&r.request().method()==='POST',{timeout:150000});
 await panel.getByRole('button',{name:'Analyze side arm',exact:true}).click();
 const response=await responsePromise;
 assert.equal(response.status(),200);
 const result=await response.json();
 assert.equal(result.hybrid_detection.inference_ran,true);
 assert.equal(result.hybrid_detection.proposals_used,true);
 assert.equal(result.hybrid_detection.overlap_completion,true);
 assert.equal(result.detection_report.status,'review-required');
 assert.deepEqual(result.detection_report.missing_parts,[]);
 assert.ok(result.annotations.every(a=>!a.approved));
 await panel.getByText('AI assisted arm boundaries used · experimental',{exact:true}).waitFor();
 assert.equal(await panel.locator('polygon').count()>0,true);
 assert.equal(await panel.getByRole('button',{name:'Approve arm & regenerate',exact:true}).isEnabled(),true);
 await panel.screenshot({path:'artifacts/sam2/side-overlap-ui.png'});
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false);
 assert.deepEqual(errors,[]);
 const after=await (await http.get(`/api/projects/${projectId}`)).json();
 assert.deepEqual(after,before,'Analysis must not alter the saved project or approve geometry');
 console.log('Real SAM side overlap: usable draft, truthful inference, correct UI labels, mobile layout and unchanged project passed.');
}finally{await browser.close();await http.dispose();}
