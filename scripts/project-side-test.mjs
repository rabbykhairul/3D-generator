import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const sidePath=process.argv[2];if(!sidePath)throw new Error('Pass a side photograph path.');
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});const base='http://127.0.0.1:5188';
let id;
try{
 let p=await (await page.request.post(base+'/api/samples/synthetic-opaque-full-rim/import')).json();id=p.id;
 await page.request.patch(base+'/api/projects/'+id,{data:{name:'Side and removal regression'}});
 await page.request.post(base+'/api/projects/'+id+'/photos',{multipart:{view:'left',file:{name:'side.png',mimeType:'image/png',buffer:await fs.readFile(sidePath)}}});
 await page.goto(base);await page.getByRole('heading',{name:'Side and removal regression',exact:true}).waitFor();
 await page.getByRole('button',{name:/Generate model/}).first().click();
 await page.getByText(/One visible arm was separated/).waitFor();
 const mirror=page.getByRole('checkbox',{name:/Also mirror/});assert.equal(await mirror.isChecked(),false);
 await page.screenshot({path:'artifacts/side-reference/side-review-ui.png',fullPage:true});
 await page.getByRole('button',{name:'Approve arm & regenerate'}).click();
 await page.getByText('Model generated',{exact:true}).waitFor({timeout:30000});
 p=await (await page.request.get(base+'/api/projects/'+id)).json();
 assert.ok(p.artifact.metadata.parts.includes('temple_left'));assert.ok(!p.artifact.metadata.parts.includes('temple_right'));
 await page.getByRole('tab',{name:/Photos/}).click();
 await page.getByRole('button',{name:'Analyze side arm'}).click();await mirror.waitFor();await mirror.check();
 await page.getByRole('button',{name:'Approve arm & regenerate'}).click();
 // Poll the newly submitted revision, not the previous banner.
 for(let n=0;n<60;n++) {p=await (await page.request.get(base+'/api/projects/'+id)).json();if(!p.stale&&p.artifact.metadata.parts.includes('temple_right'))break;await new Promise(r=>setTimeout(r,250));}
 assert.ok(p.artifact.metadata.parts.includes('temple_right'));
 assert.match(p.artifact.metadata.provenance.arms.right,/symmetry/);
 await page.getByRole('button',{name:`Remove Side and removal regression ${id.slice(0,6)}`,exact:true}).click();
 await page.getByRole('button',{name:'Cancel',exact:true}).click();
 assert.equal((await (await page.request.get(base+'/api/projects')).json()).some(x=>x.id===id),true);
 await page.getByRole('button',{name:`Remove Side and removal regression ${id.slice(0,6)}`,exact:true}).click();
 await page.getByRole('button',{name:'Remove project',exact:true}).click();
 await page.getByRole('dialog',{name:'Remove saved project',exact:true}).waitFor({state:'hidden'});
 assert.equal((await (await page.request.get(base+'/api/projects')).json()).some(x=>x.id===id),false);
 await page.reload();await page.getByRole('button',{name:'Removed projects / Restore'}).click();
 const row=page.locator('.trash-list > div').filter({hasText:id.slice(0,6)});
 await row.getByRole('button',{name:'Restore',exact:true}).click();await row.waitFor({state:'hidden'});
 assert.equal((await (await page.request.get(base+'/api/projects')).json()).some(x=>x.id===id),true);
 await fs.writeFile('artifacts/side-reference/browser-results.json',JSON.stringify({passed:true,checks:['real thumbnail single-arm extraction','review required','no automatic mirror','reviewed arm exported','explicit mirror metadata','cancel removal','remove active project','reload','restore preserves project']},null,2));
 console.log('Side workflow and project removal/restore passed.');
}finally{if(id)await page.request.delete(base+'/api/projects/'+id);await browser.close();}
