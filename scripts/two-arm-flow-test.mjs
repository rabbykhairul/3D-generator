import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});const base='http://127.0.0.1:5188';let id;
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 const image=await fs.readFile(process.argv[2]);
 let p=await (await page.request.post(base+'/api/samples/synthetic-opaque-full-rim/import')).json();id=p.id;
 await page.request.patch(base+'/api/projects/'+id,{data:{name:'Two-arm sequence QA',annotations:[]}});
 for(const view of ['left','right','inner'])await page.request.post(base+'/api/projects/'+id+'/photos',{multipart:{view,file:{name:view+'.png',mimeType:'image/png',buffer:image}}});
 await page.goto(base);await page.getByRole('heading',{name:'Two-arm sequence QA',exact:true}).waitFor();
 assert.equal(await page.locator('.capture-slot').count(),5);
 await page.getByRole('button',{name:/Generate model/}).first().click();
 await page.getByRole('img',{name:'Detected outlines on current front photo'}).waitFor();
 await page.getByRole('button',{name:'Approve outlines & generate draft'}).click();
 for(const side of ['left','right']){
  const panel=page.locator('section.quick-draft').filter({has:page.getByRole('heading',{name:`Reconstruct the ${side} arm`,exact:true})});
  await panel.getByText(/One visible arm was separated/).waitFor();
  p=await (await page.request.get(base+'/api/projects/'+id)).json();assert.equal(p.artifact,null,'Front approval must not skip side review and generate early');
  await panel.getByRole('button',{name:'Approve arm & regenerate'}).click();
 }
 await page.getByText('Model generated',{exact:true}).waitFor({timeout:30000});
 p=await (await page.request.get(base+'/api/projects/'+id)).json();
 assert.ok(p.artifact.metadata.parts.includes('temple_left')&&p.artifact.metadata.parts.includes('temple_right'));
 assert.equal(p.artifact.metadata.provenance.arms.right,'approved side outline');
 const colors=page.getByLabel('Surface display');
 await colors.waitFor();assert.equal(await colors.inputValue(),'photo');
 await page.getByText('Loading exported model…').waitFor({state:'hidden'});
 await colors.selectOption('material');await page.getByText('Loading exported model…').waitFor({state:'hidden'});
 await colors.selectOption('photo');await page.getByText('Loading exported model…').waitFor({state:'hidden'});
 await page.getByRole('tab',{name:/Photos/}).click();
 const right=p.photos.find(x=>x.view==='right');
 await page.getByLabel('Photo view '+right.id.slice(0,6)).selectOption('inner-right');
 await page.getByRole('heading',{name:'Reconstruct the inner-right arm'}).waitFor();
 await page.getByRole('button',{name:/Regenerate model/}).first().click();
 const inner=page.locator('section.quick-draft').filter({has:page.getByRole('heading',{name:'Reconstruct the inner-right arm',exact:true})});
 await inner.getByText(/One visible arm was separated/).waitFor();
 await inner.getByRole('button',{name:'Approve arm & regenerate'}).click();
 await page.getByText('Model generated',{exact:true}).waitFor({timeout:30000});
 for(let n=0;n<60;n++){p=await (await page.request.get(base+'/api/projects/'+id)).json();if(!p.stale)break;await new Promise(r=>setTimeout(r,250));}
 assert.equal(p.stale,false);assert.equal(p.artifact.metadata.fit.side.right.source_photo_id,right.id);
 assert.deepEqual(errors,[]);
 await page.screenshot({path:'artifacts/two-arm-flow.png',fullPage:true});
 await fs.writeFile('artifacts/two-arm-results.json',JSON.stringify({passed:true,checks:['front approval routes to left then right review','both arms actually exported','inner upload accepted as reference','relabel right to inner-right','inner-right geometry export','photo-color mode toggles without errors'],errors},null,2));
 console.log('Two-arm, inner-view, relabel and color-preview workflow passed.');
}finally{if(id)await page.request.delete(base+'/api/projects/'+id);await browser.close();}
