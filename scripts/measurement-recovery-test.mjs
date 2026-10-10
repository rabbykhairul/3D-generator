import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1000}});const base='http://127.0.0.1:5188';let id;
try{
 let p=await (await page.request.post(base+'/api/samples/synthetic-opaque-full-rim/import')).json();id=p.id;
 const measurements={frame_width:{value:.2,source:'manual'},lens_width:{value:.08,source:'inferred'},lens_height:{value:.06,source:'inferred'},bridge_width:{value:1,source:'inferred'},temple_length:{value:140,source:'default'}};
 await page.request.patch(base+'/api/projects/'+id,{data:{name:'Measurement recovery QA',measurements}});
 await page.goto(base);await page.getByRole('heading',{name:'Measurement recovery QA',exact:true}).waitFor();
 await page.getByRole('button',{name:/Generate model/}).first().click();
 await page.getByRole('heading',{name:'Dimensions & materials',exact:true}).waitFor();
 assert.ok(await page.getByText(/frame width is 0.2 mm/).count());
 assert.equal((await (await page.request.get(base+'/api/projects/'+id)).json()).measurements.frame_width.value,.2);
 await page.getByRole('button',{name:'Use approximate draft sizes (140 mm)',exact:true}).click();
 assert.equal(await page.locator('#measure-frame_width').inputValue(),'140');
 assert.equal((await (await page.request.get(base+'/api/projects/'+id)).json()).measurements.frame_width.value,.2,'Reset must stay unsaved until explicitly saved');
 await page.getByRole('button',{name:'Save dimensions & appearance',exact:true}).click();
 await page.getByText('All changes saved',{exact:true}).waitFor();
 p=await (await page.request.get(base+'/api/projects/'+id)).json();
 assert.equal(p.measurements.frame_width.value,140);assert.equal(p.measurements.frame_width.source,'default');
 assert.ok(2*p.measurements.lens_width.value+p.measurements.bridge_width.value<=140*1.02);
 await page.getByRole('button',{name:/Generate model/}).first().click();
 await page.getByText('Model generated',{exact:true}).waitFor({timeout:30000});
 p=await (await page.request.get(base+'/api/projects/'+id)).json();assert.equal(p.stale,false);
 assert.equal(p.artifact.metadata.dimensions_mm.frame_width,140);
 await fs.writeFile('artifacts/measurement-recovery-results.json',JSON.stringify({passed:true,checks:['exact reported invalid values','preflight units message','no silent conversion','explicit reset then save','consistent inferred dimensions','actual successful GLB generation']},null,2));
 console.log('Measurement recovery and generation passed.');
}finally{if(id)await page.request.delete(base+'/api/projects/'+id);await browser.close();}
