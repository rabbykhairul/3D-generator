import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';

const out=new URL('../artifacts/',import.meta.url);
await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const context=await browser.newContext({viewport:{width:1440,height:1000}});
const page=await context.newPage();
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
const base='http://127.0.0.1:5188';
try{
 await page.goto(base);
 await page.getByRole('button',{name:'New project',exact:false}).click();
 await page.getByLabel('Project name').fill('Browser QA — missing evidence');
 await page.getByRole('button',{name:'Create project',exact:true}).click();
 await page.getByRole('heading',{name:'Browser QA — missing evidence'}).waitFor();
 await page.getByRole('button',{name:/Generate model/}).first().click();
 await page.getByText('Add a straight front photo in the Front slot to start.',{exact:true}).waitFor({timeout:30000});
 await page.screenshot({path:new URL('qa-correction.png',out).pathname,fullPage:true});

 await page.getByRole('button',{name:/Import sample/}).click();
 const modal=page.getByRole('dialog');
 const sample=modal.locator('.sample-item').filter({hasText:'Opaque Full Rim · synthetic'});
 await sample.getByRole('button',{name:'Import',exact:true}).click();
 await page.getByRole('heading',{name:'Opaque Full Rim — synthetic fixture'}).waitFor();
 await page.getByRole('button',{name:/Generate model/}).first().click();
 await page.getByText('Model generated',{exact:true}).waitFor({timeout:30000});
 await page.getByLabel('Interactive three-dimensional eyewear model').waitFor();
 await page.getByText('Loading exported model…').waitFor({state:'hidden',timeout:20000});
 assert.equal(await page.getByText(/Could not load the exported GLB/).count(),0);
 await page.getByLabel('Wireframe',{exact:true}).check();
 await page.getByLabel('Wireframe',{exact:true}).uncheck();
 await page.getByRole('button',{name:'Front',exact:true}).click();
 await page.screenshot({path:new URL('qa-model.png',out).pathname,fullPage:true});
 const href=await page.getByRole('link',{name:/Download GLB/}).getAttribute('href');
 const glb=await context.request.get(base+href);
 assert.equal(glb.status(),200);assert.equal((await glb.body()).subarray(0,4).toString(),'glTF');
 await page.getByRole('tab',{name:/Dimensions/}).click();
 await page.screenshot({path:new URL('qa-dimensions.png',out).pathname,fullPage:true});
 await page.getByRole('button',{name:'Advanced outline editor',exact:true}).click();
 await page.screenshot({path:new URL('qa-outlines.png',out).pathname,fullPage:true});
 await page.reload();
 await page.getByRole('heading',{name:'Opaque Full Rim — synthetic fixture'}).waitFor();
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:new URL('qa-mobile.png',out).pathname,fullPage:true});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2);
 assert.equal(overflow,false,'Mobile page should not overflow horizontally');
 assert.deepEqual(errors,[]);
 await fs.writeFile(new URL('browser-results.json',out),JSON.stringify({passed:true,checks:['create','missing-evidence failure','synthetic sample import','real CPU generation','GLB WebGL load','wireframe','download','outline screen','dimensions screen','reload persistence','mobile overflow'],pageErrors:errors},null,2));
 console.log('Browser integration checks passed. Screenshots in artifacts/.');
}catch(error){
 await page.screenshot({path:new URL('qa-failure.png',out).pathname,fullPage:true});
 await fs.writeFile(new URL('qa-failure.txt',out),(await page.locator('body').innerText())+'\nERRORS\n'+JSON.stringify(errors));
 throw error;
}finally{await browser.close();}
