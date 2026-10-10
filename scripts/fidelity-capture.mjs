// Read-only pinned-asset comparison in the existing preview, never a project edit.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {chromium} from '@playwright/test';
const [mode='baseline',candidateDir='artifacts/fidelity/candidate']=process.argv.slice(2);
assert.ok(['baseline','candidate'].includes(mode));
const base='http://127.0.0.1:5188';
const id='d9cb85ea817b4a7d968e31135814cab4';
const out=path.resolve('artifacts/fidelity');
await fs.mkdir(out,{recursive:true});
let project,bytes;
if(mode==='baseline'){
 const response=await fetch(`${base}/api/projects/${id}`);assert.equal(response.status,200);
 project=await response.json();
 bytes=Buffer.from(await (await fetch(base+project.artifact.model_url)).arrayBuffer());
 await fs.writeFile(path.join(out,'baseline-project.json'),JSON.stringify(project,null,2),{flag:'wx'});
 await fs.writeFile(path.join(out,'baseline-model.glb'),bytes,{flag:'wx'});
}else{
 project=JSON.parse(await fs.readFile(path.join(out,'baseline-project.json'),'utf8'));
 const metadata=JSON.parse(await fs.readFile(path.join(candidateDir,'metadata.json'),'utf8'));
 bytes=await fs.readFile(path.join(candidateDir,'model.glb'));
 project.artifact={...project.artifact,metadata,job_id:'fidelity-candidate',model_url:'/api/artifacts/fidelity-candidate/model.glb'};
}
project.stale=false;
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 await page.route(`${base}/api/projects/${id}`,route=>route.fulfill({json:project}));
 await page.route(`${base}${project.artifact.model_url}`,route=>route.fulfill({body:bytes,contentType:'model/gltf-binary'}));
 await page.goto(base);
 await page.locator('.project-nav').filter({hasText:id.slice(0,6)}).first().click();
 await page.getByRole('tab',{name:/Review \/ AR/}).click();
 const canvas=page.getByLabel('Interactive three-dimensional eyewear model');
 for(const display of ['photo','clay','silhouette']){
  await page.getByLabel('Surface display').selectOption(display);
  await page.waitForTimeout(1000);
  await page.locator('.mv-metrics dd').first().waitFor({timeout:30000});
  await page.locator('.mv-stage-status').waitFor({state:'hidden'});
  for(const [view,label] of [['front','Front'],['side','Other side'],['threequarter','Orbit'],['top','Top']]){
   await page.getByRole('button',{name:label,exact:true}).click();await page.waitForTimeout(400);
   await canvas.screenshot({path:path.join(out,`${mode}-${view}-${display}.png`)});
   if(view==='threequarter'||view==='side'){
    const b=await canvas.boundingBox();await page.mouse.move(b.x+b.width*.52,b.y+b.height*.52);
    await page.mouse.wheel(0,-780);await page.waitForTimeout(500);
    await canvas.screenshot({path:path.join(out,`${mode}-${view}-${display}-closeup.png`)});
   }
  }
 }
 assert.deepEqual(errors,[]);
 await fs.writeFile(path.join(out,`${mode}-capture.json`),JSON.stringify({project:id,revision:project.revision,job:project.artifact.job_id,sha256:createHash('sha256').update(bytes).digest('hex'),errors},null,2));
 console.log(`${mode} captures passed`);
}finally{await browser.close();}
