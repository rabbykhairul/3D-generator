import fs from 'node:fs/promises';
import {chromium} from '@playwright/test';
const folder='artifacts/clear-material';
const source=JSON.parse(await fs.readFile(`${folder}/project.json`,'utf8'));
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
try {
 for(const label of ['before','after']) {
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  const metadata=JSON.parse(await fs.readFile(`${folder}/${label}/metadata.json`,'utf8'));
  const project={...source,stale:false,artifact:{job_id:'material-qa',revision:source.revision,model_url:'/api/artifacts/material-qa/model.glb',metadata}};
  await page.route('**/api/**',async route=>{
   const u=new URL(route.request().url());
   if(u.pathname.endsWith('/model.glb')) return route.fulfill({body:await fs.readFile(`${folder}/${label}/model.glb`),contentType:'model/gltf-binary'});
   const photo=source.photos.find(p=>u.pathname.includes(p.id));
   if(photo) return route.fulfill({body:await fs.readFile(photo.path),contentType:'image/png'});
   return route.fulfill({json:u.pathname==='/api/projects'?[project]:project});
  });
  await page.goto('http://127.0.0.1:5199');
  await page.locator('.project-nav').filter({hasText:source.id.slice(0,6)}).first().click();
  await page.getByRole('tab',{name:/Review \/ AR/}).click();
  const canvas=page.getByLabel('Interactive three-dimensional eyewear model');
  await canvas.waitFor();
  await page.locator('.mv-stage-status').waitFor({state:'hidden'});
  await page.getByRole('button',{name:'Front',exact:true}).click();
  await page.waitForTimeout(1000);
  await canvas.screenshot({path:`${folder}/${label}-preview.png`});
  await page.close();
 }
} finally {await browser.close();}
