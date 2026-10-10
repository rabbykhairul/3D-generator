import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
try{
 await page.goto('http://127.0.0.1:5188');
 await page.locator('.project-nav').filter({hasText:'Olive frame — occlusion fix review'}).first().click();
 await page.getByRole('tab',{name:/Review \/ AR/}).click();
 await page.getByLabel('Interactive three-dimensional eyewear model').waitFor();
 await page.getByText('Loading exported model…').waitFor({state:'hidden'});
 await page.getByRole('button',{name:'Front',exact:true}).click();
 await page.locator('.mv-root').screenshot({path:'artifacts/occluded-reference/model-front.png'});
 await page.getByRole('button',{name:'Orbit',exact:true}).click();
 await page.locator('.mv-root').screenshot({path:'artifacts/occluded-reference/model-orbit.png'});
 assert.ok(await page.getByText(/Incomplete reconstruction: unverified arms are omitted/).count());
 console.log('Corrected real-photo GLB loaded; incomplete status verified.');
}finally{await browser.close();}
