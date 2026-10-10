import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
const result=JSON.parse(await fs.readFile('artifacts/public-metal-result.json','utf8'));
if(result.job.status!=='completed')throw new Error('Generate the reviewed reference first.');
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
try{
 await page.goto('http://127.0.0.1:5188');
 await page.locator('.project-nav').filter({hasText:'Zenni 137912 — reviewed photo traces'}).first().click();
 await page.getByRole('tab',{name:/Review \/ AR/}).click();
 await page.getByLabel('Interactive three-dimensional eyewear model').waitFor();
 await page.getByText('Loading exported model…').waitFor({state:'hidden'});
 await page.getByRole('button',{name:'Front',exact:true}).click();
 await page.locator('.mv-root').screenshot({path:'artifacts/qa-public-model.png'});
 await page.getByRole('button',{name:'Overlay',exact:true}).click();
 await page.locator('.mv-root').screenshot({path:'artifacts/qa-public-comparison.png'});
 console.log('Reviewed public model and comparison screenshots saved.');
}finally{await browser.close();}
