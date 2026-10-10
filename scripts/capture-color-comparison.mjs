import {chromium} from '@playwright/test';
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
try{
 await page.goto('http://127.0.0.1:5188');
 await page.locator('.project-nav').filter({hasText:process.argv[2]}).first().click();
 await page.getByRole('tab',{name:/Review \/ AR/}).click();
 await page.getByLabel('Interactive three-dimensional eyewear model').waitFor();
 await page.getByText('Loading exported model…').waitFor({state:'hidden'});
 await page.getByRole('button',{name:'Front',exact:true}).click();
 await page.getByLabel('Interactive three-dimensional eyewear model').screenshot({path:'artifacts/photo-color-preview.png'});
 await page.getByLabel('Surface display').selectOption('material');
 await page.getByText('Loading exported model…').waitFor({state:'hidden'});
 await page.getByRole('button',{name:'Front',exact:true}).click();
 await page.getByLabel('Interactive three-dimensional eyewear model').screenshot({path:'artifacts/lit-color-preview.png'});
 console.log('Photo-color and studio-lighting screenshots saved.');
}finally{await browser.close();}
