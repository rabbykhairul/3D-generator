import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const base='http://127.0.0.1:5188';
const out=new URL('../artifacts/',import.meta.url);
const face=new URL('grace_hopper.jpg',out).pathname;
const hardware=process.env.AR_HARDWARE==='1';
const browser=await chromium.launch({channel:'chrome',headless:true,args:hardware?[]:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const context=await browser.newContext({viewport:{width:1440,height:1100}});
const page=await context.newPage();
page.setDefaultTimeout(60000);
const errors=[];const cameraPosts=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(r.method()==='POST')cameraPosts.push(r.url());});
await page.addInitScript(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Test denial','NotAllowedError');};});
try{
 const projects=await (await context.request.get(base+'/api/projects')).json();
 const project=projects.find(p=>p.artifact&&p.category==='opaque-full-rim');
 assert.ok(project,'Generate a sample first');
 await page.goto(base);
 await page.locator('.project-nav').filter({hasText:project.name}).first().click();
 await page.getByRole('tab',{name:/Review \/ AR/}).click();
 await page.locator('.viewer-switch').getByRole('button',{name:'Webcam AR'}).click();
 if(process.env.AR_LOW==='1')await page.locator('.mv-header-controls select').selectOption('low');
 await page.getByRole('button',{name:'Start camera',exact:true}).click();
 await page.getByText('Camera permission was denied. You can upload a local photo instead.').waitFor();
 console.log('Camera denial handled.');
 await page.locator('.mv-ar-actions input[type=file]').setInputFiles(face);
 await page.getByText('Face aligned · approximate placement',{exact:false}).waitFor({timeout:90000});
 await page.locator('.mv-tracking-status small').waitFor();
 await page.locator('.mv-ar-stage').screenshot({path:new URL('qa-ar-photo.png',out).pathname});
 const stillPerformance=await page.locator('.mv-tracking-status').innerText();
 console.log(stillPerformance);
 await page.getByLabel('Preview framing').selectOption('contain');
 await page.locator('.mv-ar-stage').screenshot({path:new URL('qa-ar-photo-contained.png',out).pathname});
 // Synthetic stream made from a public fixture, never the user's camera.
 const dataUri='data:image/jpeg;base64,'+(await fs.readFile(face)).toString('base64');
 await page.evaluate(async src=>{
  window.__stoppedTracks=0;
  window.__showTestFace=true;
  navigator.mediaDevices.getUserMedia=async()=>{
   const img=new Image();img.src=src;await img.decode();
   const canvas=document.createElement('canvas');canvas.width=640;canvas.height=640;
   const ctx=canvas.getContext('2d');ctx.fillStyle='#111';ctx.fillRect(0,0,640,640);ctx.drawImage(img,64,0,512,640);
   const stream=canvas.captureStream(15);
   const timer=setInterval(()=>{ctx.fillStyle='#111';ctx.fillRect(0,0,640,640);if(window.__showTestFace)ctx.drawImage(img,64,0,512,640);},65);
   for(const track of stream.getTracks()){const stop=track.stop.bind(track);track.stop=()=>{window.__stoppedTracks++;clearInterval(timer);stop();};}
   return stream;
  };
 },dataUri);
 await page.getByRole('button',{name:'Start camera',exact:true}).click();
 await page.getByText('Face tracked · approximate placement',{exact:false}).waitFor({timeout:90000});
 await page.waitForFunction(()=>/\b[1-9]\d* tracking updates/.test(document.querySelector('.mv-tracking-status')?.textContent||''));
 const livePerformance=await page.locator('.mv-tracking-status').innerText();
 await page.locator('.mv-ar-stage').screenshot({path:new URL('qa-ar-live-fixture.png',out).pathname});
 await page.evaluate(()=>{window.__showTestFace=false;});
 await page.getByText('Move your face into view',{exact:false}).waitFor();
 await page.evaluate(()=>{window.__showTestFace=true;});
 await page.getByText('Face tracked · approximate placement',{exact:false}).waitFor();
 await page.getByRole('button',{name:'Stop camera',exact:true}).click();
 assert.ok(await page.evaluate(()=>window.__stoppedTracks)>0);
 assert.equal(cameraPosts.length,0,'AR must not upload face data');
 assert.deepEqual(errors,[]);
 await fs.writeFile(new URL(hardware?'ar-hardware-results.json':'ar-results.json',out),JSON.stringify({passed:true,renderer:hardware?'Chrome default GPU selection':'forced SwiftShader software renderer',quality:process.env.AR_LOW==='1'?'low':'balanced',stillPerformance,livePerformance,checks:['camera denial','real MediaPipe still-photo landmarks','contain framing','synthetic webcam stream tracking','tracking loss and recovery','stop releases tracks','zero AR POST uploads'],limitations:['No user camera accessed','Synthetic stream uses a single public portrait','Headless Chrome, not a mobile benchmark'],pageErrors:errors},null,2));
 console.log('AR browser checks passed.');
}catch(error){await page.screenshot({path:new URL('qa-ar-failure.png',out).pathname,fullPage:true});await fs.writeFile(new URL('qa-ar-failure.txt',out),(await page.locator('body').innerText())+'\n'+JSON.stringify(errors));throw error;}
finally{await browser.close();}
