import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import ts from 'typescript';

const require=createRequire(import.meta.url);
const source=(await fs.readFile('frontend/src/components/PhotoGuide.tsx','utf8')).replace("import './photoGuide.css';",'');
const compiled=ts.transpile(source,{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.React});
const code=`import React from ${JSON.stringify(pathToFileURL(require.resolve('react')).href)};\n${compiled}`;
const {default:PhotoGuide}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const chosen=[];
const tree=PhotoGuide({onChoose:view=>chosen.push(view)});
const html=renderToStaticMarkup(tree);
assert.match(html,/<details[^>]*open=""/,'upload examples are visible by default');
assert.equal((html.match(/<img /g)||[]).length,3);
assert.match(html,/explicitly approve mirroring/);
assert.match(html,/same product and color/);
assert.match(html,/Automatic suggestions always need approval/);
function buttons(node){
 if(!React.isValidElement(node))return [];
 return [...(node.type==='button'?[node]:[]),...React.Children.toArray(node.props.children).flatMap(buttons)];
}
for(const button of buttons(tree))button.props.onClick();
assert.deepEqual(chosen,['front','right','front-left','inner']);
assert.ok(buttons(PhotoGuide({onChoose:()=>{},disabled:true})).every(button=>button.props.disabled));
for(const name of ['front','side','angle'])assert.ok((await fs.stat(`frontend/public/upload-guide/${name}.png`)).size>1000);
console.log('Upload guide: rendered examples, accessible labels, view actions, busy state and local assets passed.');
