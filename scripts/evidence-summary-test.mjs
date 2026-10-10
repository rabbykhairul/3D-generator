import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import ts from 'typescript';

const require=createRequire(import.meta.url);
const source=(await fs.readFile('frontend/src/components/EvidenceSummary.tsx','utf8')).replace("import './evidenceSummary.css';",'');
const compiled=ts.transpile(source,{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.React});
const code=`import React from ${JSON.stringify(pathToFileURL(require.resolve('react')).href)};\n${compiled}`;
const {default:Summary}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const project={photos:[],artifact:{metadata:{multiview:{views:[
 {photo_id:'f',view:'front',status:'used',usage:'Approved outlines'},
 {photo_id:'a',view:'front-left',status:'accepted',reason:'appearance_consistent'},
 {photo_id:'i',view:'inner',status:'review',reason:'ambiguous'},
],limitations:['Depth remains inferred']}}}};
const html=renderToStaticMarkup(React.createElement(Summary,{project}));
assert.match(html,/1 of 3 reported photos applied/);
assert.match(html,/Checked, not applied/);
assert.match(html,/Needs review/);
assert.match(html,/Depth remains inferred/);
const old=renderToStaticMarkup(React.createElement(Summary,{project:{photos:[]}}));
assert.match(old,/no per-photo use report/);
console.log('Evidence summary: actual use counts, consistency-only label, exclusions and legacy fallback passed.');
