/* Read-only actual TS/React component probes in jsdom; no live product/API. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const Module = require('node:module');
const { spawnSync } = require('node:child_process');
const ART = __dirname;
const ROOT = path.resolve(ART,'../../../..');
const FRONT = path.join(ROOT,'frontend');
const frontRequire = Module.createRequire(path.join(FRONT,'package.json'));
const ts = frontRequire('typescript');
const names = ['frontend/src/components/AnalysisCorrection.tsx','frontend/src/components/NativeForecast.tsx',
  'frontend/src/data/correctionApi.ts','frontend/src/data/correctionDraft.ts','frontend/src/data/nativeForecast.ts',
  'frontend/src/data/repository.ts','frontend/src/app/AnalysisContext.tsx','frontend/src/styles/correction.css',
  'frontend/src/test/correctionFixtures.ts','src/scoring/corrected_rule_rebuild.py'];
function pins() { return Object.fromEntries(names.map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT,name))).digest('hex')])); }
const before = pins();
for (const name of names) { const target=path.join(ART,'sources',name); fs.mkdirSync(path.dirname(target),{recursive:true}); fs.copyFileSync(path.join(ROOT,name),target); }
// Transpile the actual checked-in TypeScript in memory, retaining normal
// package resolution from its original path. No application files are written.
for (const extension of ['.ts','.tsx']) Module._extensions[extension]=(module,filename)=> {
  const source=fs.readFileSync(filename,'utf8');
  const output=ts.transpileModule(source,{fileName:filename,compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;
  module._compile(output,filename);
};
Module._extensions['.css']=()=>{};
const { JSDOM }=frontRequire('jsdom');
const dom=new JSDOM('<!doctype html><html><body></body></html>',{url:'http://localhost/'});
for (const name of ['window','document','HTMLElement','HTMLInputElement','Node','Event','MouseEvent','KeyboardEvent','sessionStorage','localStorage']) globalThis[name]=name==='window'?dom.window:dom.window[name];
Object.defineProperty(globalThis,'navigator',{configurable:true,value:dom.window.navigator});
globalThis.IS_REACT_ACT_ENVIRONMENT=true;
const React=frontRequire('react');
const {render,screen,waitFor,cleanup,act}=frontRequire('@testing-library/react');
const {AnalysisCorrection}=frontRequire('./src/components/AnalysisCorrection.tsx');
const {correctionApi,CorrectionCommandSchema}=frontRequire('./src/data/correctionApi.ts');
const {commandBase,amendInitial}=frontRequire('./src/data/correctionDraft.ts');
const fixtures=frontRequire('./src/test/correctionFixtures.ts');

(async()=> {
  const source=fixtures.reviewFixture(); source.is_current=false;
  correctionApi.session=async()=>({required:true,authenticated:true});
  correctionApi.source=async()=>source;
  correctionApi.list=async()=>[fixtures.correctionStatus('COMPLETED')];
  const followed=[];
  const run={summary:{id:fixtures.analysisId},analysisGeneration:fixtures.generation,analysisRevision:0,pointAnalysis:null};
  const view=render(React.createElement(AnalysisCorrection,{run,onCompleted:token=>followed.push(token),onSelectEvent:()=>{}}));
  await screen.findByText(/This generation has been superseded/);
  await act(async()=>{await new Promise(resolve=>setTimeout(resolve,0));});
  const historical={displayed_generation:fixtures.generation,source_is_current:false,
    automatic_navigation_calls:followed.length,automatic_navigation_target:followed[0]??null,
    issue_reproduced:followed.length>0};
  view.unmount(); cleanup();
  const current=fixtures.reviewFixture();
  const command=commandBase(current,'EXPLICIT-FRONTEND-REVIEW','First reviewed line.\nSecond reviewed line.','Original source frame 10');
  command.initial_state_amendment={expected_initial_state_sha256:current.initial_state_sha256,
    replacement_initial_state:amendInitial(current,{p1:0,p2:1,server:1,serve:1})};
  const parsed=CorrectionCommandSchema.safeParse(command);
  const complete={...command,analysis_id:fixtures.analysisId,reviewer_id:'account_EXPLICIT_REVIEW_SOFTWARE_FIXTURE'};
  const commandPath=path.join(ART,'frontend_command.json'); fs.writeFileSync(commandPath,JSON.stringify(complete,null,2)+'\n');
  const native=spawnSync('C:\\Users\\ac-98\\AppData\\Local\\Programs\\Python\\Python313\\python.exe',
    [path.join(ART,'probe_native_command.py'),commandPath],{cwd:ROOT,encoding:'utf8'});
  if(native.status!==0) throw new Error(native.stderr||native.stdout);
  const nativeResult=JSON.parse(native.stdout.trim());
  const multiline={frontend_schema_accepted:parsed.success,reason_contains_newline:command.reason.includes('\n'),
    ...nativeResult,issue_reproduced:parsed.success&&!nativeResult.native_accepted};
  const after=pins();
  const receipt={scope:'ACTUAL_TS_DRAFT_ZOD_NATIVE_COMMAND_ADMISSION_AND_REACT_HISTORY_EFFECT_IN_JSDOM; MOCK_HTTP_ONLY',
    historical_navigation:historical,multiline_reason:multiline,source_before:before,source_after:after,
    source_unchanged:JSON.stringify(before)===JSON.stringify(after)};
  fs.writeFileSync(path.join(ART,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify({historical_navigation:historical,multiline_reason:multiline,source_unchanged:receipt.source_unchanged}));
  dom.window.close();
  process.exitCode=receipt.source_unchanged?0:1;
})().catch(error=>{console.error(error);dom.window.close();process.exitCode=1;});
