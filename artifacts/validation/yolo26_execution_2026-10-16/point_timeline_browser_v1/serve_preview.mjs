import {readFile} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';

const directory=path.dirname(fileURLToPath(import.meta.url));
const workspace=path.resolve(directory,'../../../..');
const frontend=path.join(workspace,'frontend');
const fixture=path.resolve(process.argv[2] ?? '');
if(!fixture.startsWith(workspace+path.sep) || !fixture.endsWith('.json')) throw new Error('Require a saved workspace JSON fixture');
await readFile(fixture);
const {createServer}=await import(pathToFileURL(path.join(frontend,'node_modules/vite/dist/node/index.js')).href);
const server=await createServer({configFile:path.join(frontend,'vite.config.ts'),root:frontend,
  plugins:[{name:'private-point-preview',configureServer(vite){
    // Install before Vite's HTML fallback so this isolated harness does not
    // enter the product router. Production source and routing stay unchanged.
    vite.middlewares.use('/point-fixture.json',async (_request,response,next)=>{
      try {response.setHeader('Content-Type','application/json');response.end(await readFile(fixture));}
      catch(error){next(error);}
    });
    vite.middlewares.use('/point-preview-entry.js',async (_request,response,next)=>{
      try {response.setHeader('Content-Type','application/javascript');response.end(await readFile(path.join(directory,'preview_entry.js')));}
      catch(error){next(error);}
    });
    vite.middlewares.use('/point-preview',async (_request,response,next)=>{
      try {const html=await readFile(path.join(directory,'preview.html'),'utf8');
        response.setHeader('Content-Type','text/html');response.end(await vite.transformIndexHtml('/point-preview',html));}
      catch(error){next(error);}
    });
  }}],server:{host:'127.0.0.1',port:5197,strictPort:true,fs:{allow:[workspace]}}});
await server.listen();
console.log(JSON.stringify({url:'http://127.0.0.1:5197/point-preview',fixture,pid:process.pid,synthetic:true}));
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,async()=>{await server.close();process.exit(0);});
