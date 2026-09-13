import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const run=path.join(root,'outputs',process.argv[2]||'full_q4_20260913');
const dates=rows=>rows.map(r=>[r[0]===null?null:new Date(r[0]+'T00:00:00Z'),...r.slice(1)]);
for(const [mode,file] of [['q42','result4-2'],['q43','result4-3']]){
 const payload=JSON.parse(await fs.readFile(path.join(run,mode+'_workbook_payload.json'),'utf8'));
 const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'data/raw/附件/附件5',file+'.xlsx')));
 const qa=path.join(run,'workbook_qa',file);await fs.mkdir(qa,{recursive:true});const previews=[];
 for(const [name,key] of (mode==='q42'?[['计划购电量','plan']]:[['计划购电量','plan'],['调整购电量','adjusted']])){
  const s=wb.worksheets.getItem(name);s.getRange('A1:EQ1').values=[payload.headers];
  s.getRange('A1').values=[['日期\\时间\n电量：kWh']];
  s.getRange('EP1').values=[[key==='plan'?'全天计划购电量\n（kWh）':'全天最终正常购电量\n（kWh）']];
  s.getRange('EQ1').values=[[mode==='q43'&&key==='plan'?'原计划购电费\n（元）':'全天实际总购电费\n含紧急购电（元）']];
  s.getRange('A2:EQ335').values=dates(payload[key]);
  s.getRange('A2:A335').setNumberFormat('yyyy-mm-dd');s.getRange('B2:EQ335').setNumberFormat('#,##0.00');
  s.getRange('EP2').formulas=[['=SUM(B2:EO2)']];s.getRange('EP2:EP335').fillDown();
  s.getRange('A1:EQ335').format.font={name:'宋体',size:10};
  s.getRange('A1:EQ1').format.rowHeight=48;s.getRange('A1:EQ1').format.wrapText=true;
  s.getRange('A1:A335').format.columnWidth=16;s.getRange('B1:EO335').format.columnWidth=15;
  s.getRange('EP1:EQ335').format.columnWidth=24;s.getRange('A2:EQ335').format.rowHeight=17;
  s.getRange('A1:EQ1').format.horizontalAlignment='center';s.getRange('A2:A335').format.horizontalAlignment='center';
  s.freezePanes.freezeRows(1);s.freezePanes.freezeColumns(1);
  const sums=s.getRange('EP2:EP335').values.flat();
  if(sums.some((v,i)=>typeof v!=='number'||Math.abs(v-payload[key][i][145])>1e-5))throw Error(name+' totals');
  previews.push([name,'A1:E6'],[name,'EN330:EQ335']);
 }
 const b=wb.worksheets.getItem('充放电量');
 for(let i=1;i<334;i++)b.getRangeByIndexes(1+i*6,0,6,6).copyFrom(b.getRange('A2:F7'),'all');
 b.getRange('A2:F2005').values=dates(payload.battery);b.getRange('A2:A2005').setNumberFormat('yyyy-mm-dd');
 b.getRange('C1:D1').values=[['充电量（kWh）','放电量（kWh）']];b.getRange('F1').values=[['储电量（kWh）']];
 b.getRange('C2:D2005').setNumberFormat('#,##0.00');b.getRange('F2:F2005').setNumberFormat('#,##0.00');
 b.getRange('A1:F2005').format.font={name:'宋体',size:10};b.getRange('A1:F2005').format.columnWidth=18;
 b.getRange('A2:F2005').format.rowHeight=17;b.getRange('A1:F1').format.rowHeight=28;
 b.getRange('A1:F2005').format.horizontalAlignment='center';b.getRange('C2:D2005').format.horizontalAlignment='right';
 b.getRange('F2:F2005').format.horizontalAlignment='right';
 const e=wb.worksheets.getItem('紧急购电量');e.getRange('A2:C11').clear({applyTo:'contents'});
 const end=payload.emergency.length+1;e.getRange(`A2:C${end}`).values=dates(payload.emergency);
 e.getRange('C1').values=[['购电量（kWh）']];e.getRange(`A2:A${end}`).setNumberFormat('yyyy-mm-dd');
 e.getRange(`C2:C${end}`).setNumberFormat('#,##0.00');
 e.getRange(`A1:C${end}`).format.font={name:'宋体',size:10};e.getRange(`A1:C${end}`).format.columnWidth=20;
 e.getRange(`A2:C${end}`).format.rowHeight=17;e.getRange('A1:C1').format.rowHeight=28;
 e.getRange(`A1:C${end}`).format.horizontalAlignment='center';e.getRange(`C2:C${end}`).format.horizontalAlignment='right';
 e.getRange(`A2:C${end}`).format.borders={preset:'none'};
 e.getRange(`A2:C${end}`).format.borders={left:{style:'thin',color:'#333333'},right:{style:'thin',color:'#333333'},insideVertical:{style:'thin',color:'#333333'}};
 for(let i=0;i<payload.emergency.length;i++)if(i===payload.emergency.length-1||payload.emergency[i+1][0]!==null)
  e.getRangeByIndexes(i+1,0,1,3).format.borders={bottom:{style:'thin',color:'#333333'}};
 for(const s of [b,e])s.freezePanes.freezeRows(1);
 previews.push(['充放电量','A1:F13'],['充放电量','A2000:F2005'],['紧急购电量','A1:C12'],['紧急购电量',`A${Math.max(2,end-5)}:C${end}`]);
 for(const [sheetName,range] of previews){const png=await wb.render({sheetName,range,scale:1.5,format:'png'});await fs.writeFile(path.join(qa,sheetName+'_'+range.replace(':','_')+'.png'),new Uint8Array(await png.arrayBuffer()));}
 const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1500});
 const check=await wb.inspect({kind:'table',range:(mode==='q42'?'计划购电量':'调整购电量')+'!EP330:EQ335',include:'values,formulas',tableMaxRows:6,tableMaxCols:2,maxChars:1800});
 await fs.writeFile(path.join(qa,'artifact_checks.json'),JSON.stringify({errors:scan.ndjson,totals:check.ndjson},null,2));
 await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(run,'deliverables',file+'.xlsx'));
 console.log('Exported',file);
}
