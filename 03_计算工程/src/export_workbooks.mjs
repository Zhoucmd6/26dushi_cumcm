import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const run=path.resolve(process.argv[2]);
const payload=JSON.parse(await fs.readFile(path.join(run,'workbook_payload.json'),'utf8'));
const out=path.join(run,'deliverables');
const qa=path.join(run,'workbook_qa');
await fs.mkdir(out,{recursive:true});
await fs.mkdir(qa,{recursive:true});
const checks=[];
const datesToExcel=rows=>rows.map(row=>[row[0]===null?null:new Date(`${row[0]}T00:00:00Z`),...row.slice(1)]);
const formatOnly=process.argv.includes('--format-only');

function finishExtendedFormatting(workbook) {
  const battery=workbook.worksheets.getItem('充放电量');
  const body=battery.getRange('A2:F2005');
  body.format.font={name:'宋体',size:10};
  body.format.rowHeight=15;
  body.format.horizontalAlignment='center';
  battery.getRange('C2:D2005').format.horizontalAlignment='right';
  battery.getRange('F2:F2005').format.horizontalAlignment='right';
  body.format.borders={preset:'none'};
  body.format.borders={left:{style:'thin',color:'#333333'},right:{style:'thin',color:'#333333'},insideVertical:{style:'thin',color:'#333333'}};
  for(let i=0;i<334;i++) battery.getRangeByIndexes(6+i*6,0,1,6).format.borders={bottom:{style:'thin',color:'#333333'}};
  const emergency=workbook.worksheets.getItem('紧急购电量');
  const end=payload.q2.emergency.length+1;
  const events=emergency.getRange(`A2:C${end}`);
  events.format.font={name:'宋体',size:10};
  events.format.rowHeight=15;
  events.format.horizontalAlignment='center';
  emergency.getRange(`C2:C${end}`).format.horizontalAlignment='right';
  events.format.borders={preset:'none'};
  events.format.borders={left:{style:'thin',color:'#333333'},right:{style:'thin',color:'#333333'},insideVertical:{style:'thin',color:'#333333'}};
  for(let i=0;i<payload.q2.emergency.length;i++) {
    if(i===payload.q2.emergency.length-1 || payload.q2.emergency[i+1][0]!==null) {
      emergency.getRangeByIndexes(i+1,0,1,3).format.borders={bottom:{style:'thin',color:'#333333'}};
    }
  }
}

for (const name of formatOnly?['result2']:['result1','result2']) {
  const input=formatOnly?path.join(out,`${name}.xlsx`):path.join(root,'data','raw','附件','附件5',`${name}.xlsx`);
  const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(input));
  const purchase=workbook.worksheets.getItem('计划购电量');
  const battery=workbook.worksheets.getItem('充放电量');
  let previews;
  if(formatOnly) {
    finishExtendedFormatting(workbook);
    previews=[['充放电量','A1:F13'],['充放电量','A2000:F2005'],['紧急购电量','A1:C12']];
  } else if(name==='result1') {
    purchase.getRange('A2:B145').values=payload.q1.purchases;
    purchase.getRange('B1').values=[['购电量（kWh）\n数据：附件1']];
    purchase.getRange('A1:B1').format.rowHeight=40;
    purchase.getRange('B1').format.wrapText=true;
    purchase.getRange('B2:B145').setNumberFormat('#,##0.00');
    battery.getRange('A2:E7').values=payload.q1.battery;
    battery.getRange('B2:C7').setNumberFormat('#,##0.00');
    battery.getRange('E2:E3').setNumberFormat('#,##0.00');
    const values=purchase.getRange('B2:B145').values.flat();
    const total=values.reduce((a,b)=>a+b,0);
    if(Math.abs(total-payload.q1.purchase_kwh)>1e-5) throw new Error('Q1 purchase total mismatch');
    previews=[['计划购电量','A1:B10'],['计划购电量','A138:B145'],['充放电量','A1:E7']];
  } else {
    purchase.getRange('A1:EQ1').values=[payload.q2.headers];
    purchase.getRange('A1').values=[['日期\\时间\n附件1、2']];
    purchase.getRange('A1:EQ1').format.rowHeight=40;
    purchase.getRange('A1:EQ1').format.wrapText=true;
    purchase.getRange('A2:EQ335').values=datesToExcel(payload.q2.purchases);
    purchase.getRange('A2:A335').setNumberFormat('yyyy-mm-dd');
    purchase.getRange('B2:EQ335').setNumberFormat('#,##0.00');
    purchase.getRange('EP2').formulas=[['=SUM(B2:EO2)']];
    purchase.getRange('EP2:EP335').fillDown();
    purchase.getRange('EP1:EQ335').format.columnWidth=18;
    for(let i=1;i<334;i++) battery.getRangeByIndexes(1+i*6,0,6,6).copyFrom(battery.getRange('A2:F7'),'all');
    battery.getRange('A2:F2005').values=datesToExcel(payload.q2.battery);
    battery.getRange('A2:A2005').setNumberFormat('yyyy-mm-dd');
    battery.getRange('C2:D2005').setNumberFormat('#,##0.00');
    battery.getRange('F2:F2005').setNumberFormat('#,##0.00');
    const emergency=workbook.worksheets.getItem('紧急购电量');
    emergency.getRange('A2:C11').clear({applyTo:'contents'});
    for(let i=1;i<payload.q2.emergency.length;i++) emergency.getRangeByIndexes(1+i,0,1,3).copyFrom(emergency.getRange('A2:C2'),'all');
    const end=payload.q2.emergency.length+1;
    emergency.getRange(`A2:C${end}`).values=datesToExcel(payload.q2.emergency);
    emergency.getRange(`A2:A${end}`).setNumberFormat('yyyy-mm-dd');
    emergency.getRange(`C2:C${end}`).setNumberFormat('#,##0.00');
    // 扩展后的行沿用模板字体与列结构，区间列适度加宽以显示合并事件。
    emergency.getRange(`B1:B${end}`).format.columnWidth=20;
    finishExtendedFormatting(workbook);
    previews=[['计划购电量','A1:E7'],['计划购电量','EN330:EQ335'],['充放电量','A1:F13'],
              ['充放电量','A2000:F2005'],['紧急购电量','A1:C12']];
    const sums=purchase.getRange('EP2:EP335').values.flat();
    if(sums.some((v,i)=>Math.abs(v-payload.q2.purchases[i][145])>1e-5)) throw new Error('Q2 daily sum formula mismatch');
  }
  for(const [sheetName,range] of previews) {
    const preview=await workbook.render({sheetName,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(qa,`${name}_${sheetName}_${range.replace(':','_')}.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  const inspected=await workbook.inspect({kind:'table',range:'计划购电量!A1:E6',include:'values,formulas',tableMaxRows:6,tableMaxCols:5,maxChars:1500});
  const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
                                       options:{useRegex:true,maxResults:20},summary:'formula errors',maxChars:1500});
  checks.push({file:`${name}.xlsx`,inspection:inspected.ndjson,formula_scan:errors.ndjson});
  await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(out,`${name}.xlsx`));
  console.log(`Exported ${name}.xlsx`);
}
await fs.writeFile(path.join(qa,formatOnly?'format_checks.json':'artifact_checks.json'),JSON.stringify(checks,null,2));
