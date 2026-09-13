import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'reviews','template_inspection');
await fs.mkdir(out,{recursive:true});
for (const name of ['result1','result2']) {
  const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'data','raw','附件','附件5',`${name}.xlsx`)));
  console.log(name,(await workbook.inspect({kind:'workbook,sheet,table',maxChars:2400,tableMaxRows:2,tableMaxCols:3})).ndjson);
  const specs=name==='result1' ? [['计划购电量','A1:B9'],['充放电量','A1:E7']]
    : [['计划购电量','A1:E7'],['充放电量','A1:F8'],['紧急购电量','A1:C11']];
  for (const [sheetName,range] of specs) {
    const preview=await workbook.render({sheetName,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(out,`${name}_${sheetName}.png`),new Uint8Array(await preview.arrayBuffer()));
  }
}
