param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$v2Project=$PSScriptRoot
$v2Run=(Resolve-Path -LiteralPath $RunDirectory).Path
$v2Runtime=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node'
$v2Node=Join-Path $v2Runtime 'bin\node.exe'
$v2Packages=Join-Path $v2Runtime 'node_modules'
if(!(Test-Path -LiteralPath $v2Node) -or !(Test-Path -LiteralPath $v2Packages)) {
    throw '找不到已配置的Codex Node依赖。数值Python结果仍可独立复现。'
}
$v2Link=Join-Path $v2Project 'node_modules'
if(!(Test-Path -LiteralPath $v2Link)) {
    New-Item -ItemType Junction -Path $v2Link -Target $v2Packages | Out-Null
}
& $v2Node (Join-Path $v2Project 'src\export_workbooks_v2.mjs') $v2Run
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& 'D:\Anaconda\python.exe' (Join-Path $v2Project 'src\verify_workbooks_v2.py') $v2Run
exit $LASTEXITCODE
