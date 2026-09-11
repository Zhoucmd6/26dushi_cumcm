$ErrorActionPreference = 'Stop'
$dependencyTarget = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
$linkPath = Join-Path $PSScriptRoot 'node_modules'
if (-not (Test-Path -LiteralPath $dependencyTarget -PathType Container)) {
    throw '没有找到本机Excel导出依赖目录，请先配置Node.js及@oai/artifact-tool。'
}
$dependencyTarget = (Resolve-Path -LiteralPath $dependencyTarget).Path
if (Test-Path -LiteralPath $linkPath) {
    $existing = Get-Item -LiteralPath $linkPath
    if ($existing.LinkType -ne 'Junction' -or [string]$existing.Target -ne $dependencyTarget) {
        throw 'node_modules已存在且不是预期联接，请先核实，程序不会覆盖。'
    }
} else {
    New-Item -ItemType Junction -Path $linkPath -Target $dependencyTarget | Out-Null
}
Write-Output 'Excel导出依赖已连接。'
