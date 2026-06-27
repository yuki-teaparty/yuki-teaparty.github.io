# ─────────────────────────────────────────────────────────────
# 重新编译博客：content/posts/*.md  →  blog/*.html
#
#   .\build.ps1            只编译
#   .\build.ps1 -Serve     编译后起本地预览并打开浏览器（Ctrl+C 停止）
#   .\build.ps1 -Serve -Port 9000
#
# 注：知乎→markdown 的迁移（npm run migrate）是一次性的，平时不用跑。
#     以后写新文章 = 在 content/posts/ 新建 .md → 跑这个脚本。
# ─────────────────────────────────────────────────────────────
param(
  [switch]$Serve,
  [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# 首次运行自动装依赖
if (-not (Test-Path "$PSScriptRoot\node_modules")) {
  Write-Host "首次运行：安装依赖 (npm install)…" -ForegroundColor Cyan
  npm install
}

Write-Host "编译中 (npm run build)…" -ForegroundColor Cyan
npm run build
if ($LASTEXITCODE -ne 0) { throw "构建失败" }
Write-Host "✓ 编译完成" -ForegroundColor Green

if ($Serve) {
  # 后台起静态服务（根目录服务，绝对路径 /assets /blog 才能正确解析）
  $server = Start-Process python -ArgumentList '-m', 'http.server', $Port `
    -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow
  try {
    # 等端口就绪再开浏览器
    for ($i = 0; $i -lt 20; $i++) {
      try { Invoke-WebRequest "http://localhost:$Port/" -UseBasicParsing -TimeoutSec 1 | Out-Null; break }
      catch { Start-Sleep -Milliseconds 250 }
    }
    Start-Process "http://localhost:$Port/blog/"
    Write-Host "预览：http://localhost:$Port/  —— 按 Ctrl+C 停止" -ForegroundColor Yellow
    Wait-Process -Id $server.Id
  } finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
  }
}
