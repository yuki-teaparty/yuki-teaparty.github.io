#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 重新编译博客：content/posts/*.md  →  blog/*.html
#
#   ./build.sh                只编译
#   ./build.sh --serve        编译后起本地预览并打开浏览器（Ctrl+C 停止）
#   ./build.sh --serve 9000   指定端口
#
# 注：知乎→markdown 的迁移（npm run migrate）是一次性的，平时不用跑。
#     以后写新文章 = 在 content/posts/ 新建 .md → 跑这个脚本。
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# 首次运行自动装依赖
if [ ! -d node_modules ]; then
  echo "首次运行：安装依赖 (npm install)…"
  npm install
fi

echo "编译中 (npm run build)…"
npm run build
echo "✓ 编译完成"

if [ "${1:-}" = "--serve" ] || [ "${1:-}" = "-s" ]; then
  PORT="${2:-8765}"
  # 后台延迟开浏览器（用 python 的 webbrowser，跨平台可靠）
  ( sleep 1; python -m webbrowser -t "http://localhost:$PORT/blog/" >/dev/null 2>&1 || true ) &
  echo "预览：http://localhost:$PORT/  —— 按 Ctrl+C 停止"
  # 根目录服务，绝对路径 /assets /blog 才能正确解析
  python -m http.server "$PORT"
fi
