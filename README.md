# yuki-teaparty.github.io

老鸽子 Yuki 的个人主页 —— 🕊️ 茶话会主题的单页站点。

线上地址：<https://yuki-teaparty.github.io/>

## 技术栈

主页是纯手写静态 HTML / CSS / JS；**博客**则由一个轻量 Node 脚本把 markdown 编译成同主题的网页。
两者最终都是静态文件，`.nojekyll` 让 GitHub Pages 直接按原样服务。

| 路径 | 说明 |
|------|------|
| `index.html` | 主页（Hero / 关于 / 精选作品 / 页脚），手写维护 |
| `content/posts/*.md` | **博客文章源**（front-matter 只放内容元信息）|
| `content/series.yaml` | **站点结构清单**：列出哪些文章发布、归哪个专题、排什么顺序 |
| `blog/index.html`、`blog/posts/*.html` | 由 build 生成的博客页面（提交进仓库）|
| `assets/css/style.css` | 茶话会主题样式（含顶部导航）|
| `assets/css/blog.css` | 博客文章排版样式 |
| `assets/img/posts/<slug>/` | 文章配图 |
| `assets/js/main.js` | 滚动淡入效果 |
| `tools/zhihu2md.mjs` | **一次性**：知乎"保存网页"HTML → markdown + 下载图片 |
| `tools/build.mjs`、`tools/templates/` | markdown → 博客 HTML 的编译器与模板 |
| `build.ps1` / `build.sh` | 一键重新编译（可选 `-Serve` 预览）|
| `.nojekyll` | 关闭 GitHub Pages 的 Jekyll 处理，按原样部署 |

## 本地预览

主页可以直接用浏览器打开 `index.html`。

**博客页面用的是根绝对路径**（`/assets`、`/blog`），必须从仓库根目录起静态服务才能正确加载——直接 `file://` 打开会丢样式。用自带脚本最省事：

```powershell
.\build.ps1 -Serve          # 编译 + 起服务 + 自动开浏览器，Ctrl+C 停止
```

或手动：

```bash
python -m http.server 4000   # 在仓库根目录运行，访问 http://localhost:4000/blog/
```

## 写博客 / 重新编译

文章的真正源头是 `content/posts/` 里的 markdown，`blog/` 下的 HTML 是编译产物。

1. 在 `content/posts/` 新建或编辑 `.md`。front-matter 只放内容元信息——**不再需要 draft / order / series**（发布与否、归哪个专题、排第几，全部交给 `content/series.yaml`）：

   ```yaml
   ---
   title: "文章标题"
   date: "2024-01-01 12:00"
   slug: my-post           # 决定输出文件名 blog/posts/my-post.html
   summary: "一句话摘要"
   source: ""              # 可选（知乎搬运遗留）
   original_url: ""        # 可选
   ---
   ```

   正文里公式照常用 `$行内$` / `$$行间$$`（运行时由 MathJax 渲染；`\bm` 等非标准宏已在模板的 `macros` 里补好）。
   图片放 `assets/img/posts/<slug>/`，用 `/assets/img/posts/<slug>/xxx.jpg` 这样的绝对路径引用。

2. 把文件名（不含 `.md`）登记进 `content/series.yaml`，**没登记就不会发布**。专题从上到下＝页面从新到旧，专题内从上到下＝阅读顺序：

   ```yaml
   我的专题:
     - my-post
   ```

3. 重新编译并预览：

   ```powershell
   .\build.ps1 -Serve        # PowerShell
   # 或  ./build.sh --serve   # Git Bash
   ```

4. 满意后提交，GitHub Pages 自动上线：

   ```bash
   git add content/ blog/ assets/ && git commit -m "..." && git push
   ```

> 提示：`tools/zhihu2md.mjs`（`npm run migrate`）是**一次性**的知乎搬运工具，平时不要再跑——重跑会重新抓图并覆盖你手改过的 markdown。

## 自定义

- **头像**：把图片放到 `assets/img/avatar.jpg`，再把 `index.html` 里 `.avatar` 中的占位 `<span>` 换成 `<img src="./assets/img/avatar.jpg" alt="老鸽子 Yuki">`。
- **文字 / 兴趣 / 社交链接 / 作品**：都直接在 `index.html` 里改对应区块即可。
- **配色 / 字体**：在 `assets/css/style.css` 顶部的 `:root` 变量里调。
