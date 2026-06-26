# yuki-teaparty.github.io

老鸽子 Yuki 的个人主页 —— 🕊️ 茶话会主题的单页站点。

线上地址：<https://yuki-teaparty.github.io/>

## 技术栈

纯静态 HTML / CSS / JS，**不依赖任何构建工具**（`.nojekyll` 让 GitHub Pages 直接服务原始文件）。

| 路径 | 说明 |
|------|------|
| `index.html` | 全部页面内容（Hero / 关于 / 精选作品 / 页脚）|
| `assets/css/style.css` | 茶话会主题样式 |
| `assets/js/main.js` | 滚动淡入效果 |
| `.nojekyll` | 关闭 GitHub Pages 的 Jekyll 处理，按原样部署 |

## 本地预览

直接用浏览器打开 `index.html` 即可；或起一个静态服务器：

```bash
python -m http.server 4000   # 然后访问 http://localhost:4000
```

## 自定义

- **头像**：把图片放到 `assets/img/avatar.jpg`，再把 `index.html` 里 `.avatar` 中的占位 `<span>` 换成 `<img src="./assets/img/avatar.jpg" alt="老鸽子 Yuki">`。
- **文字 / 兴趣 / 社交链接 / 作品**：都直接在 `index.html` 里改对应区块即可。
- **配色 / 字体**：在 `assets/css/style.css` 顶部的 `:root` 变量里调。
