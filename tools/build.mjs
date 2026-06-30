// ─────────────────────────────────────────────────────────────
// build：content/posts/*.md  →  blog/index.html + blog/posts/*.html
//
//   node tools/build.mjs
//
// - markdown-it 渲染正文，自定义 math 插件把 $..$ / $$..$$ 保护成
//   MathJax 的 \(..\) / \[..\]（运行时由 MathJax 渲染，见模板）
// - 套用 tools/templates 下的模板 + 茶话会主题
// ─────────────────────────────────────────────────────────────
import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import anchor from 'markdown-it-anchor';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..');
const CONTENT_DIR = path.join(REPO, 'content', 'posts');
const BLOG_DIR = path.join(REPO, 'blog');
const POSTS_OUT = path.join(BLOG_DIR, 'posts');
const TPL_DIR = path.join(__dirname, 'templates');

// ── 工具 ────────────────────────────────────────────────────────
const escapeHtml = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// 把 TeX 文本放进 HTML 正文：只需转义 & < >（MathJax 读 textContent 会还原）
const escapeTex = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function fill(tpl, vars) {
  return tpl.replace(/\{\{(\w+)\}\}/g, (_, k) => (k in vars ? vars[k] : ''));
}

// ── markdown-it 数学插件（保护 $..$ / $$..$$，输出给 MathJax）─────
function mathPlugin(md) {
  // 行内 $...$
  md.inline.ruler.after('escape', 'math_inline', (state, silent) => {
    const start = state.pos;
    if (state.src.charCodeAt(start) !== 0x24 /* $ */) return false;
    const max = state.posMax;
    let pos = start + 1;
    while (pos < max && (state.src.charCodeAt(pos) !== 0x24 || state.src.charCodeAt(pos - 1) === 0x5c /* \ */)) {
      pos++;
    }
    if (pos >= max) return false; // 无闭合 $
    if (pos === start + 1) return false; // 空 $$（让 block 规则处理）
    if (!silent) {
      const token = state.push('math_inline', 'math', 0);
      token.content = state.src.slice(start + 1, pos);
    }
    state.pos = pos + 1;
    return true;
  });

  // 行间 $$ ... $$（独占若干行）
  md.block.ruler.before('fence', 'math_block', (state, startLine, endLine, silent) => {
    const startPos = state.bMarks[startLine] + state.tShift[startLine];
    const firstMax = state.eMarks[startLine];
    if (startPos + 2 > firstMax) return false;
    if (state.src.slice(startPos, startPos + 2) !== '$$') return false;

    let content;
    let nextLine = startLine;
    const firstRest = state.src.slice(startPos + 2, firstMax);

    if (firstRest.trim().endsWith('$$') && firstRest.trim().length > 2) {
      // 单行 $$ ... $$
      content = firstRest.trim().slice(0, -2);
    } else {
      const lines = [];
      if (firstRest.trim()) lines.push(firstRest);
      let found = false;
      for (nextLine = startLine + 1; nextLine < endLine; nextLine++) {
        const mb = state.bMarks[nextLine] + state.tShift[nextLine];
        const me = state.eMarks[nextLine];
        const line = state.src.slice(mb, me);
        const idx = line.indexOf('$$');
        if (idx !== -1) {
          if (line.slice(0, idx).trim()) lines.push(line.slice(0, idx));
          found = true;
          break;
        }
        lines.push(line);
      }
      if (!found) return false;
      content = lines.join('\n');
    }
    if (silent) return true;

    state.line = nextLine + 1;
    const token = state.push('math_block', 'math', 0);
    token.block = true;
    token.content = content.trim();
    token.map = [startLine, state.line];
    return true;
  });

  md.renderer.rules.math_inline = (tokens, idx) => '\\(' + escapeTex(tokens[idx].content) + '\\)';
  md.renderer.rules.math_block = (tokens, idx) =>
    '<div class="math-display">\\[' + escapeTex(tokens[idx].content) + '\\]</div>\n';
}

const md = new MarkdownIt({ html: true, linkify: true, typographer: false, breaks: false });
md.use(mathPlugin);
md.use(anchor, { permalink: anchor.permalink.headerLink(), slugify: (s) => encodeURIComponent(s.trim().replace(/\s+/g, '-')) });

// ── 摘要清洗（去掉知乎导航类前缀，截断）────────────────────────
function makeExcerpt(summary, limit = 120) {
  let s = (summary || '').replace(/\s+/g, ' ').trim();
  s = s.replace(/^前文导航\s*/, '').replace(/^(康拉德|Mizore)[:：].*?(?=在|本文|这|我们|首先|$)/, '');
  if (s.length > limit) s = s.slice(0, limit).trim() + '…';
  return s;
}

// ── 主流程 ──────────────────────────────────────────────────────
async function main() {
  const files = (await readdir(CONTENT_DIR)).filter((f) => f.endsWith('.md'));
  if (!files.length) {
    console.error('✗ content/posts/ 下没有 markdown 文件，先跑 npm run migrate');
    process.exit(1);
  }

  const postTpl = await readFile(path.join(TPL_DIR, 'post.html'), 'utf8');
  const indexTpl = await readFile(path.join(TPL_DIR, 'index.html'), 'utf8');

  // 读取 + 解析全部文章
  const posts = [];
  for (const f of files) {
    const raw = await readFile(path.join(CONTENT_DIR, f), 'utf8');
    const { data, content } = matter(raw);
    if (data.draft) continue; // draft 文章不参与编译（不进侧栏/列表，也不生成 HTML）
    const slug = data.slug || f.replace(/\.md$/, '');
    posts.push({
      slug,
      title: data.title || slug,
      date: data.date || '',
      order: typeof data.order === 'number' ? data.order : 999,
      summary: data.summary || '',
      source: data.source || '',
      original_url: data.original_url || '',
      html: md.render(content),
    });
  }
  posts.sort((a, b) => a.order - b.order || a.slug.localeCompare(b.slug));

  // 左侧栏：文章目录（current 高亮）
  const sidebarList = (currentSlug) =>
    posts
      .map((p) => {
        const cur = p.slug === currentSlug ? ' is-current' : '';
        const aria = p.slug === currentSlug ? ' aria-current="page"' : '';
        return `          <li><a class="sidebar__link${cur}"${aria} href="/blog/posts/${p.slug}.html">${escapeHtml(p.title)}</a></li>`;
      })
      .join('\n');

  await mkdir(POSTS_OUT, { recursive: true });

  // 逐篇生成
  for (let i = 0; i < posts.length; i++) {
    const p = posts[i];
    const prev = posts[i - 1];
    const next = posts[i + 1];
    const prevLink = prev
      ? `<a class="post-nav-link post-nav-link--prev" href="/blog/posts/${prev.slug}.html"><span class="post-nav-link__label">← 上一篇</span><span class="post-nav-link__title">${escapeHtml(prev.title)}</span></a>`
      : '<span></span>';
    const nextLink = next
      ? `<a class="post-nav-link post-nav-link--next" href="/blog/posts/${next.slug}.html"><span class="post-nav-link__label">下一篇 →</span><span class="post-nav-link__title">${escapeHtml(next.title)}</span></a>`
      : '<span></span>';

    // 原创文章没有知乎 original_url，此时不渲染「原文」链接（否则会得到 href="" 的死链）
    const originLink = p.original_url
      ? `<a class="post__origin" href="${escapeHtml(p.original_url)}" target="_blank" rel="noopener">${escapeHtml(p.source)}原文</a>`
      : '';

    const out = fill(postTpl, {
      title: escapeHtml(p.title),
      summary: escapeHtml(makeExcerpt(p.summary, 160)),
      date: escapeHtml(p.date),
      origin_link: originLink,
      content: p.html,
      prev_link: prevLink,
      next_link: nextLink,
      sidebar_list: sidebarList(p.slug),
    });
    await writeFile(path.join(POSTS_OUT, `${p.slug}.html`), out, 'utf8');
  }

  // 列表页
  const cards = posts
    .map((p) => {
      const excerpt = escapeHtml(makeExcerpt(p.summary));
      return `        <article class="card card--link">
          <div class="card__head">
            <h3 class="card__title"><a class="card__titlelink" href="/blog/posts/${p.slug}.html">${escapeHtml(p.title)}</a>${p.date ? `<span class="card__date">${escapeHtml(p.date)}</span>` : ''}</h3>
          </div>
          <p class="card__desc">${excerpt}</p>
        </article>`;
    })
    .join('\n');

  await writeFile(path.join(BLOG_DIR, 'index.html'), fill(indexTpl, { posts: cards, sidebar_list: sidebarList('') }), 'utf8');

  console.log(`✓ 生成 ${posts.length} 篇文章 + 列表页`);
  console.log(`  blog/index.html`);
  posts.forEach((p) => console.log(`  blog/posts/${p.slug}.html  (#${p.order} ${p.title})`));
}

main();
