// ─────────────────────────────────────────────────────────────
// build：content/posts/*.md  →  blog/index.html + blog/posts/*.html
//
//   node tools/build.mjs
//
// - markdown-it 渲染正文，自定义 math 插件把 $..$ / $$..$$ 保护成
//   MathJax 的 \(..\) / \[..\]（运行时由 MathJax 渲染，见模板）
// - 套用 tools/templates 下的模板 + 茶话会主题
// ─────────────────────────────────────────────────────────────
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import yaml from 'js-yaml'; // gray-matter 的传递依赖；用来解析 content/series.yaml
import MarkdownIt from 'markdown-it';
import anchor from 'markdown-it-anchor';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..');
const CONTENT_DIR = path.join(REPO, 'content', 'posts');
const MANIFEST = path.join(REPO, 'content', 'series.yaml');
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
  const postTpl = await readFile(path.join(TPL_DIR, 'post.html'), 'utf8');
  const indexTpl = await readFile(path.join(TPL_DIR, 'index.html'), 'utf8');

  // content/series.yaml 是唯一的结构来源：决定发布哪些文章、归属哪个专题、以及顺序。
  // 专题从上到下 = 页面从前到后；专题内文章从上到下 = 阅读顺序（第 1 篇在前）。
  // 只有列在清单里的才发布，其余（含 claude_drafts/）一律当草稿跳过；front-matter 不再需要 draft / order / series。
  const manifest = yaml.load(await readFile(MANIFEST, 'utf8'));
  const entries = manifest && typeof manifest === 'object' ? Object.entries(manifest) : [];
  if (!entries.length) {
    console.error('✗ content/series.yaml 为空或格式不对（应为「专题名: [文件名, ...]」的映射）');
    process.exit(1);
  }

  // 按清单顺序读取每篇文章
  const seriesOrder = [];
  const seriesMap = new Map();
  for (const [key, names] of entries) {
    const list = [];
    seriesMap.set(key, list);
    seriesOrder.push(key);
    for (const name of names || []) {
      let raw;
      try {
        raw = await readFile(path.join(CONTENT_DIR, `${name}.md`), 'utf8');
      } catch {
        console.error(`✗ series.yaml 里列了 "${name}"，但 content/posts/${name}.md 不存在`);
        process.exit(1);
      }
      const { data, content } = matter(raw);
      list.push({
        slug: data.slug || name,
        title: data.title || name,
        date: data.date || '',
        summary: data.summary || '',
        source: data.source || '',
        original_url: data.original_url || '',
        html: md.render(content),
      });
    }
  }
  const posts = seriesOrder.flatMap((key) => seriesMap.get(key));

  // 同专题内的上一篇/下一篇（不跨专题互链）
  const navOf = new Map();
  for (const key of seriesOrder) {
    const group = seriesMap.get(key);
    group.forEach((p, i) => navOf.set(p.slug, { prev: group[i - 1], next: group[i + 1] }));
  }

  // 左侧栏：按专题分组（current 高亮）
  const sidebarNav = (currentSlug) =>
    seriesOrder
      .map((key) => {
        const items = seriesMap
          .get(key)
          .map((p) => {
            const cur = p.slug === currentSlug ? ' is-current' : '';
            const aria = p.slug === currentSlug ? ' aria-current="page"' : '';
            return `            <li><a class="sidebar__link${cur}"${aria} href="/blog/posts/${p.slug}.html">${escapeHtml(p.title)}</a></li>`;
          })
          .join('\n');
        return `        <div class="sidebar__group">
          <p class="sidebar__heading">${escapeHtml(key)}</p>
          <ul class="sidebar__list">
${items}
          </ul>
        </div>`;
      })
      .join('\n');

  await mkdir(POSTS_OUT, { recursive: true });

  // 逐篇生成
  for (const p of posts) {
    const { prev, next } = navOf.get(p.slug);
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
      sidebar_nav: sidebarNav(p.slug),
    });
    await writeFile(path.join(POSTS_OUT, `${p.slug}.html`), out, 'utf8');
  }

  // 列表页：按专题分节
  const sections = seriesOrder
    .map((key, idx) => {
      const cards = seriesMap
        .get(key)
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
      return `    <section class="section reveal" aria-labelledby="series-${idx}-title">
      <h2 id="series-${idx}-title" class="section__title"><span>${escapeHtml(key)}</span></h2>
      <div class="cards">
${cards}
      </div>
    </section>`;
    })
    .join('\n');

  await writeFile(path.join(BLOG_DIR, 'index.html'), fill(indexTpl, { sections, sidebar_nav: sidebarNav('') }), 'utf8');

  console.log(`✓ 生成 ${posts.length} 篇文章 + 列表页`);
  console.log(`  blog/index.html`);
  posts.forEach((p) => console.log(`  blog/posts/${p.slug}.html  (${p.title})`));
}

main();
