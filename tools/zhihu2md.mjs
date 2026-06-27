// ─────────────────────────────────────────────────────────────
// 一次性迁移 parser：知乎"保存网页" HTML  →  干净 markdown + 本地图片
//
//   node tools/zhihu2md.mjs              处理 SRC_DIR 下全部文章
//
// 产物：
//   content/posts/<slug>.md              带 YAML front-matter 的文章
//   assets/img/posts/<slug>/<hash>.<ext> 重新下载的配图（失败回退本地副本）
//   tools/migration-report.json          每篇的处理结果 / 下载失败清单
// ─────────────────────────────────────────────────────────────
import { readdir, readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';
import TurndownService from 'turndown';
import { gfm } from 'turndown-plugin-gfm';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..');

// 知乎导出目录（保存的 .html + 同名 _files 资源夹）
const SRC_DIR = 'D:/work/老电脑资料/鸽子的知乎';
const CONTENT_DIR = path.join(REPO, 'content', 'posts');
const IMG_ROOT = path.join(REPO, 'assets', 'img', 'posts');

// 系列号 → 可读 slug（系列号从标题 "(1.5)" 这种括号里解析）
const SLUG_BY_NUM = {
  '1': 'diffusion-1-sde',
  '1.5': 'diffusion-1p5-ode',
  '1.7': 'diffusion-1p7-samplers',
  '2': 'diffusion-2-ldm',
  '2.1': 'diffusion-2p1-vsd',
  '2.2': 'diffusion-2p2-lcm',
};

const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/124.0 Safari/537.36';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── 图片下载（带重试），失败回退本地 _files 副本 ──────────────────
async function downloadImage(url, destNoExt, filesDir) {
  const urlExt = (path.extname(new URL(url).pathname) || '.jpg').toLowerCase();
  const dest = destNoExt + urlExt;
  // 1) 尝试从 data-original 下载原图
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetch(url, {
        headers: {
          'User-Agent': UA,
          Referer: 'https://www.zhihu.com/',
          Accept: 'image/avif,image/webp,image/apng,image/*,*/*',
        },
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length < 200) throw new Error('suspiciously small (' + buf.length + 'B)');
      await writeFile(dest, buf);
      return { ext: urlExt, source: 'remote', bytes: buf.length };
    } catch (err) {
      if (attempt === 3) var lastErr = err;
      else await sleep(500 * attempt);
    }
  }
  // 2) 回退：复制本地 _files 里已下载的副本（扩展名以实际文件为准）
  const hash = path.basename(destNoExt); // 形如 v2-xxxx
  try {
    const files = await readdir(filesDir);
    const local = files.find((f) => f.startsWith(hash));
    if (local) {
      const ext = path.extname(local) || '.jpg';
      await copyFile(path.join(filesDir, local), destNoExt + ext);
      return { ext, source: 'local-fallback' };
    }
  } catch {
    /* _files 不存在则继续 */
  }
  return { ext: null, source: 'failed', error: String(lastErr) };
}

// ── 解码知乎外链跳转 link.zhihu.com/?target=<urlencoded> ──────────
function decodeZhihuLink(href) {
  try {
    const u = new URL(href);
    if (u.hostname.endsWith('zhihu.com') && u.pathname === '/' && u.searchParams.has('target')) {
      return u.searchParams.get('target');
    }
  } catch {
    /* 非法 URL，原样返回 */
  }
  return href;
}

// ── 判断公式是否为行间（独占一个段落）──────────────────────────
function isBlockMath(node) {
  const p = node.parentNode;
  if (!p || p.nodeName !== 'P') return false;
  for (const child of p.childNodes) {
    if (child === node) continue;
    if (child.nodeType === 3 && child.textContent.trim() === '') continue; // 纯空白
    return false; // 段落里还有别的东西 → 行内
  }
  return true;
}

function buildTurndown(urlToLocal) {
  const td = new TurndownService({
    headingStyle: 'atx',
    hr: '---',
    bulletListMarker: '-',
    codeBlockStyle: 'fenced',
    emDelimiter: '*',
    strongDelimiter: '**',
  });
  td.use(gfm);

  // 公式：ztext-math[data-tex] → $...$ / $$...$$
  td.addRule('zhihuMath', {
    filter: (node) =>
      node.nodeName === 'SPAN' &&
      node.classList.contains('ztext-math') &&
      node.getAttribute('data-tex') != null,
    replacement: (_content, node) => {
      const tex = node.getAttribute('data-tex').trim();
      if (!tex) return '';
      return isBlockMath(node) ? `\n\n$$\n${tex}\n$$\n\n` : `$${tex}$`;
    },
  });

  // 配图：<figure> → 本地图片（带 figcaption 时输出原生 <figure>）
  const figureToMd = (node) => {
    const img = node.querySelector('img[data-original]') || node.querySelector('img');
    if (!img) return '';
    const remote =
      img.getAttribute('data-original') ||
      img.getAttribute('data-actualsrc') ||
      img.getAttribute('src');
    const local = urlToLocal[remote] || remote;
    const capEl = node.querySelector('figcaption');
    const cap = capEl ? capEl.textContent.trim() : '';
    if (cap) {
      const e = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
      return `\n\n<figure>\n  <img src="${local}" alt="${e(cap)}">\n  <figcaption>${e(cap)}</figcaption>\n</figure>\n\n`;
    }
    return `\n\n![](${local})\n\n`;
  };
  td.addRule('zhihuFigure', { filter: 'figure', replacement: (_c, node) => figureToMd(node) });

  // 兜底：figure 之外的零散 <img>
  td.addRule('looseImg', {
    filter: 'img',
    replacement: (_c, node) => {
      const remote =
        node.getAttribute('data-original') ||
        node.getAttribute('data-actualsrc') ||
        node.getAttribute('src') ||
        '';
      const local = urlToLocal[remote] || remote;
      return local ? `![](${local})` : '';
    },
  });

  // 链接：解开知乎跳转
  td.addRule('zhihuLink', {
    filter: (node) => node.nodeName === 'A' && node.getAttribute('href'),
    replacement: (content, node) => {
      const href = decodeZhihuLink(node.getAttribute('href'));
      if (!content.trim()) return '';
      return `[${content}](${href})`;
    },
  });

  return td;
}

// ── 处理单篇文章 ────────────────────────────────────────────────
async function processArticle(htmlPath) {
  const html = await readFile(htmlPath, 'utf8');
  const dom = new JSDOM(html);
  const doc = dom.window.document;

  const titleEl = doc.querySelector('h1.Post-Title');
  const richText = doc.querySelector('.Post-RichText');
  if (!titleEl || !richText) {
    return { file: path.basename(htmlPath), status: 'skipped', reason: 'no Post-Title/Post-RichText' };
  }
  const title = titleEl.textContent.trim();

  // 元数据
  const ogUrl = doc.querySelector('meta[property="og:url"]')?.getAttribute('content') || '';
  const ogDesc = doc.querySelector('meta[property="og:description"]')?.getAttribute('content') || '';
  const idMatch = ogUrl.match(/\/p\/(\d+)/);
  const postId = idMatch ? idMatch[1] : '';
  const numMatch = title.match(/[（(]\s*([\d.]+)\s*[)）]/);
  const seriesNum = numMatch ? numMatch[1] : '';
  const slug = SLUG_BY_NUM[seriesNum] || (postId ? `p-${postId}` : null);
  if (!slug) return { file: path.basename(htmlPath), status: 'skipped', reason: 'cannot derive slug' };
  const order = seriesNum ? parseFloat(seriesNum) : 999;

  const timeMatch = html.match(/(编辑于|发布于)\s*([\d]{4}-[\d]{2}-[\d]{2})(?:\s+([\d]{2}:[\d]{2}))?/);
  const date = timeMatch ? timeMatch[2] + (timeMatch[3] ? ` ${timeMatch[3]}` : '') : '';

  // ── 预扫：收集正文里所有图片的 data-original，先下载 ──────────
  const filesDir = htmlPath.replace(/\.html$/i, '') + '_files';
  const imgDestDir = path.join(IMG_ROOT, slug);
  await mkdir(imgDestDir, { recursive: true });

  const urls = new Set();
  for (const img of richText.querySelectorAll('img')) {
    const u = img.getAttribute('data-original') || img.getAttribute('data-actualsrc') || img.getAttribute('src') || '';
    if (/^https?:\/\//.test(u) && /zhimg\.com/.test(u)) urls.add(u);
  }

  const urlToLocal = {};
  const failures = [];
  for (const url of urls) {
    const hash = (url.match(/v2-[0-9a-f]+/i) || [path.basename(new URL(url).pathname).split('_')[0]])[0];
    const destNoExt = path.join(imgDestDir, hash);
    const r = await downloadImage(url, destNoExt, filesDir);
    if (r.source === 'failed') {
      failures.push({ url, error: r.error });
      urlToLocal[url] = url; // 实在拿不到就保留远程链接
    } else {
      urlToLocal[url] = `/assets/img/posts/${slug}/${hash}${r.ext}`;
    }
    await sleep(120); // 礼貌限速
  }

  // ── 正文 HTML → markdown ─────────────────────────────────────
  const td = buildTurndown(urlToLocal);
  let md = td.turndown(richText.innerHTML);
  md = md.replace(/\n{3,}/g, '\n\n').trim() + '\n';

  // front-matter（title 用 JSON 引号确保含冒号/括号也安全）
  const fm = [
    '---',
    `title: ${JSON.stringify(title)}`,
    `date: ${JSON.stringify(date)}`,
    `slug: ${slug}`,
    `order: ${order}`,
    `original_url: ${JSON.stringify(ogUrl)}`,
    `summary: ${JSON.stringify(ogDesc)}`,
    'source: 知乎专栏',
    '---',
    '',
  ].join('\n');

  await mkdir(CONTENT_DIR, { recursive: true });
  await writeFile(path.join(CONTENT_DIR, `${slug}.md`), fm + md, 'utf8');

  return {
    file: path.basename(htmlPath),
    status: 'ok',
    slug,
    order,
    date,
    images: urls.size,
    imageFailures: failures.length,
    failures,
  };
}

// ── 主流程 ──────────────────────────────────────────────────────
async function main() {
  if (!existsSync(SRC_DIR)) {
    console.error(`✗ 找不到知乎源目录：${SRC_DIR}`);
    process.exit(1);
  }
  const entries = await readdir(SRC_DIR);
  const htmls = entries.filter((f) => f.toLowerCase().endsWith('.html'));
  console.log(`找到 ${htmls.length} 篇文章，开始迁移…\n`);

  const report = [];
  for (const f of htmls) {
    process.stdout.write(`· ${f} … `);
    try {
      const r = await processArticle(path.join(SRC_DIR, f));
      report.push(r);
      if (r.status === 'ok') {
        console.log(`✓ ${r.slug}  (图 ${r.images}，失败 ${r.imageFailures})`);
      } else {
        console.log(`跳过：${r.reason}`);
      }
    } catch (err) {
      console.log(`✗ 出错：${err.message}`);
      report.push({ file: f, status: 'error', error: String(err) });
    }
  }

  await writeFile(path.join(__dirname, 'migration-report.json'), JSON.stringify(report, null, 2), 'utf8');
  const ok = report.filter((r) => r.status === 'ok');
  const totalFail = ok.reduce((s, r) => s + (r.imageFailures || 0), 0);
  console.log(`\n完成：${ok.length}/${htmls.length} 篇成功，共 ${totalFail} 张图片下载失败（已回退/保留远程，详见 tools/migration-report.json）`);
}

main();
