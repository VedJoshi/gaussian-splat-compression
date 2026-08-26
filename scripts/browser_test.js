const { chromium } = require('playwright-core');
const EXE = 'C:/Users/vedti/AppData/Local/ms-playwright/chromium-1208/chrome-win64/chrome.exe';
const URL = 'http://127.0.0.1:8123/index.html?url=http://127.0.0.1:8123/truck.splat';
const OUT = 'C:/Users/vedti/NUS_CS(noOnedrive)/gaussian-splat-compression/_browser_test.png';

(async () => {
  const browser = await chromium.launch({
    executablePath: EXE,
    headless: false,
    args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=default'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errs = [];
  page.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 200)); });
  page.on('pageerror', e => errs.push('PAGEERROR: ' + String(e).slice(0, 200)));

  const t0 = Date.now();
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });

  let loaded = false;
  for (let i = 0; i < 90; i++) {
    await page.waitForTimeout(1000);
    const st = await page.evaluate(() => {
      const g = id => document.getElementById(id);
      const sp = g('spinner');
      return {
        fps: (g('fps') || {}).textContent || '',
        msg: (g('message') || {}).textContent || '',
        spinnerShown: !!sp && getComputedStyle(sp).display !== 'none',
      };
    });
    if (i % 5 === 0) console.log(`  t=${i}s fps="${st.fps}" msg="${st.msg}" spinner=${st.spinnerShown}`);
    if (st.msg && /unable|error/i.test(st.msg)) { console.log('VIEWER ERROR:', st.msg); break; }
    if (st.fps && !st.spinnerShown) { loaded = true; console.log(`LOADED at t=${i + 1}s`); break; }
  }
  const loadMs = Date.now() - t0;

  await page.waitForTimeout(5000);
  const fpsText = await page.evaluate(() => (document.getElementById('fps') || {}).textContent || '');
  const rafFps = await page.evaluate(() => new Promise(res => {
    let n = 0; const s = performance.now();
    (function tick() {
      n++;
      if (performance.now() - s < 3000) requestAnimationFrame(tick);
      else res(n / ((performance.now() - s) / 1000));
    })();
  }));
  const gl = await page.evaluate(() => {
    const c = document.createElement('canvas');
    const g = c.getContext('webgl2') || c.getContext('webgl');
    if (!g) return 'no webgl';
    const d = g.getExtension('WEBGL_debug_renderer_info');
    return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER);
  });
  const jsHeap = await page.evaluate(() => (performance.memory ? performance.memory.usedJSHeapSize : 0));

  await page.screenshot({ path: OUT });
  console.log('---RESULT---');
  console.log('loaded          :', loaded);
  console.log('load wall ms    :', loadMs);
  console.log('viewer fps text :', JSON.stringify(fpsText));
  console.log('raf fps         :', rafFps.toFixed(1));
  console.log('gl renderer     :', gl);
  console.log('js heap MiB     :', (jsHeap / 1048576).toFixed(1));
  console.log('console errors  :', errs.length ? errs.slice(0, 5) : 'none');
  console.log('screenshot      :', OUT);
  await browser.close();
})().catch(e => { console.error('FATAL', e); process.exit(1); });
