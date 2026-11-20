"""生成手机端录音页（docs/record_phone.html）：把 50 句语料嵌进去，手机浏览器直接打开就能录。

用法：
    python tools/make_record_page.py
产物：
    docs/record_phone.html —— 三种模式（气声/低声/正常）各录一遍，文件按 <mode>-<id>.webm 下载
手机录音的原始文件用 tools/import_recordings.py 转成 16k wav 并整理成 bench-modes-<mode>/
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"
OUT = ROOT / "docs" / "record_phone.html"

MODES = [
    ("whisper", "气声（只出气，声带不振动）"),
    ("low", "低声（很小声但出声带振动）"),
    ("normal", "正常音量"),
]


def load_rows() -> list[dict]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid, cat, text = line.split("\t", 2)
        rows.append({"id": sid, "cat": cat, "text": text})
    return rows


def build() -> str:
    rows = load_rows()
    modes = [{"key": k, "label": v} for k, v in MODES]
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>气声录音 · 手机端</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 16px/1.5 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #0f1116; color: #e8ecf4; padding: 16px max(14px, env(safe-area-inset-left)); }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color: #8b95a8; font-size: 13px; margin-bottom: 14px; }}
  .modes {{ display: flex; gap: 8px; margin-bottom: 12px; }}
  .modes button {{ flex: 1; padding: 10px 8px; border-radius: 10px; border: 1px solid #2a3142;
                   background: #171b24; color: #cdd5e3; font-size: 14px; }}
  .modes button.on {{ border-color: #4fc3f7; color: #4fc3f7; background: #14222e; }}
  .card {{ background: #171b24; border: 1px solid #232a38; border-radius: 14px; padding: 16px; }}
  .meta {{ display: flex; justify-content: space-between; color: #8b95a8; font-size: 13px; }}
  .sentence {{ font-size: 26px; line-height: 1.45; margin: 14px 0 18px; }}
  .hint {{ color: #7f8aa0; font-size: 13px; margin-bottom: 14px; }}
  .rec {{ width: 100%; padding: 18px; border-radius: 14px; border: 0; font-size: 19px;
          background: #b3261e; color: #fff; font-weight: 600; }}
  .rec.on {{ background: #1f6f4a; }}
  .row {{ display: flex; gap: 10px; margin-top: 12px; }}
  .row button, .row a {{ flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #2a3142;
          background: #11151d; color: #dce3ef; font-size: 15px; text-align: center;
          text-decoration: none; }}
  .row button:disabled, .row a.disabled {{ opacity: .4; pointer-events: none; }}
  .status {{ font-size: 13px; color: #8b95a8; margin-top: 10px; white-space: pre-wrap; }}
  .done {{ color: #6ee7a8; }}
  .warn {{ color: #fbbf24; }}
  .prog {{ height: 6px; background: #232a38; border-radius: 3px; overflow: hidden; margin: 10px 0 2px; }}
  .prog i {{ display: block; height: 100%; width: 0; background: #4fc3f7; transition: width .2s; }}
  footer {{ color: #66708a; font-size: 12px; margin-top: 16px; }}
</style>
</head>
<body>
<h1>气声录音 · 手机端</h1>
<div class="sub">同一批 50 句，气声 / 低声 / 正常 各录一遍。录完把文件传到电脑，跑
  <code>python tools/import_recordings.py</code> 整理评测。</div>

<div class="modes" id="modes"></div>

<div class="card">
  <div class="meta"><span id="pos">1 / {len(rows)}</span><span id="cat"></span></div>
  <div class="prog"><i id="bar"></i></div>
  <div class="sentence" id="text">…</div>
  <div class="hint" id="hint"></div>
  <button class="rec" id="rec">● 开始录音</button>
  <div class="row">
    <a id="dl" class="disabled" download>下载这段</a>
    <button id="next">下一句 →</button>
  </div>
  <div class="status" id="status"></div>
</div>

<footer id="footer"></footer>

<script>
const SENTENCES = {json.dumps(rows, ensure_ascii=False)};
const MODES = {json.dumps(modes, ensure_ascii=False)};
let mode = 'whisper', idx = 0, media = null, chunks = [], blob = null, timer = null, started = 0;

const $ = (id) => document.getElementById(id);
const modeBar = $('modes');
MODES.forEach(m => {{
  const b = document.createElement('button');
  b.textContent = m.label; b.dataset.key = m.key;
  b.onclick = () => {{ mode = m.key; paint(); }};
  modeBar.appendChild(b);
}});

function current() {{ return SENTENCES[idx]; }}
function paint() {{
  const s = current();
  $('pos').textContent = (idx + 1) + ' / ' + SENTENCES.length + ' · ' + MODES.find(m => m.key === mode).label;
  $('cat').textContent = s.cat;
  $('text').textContent = s.text;
  $('hint').textContent = mode === 'whisper'
    ? '离嘴 3–5cm，只用气流，别出声带振动。保持和平时一样的语速。'
    : (mode === 'low' ? '小声说，但正常出声（能听到嗓子）。' : '正常音量，和打电话差不多。');
  $('bar').style.width = ((idx) / SENTENCES.length * 100) + '%';
  document.querySelectorAll('#modes button').forEach(b => b.classList.toggle('on', b.dataset.key === mode));
  $('dl').classList.add('disabled'); blob = null;
  $('status').textContent = '';
  $('rec').classList.remove('on'); $('rec').textContent = '● 开始录音';
}}
paint();

$('rec').onclick = async () => {{
  if (media && media.state === 'recording') {{
    media.stop(); return;
  }}
  chunks = [];
  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{ audio: {{
      channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false
    }} }});
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/wav'];
    const mime = types.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
    media = mime ? new MediaRecorder(stream, {{ mimeType: mime, audioBitsPerSecond: 128000 }})
                 : new MediaRecorder(stream);
    media.ondataavailable = e => {{ if (e.data && e.data.size) chunks.push(e.data); }};
    media.onstop = () => {{
      clearInterval(timer);
      stream.getTracks().forEach(t => t.stop());
      const type = media.mimeType || 'audio/webm';
      const ext = type.includes('mp4') ? 'm4a' : (type.includes('wav') ? 'wav' : 'webm');
      blob = new Blob(chunks, {{ type }});
      const name = mode + '-' + current().id + '.' + ext;
      const a = $('dl');
      a.href = URL.createObjectURL(blob); a.download = name; a.classList.remove('disabled');
      const secs = ((Date.now() - started) / 1000).toFixed(1);
      $('status').innerHTML = '<span class="done">录好了：' + name + ' · ' + secs + 's</span>\\n点「下载这段」存到手机；然后「下一句」。';
      $('rec').classList.remove('on'); $('rec').textContent = '● 重录';
    }};
    media.start();
    started = Date.now();
    $('rec').classList.add('on'); $('rec').textContent = '■ 停止（说话中…）';
    $('status').textContent = '录音中…';
    timer = setInterval(() => {{ $('status').textContent = '录音中 ' + ((Date.now() - started) / 1000).toFixed(1) + 's'; }}, 200);
  }} catch (e) {{
    $('status').innerHTML = '<span class="warn">拿不到麦克风：' + e.message + '\\n需要 https 或 localhost 打开，或允许浏览器麦克风权限。</span>';
  }}
}};

$('next').onclick = () => {{
  if (media && media.state === 'recording') {{ media.stop(); }}
  blob = null; $('dl').classList.add('disabled');
  idx = (idx + 1) % SENTENCES.length;
  paint();
}};

window.addEventListener('load', () => {{
  if (!navigator.mediaDevices || !window.MediaRecorder) {{
    $('status').innerHTML = '<span class="warn">这个浏览器不支持网页录音，请用系统录音机，或换 Chrome/Safari 较新版本。</span>';
  }}
}});
</script>
</body>
</html>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    n = len(load_rows())
    print(f"生成 {OUT}（{n} 句 × {len(MODES)} 种模式 = {n * len(MODES)} 段）")


if __name__ == "__main__":
    main()