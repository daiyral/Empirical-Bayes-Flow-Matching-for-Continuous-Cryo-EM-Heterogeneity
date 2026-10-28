(function () {
  var videos = Array.prototype.slice.call(document.querySelectorAll('video'));

  function markMissing(video) {
    var card = video.closest('.video-card');
    if (card) card.classList.add('is-missing');
  }

  function play(video) {
    video.muted = true;
    var p = video.play();
    if (p && p.catch) p.catch(function () {});
  }

  // loops only run while they are on (or about to scroll onto) the screen
  var observer = 'IntersectionObserver' in window
    ? new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          var video = entry.target;
          video.dataset.offscreen = entry.isIntersecting ? '' : '1';
          if (entry.isIntersecting) play(video);
          else video.pause();
        });
      }, { rootMargin: '200px 0px' })
    : null;

  videos.forEach(function (video) {
    video.addEventListener('error', function () { markMissing(video); });
    video.addEventListener('loadedmetadata', function () {
      if (video.videoWidth && video.videoHeight) {
        video.style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
      }
    });
    // networkState is not checked here: it still reads NETWORK_NO_SOURCE until the browser starts the fetch
    if (video.error) {
      markMissing(video);
    } else if (observer) {
      observer.observe(video);
    } else {
      play(video);
    }
  });

  // browsers that block autoplay (e.g. low-power mode) start the loops on the first interaction
  function kick() {
    videos.forEach(function (video) { if (video.paused && !video.dataset.offscreen) play(video); });
    document.removeEventListener('touchstart', kick);
    document.removeEventListener('click', kick);
  }
  document.addEventListener('touchstart', kick, { passive: true });
  document.addEventListener('click', kick);

  Array.prototype.forEach.call(document.querySelectorAll('.copy-button'), function (button) {
    button.addEventListener('click', function () {
      var target = document.getElementById(button.getAttribute('data-copy-target'));
      if (!target || !navigator.clipboard) return;
      navigator.clipboard.writeText(target.textContent).then(function () {
        var label = button.textContent;
        button.textContent = 'Copied';
        setTimeout(function () { button.textContent = label; }, 1500);
      });
    });
  });
})();

// Hero pipeline: the 3D output starts as noise and slowly turns into the ensemble video, while its
// rendered image goes from a projection of that noise to a clean counterpart of the dataset image.
// Every lit pixel of the current video frame is a particle that slides in a straight line from a fixed
// random spot to its place in the frame. Without this script the video simply plays.
(function () {
  var pipeline = document.querySelector('.pipeline');
  var video = pipeline && pipeline.querySelector('.pipeline-video');
  var canvas = pipeline && pipeline.querySelector('.sample-canvas');
  var renderCanvas = pipeline && pipeline.querySelector('.render-canvas');
  if (!video || !canvas || !renderCanvas || !canvas.getContext) return;

  // seconds: pure noise, noise -> structure, clean structure, back to noise
  var HOLD_NOISE = 1.1, MORPH = 8.7, HOLD_CLEAN = 4.3, RESET = 0.9;
  var LOOP = HOLD_NOISE + MORPH + HOLD_CLEAN + RESET;
  var STEP = 1.5, FADE = 0.4;   // seconds each dataset image stays up, and its cross-fade into the next
  var FPS = 24;
  var SCALE = 0.18;      // rendered-image pixels per canvas pixel
  var CONTRAST = 210;    // grey levels a fully covered pixel moves the rendered image by

  var W = canvas.width, H = canvas.height, R = renderCanvas.width;
  var ctx = canvas.getContext('2d');
  var rctx = renderCanvas.getContext('2d');
  var grab = document.createElement('canvas');
  grab.width = W;
  grab.height = H;
  var gctx = grab.getContext('2d', { willReadFrequently: true });
  var frameOut = ctx.createImageData(W, H);
  var out32 = new Uint32Array(frameOut.data.buffer);
  var renderOut = rctx.createImageData(R, R);
  var tiles = Array.prototype.slice.call(pipeline.querySelectorAll('.dataset span'));
  var weightBox = pipeline.querySelector('.nn-weights');
  var weightCells = weightBox ? Array.prototype.slice.call(weightBox.querySelectorAll('text')) : [];
  var still = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  // where each pixel's particle sits when the output is pure noise: one Gaussian blob
  var noiseX = new Float32Array(W * H), noiseY = new Float32Array(W * H);
  (function () {
    var sigma = 0.165 * W, i, radius, angle;
    for (i = 0; i < W * H; i++) {
      radius = sigma * Math.sqrt(-2 * Math.log(1 - Math.random()));
      angle = 2 * Math.PI * Math.random();
      noiseX[i] = W / 2 + radius * Math.cos(angle);
      noiseY[i] = H / 2 + radius * Math.sin(angle);
    }
  })();

  var density = new Float32Array(R * R), fine = new Float32Array(R * R), wide = new Float32Array(R * R);
  var wider = new Float32Array(R * R), scratch = new Float32Array(R * R);
  var FULL = 255 / (SCALE * SCALE);   // density of a rendered-image pixel whose whole footprint is lit

  function kernel(sigma) {
    var reach = Math.ceil(sigma * 2.5), k = [], sum = 0, i;
    for (i = -reach; i <= reach; i++) {
      k.push(Math.exp(-i * i / (2 * sigma * sigma)));
      sum += k[k.length - 1];
    }
    return k.map(function (v) { return v / sum; });
  }
  var K_FINE = kernel(1.2), K_WIDE = kernel(3), K_WIDER = kernel(6.5);
  var K_TARGET = kernel(0.7);   // light: enough to take the grain off a dataset image without softening it

  function blur(src, dst, k) {
    var reach = (k.length - 1) / 2, x, y, j, p, acc;
    for (y = 0; y < R; y++) {
      for (x = 0; x < R; x++) {
        for (acc = 0, j = -reach; j <= reach; j++) {
          p = x + j;
          acc += src[y * R + (p < 0 ? 0 : p >= R ? R - 1 : p)] * k[j + reach];
        }
        scratch[y * R + x] = acc;
      }
    }
    for (y = 0; y < R; y++) {
      for (x = 0; x < R; x++) {
        for (acc = 0, j = -reach; j <= reach; j++) {
          p = y + j;
          acc += scratch[(p < 0 ? 0 : p >= R ? R - 1 : p) * R + x] * k[j + reach];
        }
        dst[y * R + x] = acc;
      }
    }
  }

  // what each dataset image's rendered counterpart ends up as: the same image with its noise smoothed away
  var targets = [];
  (function () {
    var url = /url\(["']?(.*?)["']?\)/.exec(tiles.length ? getComputedStyle(tiles[0]).backgroundImage : '');
    var sprite = new Image();
    if (!url) return;
    sprite.onload = function () {
      var c = document.createElement('canvas'), cx, tw = sprite.naturalWidth / 4, th = sprite.naturalHeight / 3;
      var loaded = [], raw = new Float32Array(R * R), n, i, px, smooth;
      c.width = c.height = R;
      cx = c.getContext('2d', { willReadFrequently: true });
      try {
        for (n = 0; n < tiles.length; n++) {
          cx.drawImage(sprite, tiles[n].style.getPropertyValue('--c') * tw, tiles[n].style.getPropertyValue('--r') * th,
            tw, th, 0, 0, R, R);
          px = cx.getImageData(0, 0, R, R).data;
          for (i = 0; i < R * R; i++) raw[i] = px[i * 4];
          smooth = new Float32Array(R * R);
          blur(raw, smooth, K_TARGET);
          loaded.push(smooth);
        }
        targets = loaded;
      } catch (e) {}
    };
    sprite.src = url[1];
  })();

  // u: 0 = pure noise, 1 = the video as it is; a, b: the dataset images on show, mix: how far a has faded into b
  function draw(u, a, b, mix) {
    var k = Math.pow(1 - u, 2.2);            // share of noise left in each particle's position
    var jitter = 2.5 * k, lift = 0.6 * k;    // noise flickers, and is drawn brighter than the structure
    var match = targets.length ? u * u * (3 - 2 * u) : 0;   // how far the rendered image has moved to its target
    var data, i, x, y, g, px, py, xi, yi, o, rx, ry, v, clean;

    gctx.drawImage(video, 0, 0, W, H);
    data = gctx.getImageData(0, 0, W, H).data;
    out32.fill(0);
    density.fill(0);
    for (y = 0, i = 0; y < H; y++) {
      for (x = 0; x < W; x++, i++) {
        g = data[i * 4 + 1];
        if (g < 48) continue;
        // the rendered image projects the particle's noise position, wherever the particle itself has got to
        rx = Math.floor((noiseX[i] + (Math.random() - 0.5) * 2.5 - W / 2) * SCALE + R / 2);
        ry = Math.floor((noiseY[i] + (Math.random() - 0.5) * 2.5 - H / 2) * SCALE + R / 2);
        if (rx >= 0 && rx < R && ry >= 0 && ry < R) density[ry * R + rx] += g;
        px = x + (noiseX[i] - x) * k + (Math.random() - 0.5) * jitter;
        py = y + (noiseY[i] - y) * k + (Math.random() - 0.5) * jitter;
        xi = Math.round(px);
        yi = Math.round(py);
        if (xi < 0 || xi >= W || yi < 0 || yi >= H) continue;
        g += (255 - g) * lift;
        o = yi * W + xi;
        if (g > ((out32[o] >>> 8) & 255)) out32[o] = (0xff000000 | (g << 16) | (g << 8)) >>> 0;
      }
    }
    ctx.putImageData(frameOut, 0, 0);

    // the last stretch hands over to the full-resolution video
    clean = u >= 1 ? 1 : u > 0.9 ? (u - 0.9) / 0.1 : 0;
    video.style.opacity = clean;
    canvas.style.opacity = 1 - clean;

    // noise projected with a CTF-like band-pass (dark core, light fringe), blended towards the target
    blur(density, fine, K_FINE);
    blur(density, wide, K_WIDE);
    blur(density, wider, K_WIDER);
    data = renderOut.data;
    for (i = 0; i < R * R; i++) {
      v = 134 - CONTRAST * (fine[i] - 0.95 * wide[i] + 0.25 * wider[i]) / FULL;
      if (match) v += (targets[a][i] + (targets[b][i] - targets[a][i]) * mix - v) * match;
      data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = v < 0 ? 0 : v > 255 ? 255 : v;
      data[i * 4 + 3] = 255;
    }
    rctx.putImageData(renderOut, 0, 0);
  }

  // shown until the video delivers its first frame, so the structure never flashes up before the noise
  function paintNoise() {
    var n, i, xi, yi;
    out32.fill(0);
    for (n = 0; n < 5000; n++) {
      i = Math.floor(Math.random() * W * H);
      xi = Math.round(noiseX[i]);
      yi = Math.round(noiseY[i]);
      if (xi >= 0 && xi < W && yi >= 0 && yi < H) out32[yi * W + xi] = 0xffdcdc00;
    }
    ctx.putImageData(frameOut, 0, 0);
  }

  function progress(t) {
    if (t < HOLD_NOISE) return 0;
    t -= HOLD_NOISE;
    if (t < MORPH) return t / MORPH;
    t -= MORPH;
    if (t < HOLD_CLEAN) return 1;
    return 1 - (t - HOLD_CLEAN) / RESET;
  }

  var clock = 0, last = 0, raf = 0, dead = false, shown = [];
  var weights = weightCells.map(function () { return Math.random() * 3 - 1.5; }), weightsAt = 0;

  // the numbers on the edges of the network: training keeps nudging them, in steps that shrink as it converges
  function showWeights(u, now) {
    var rate = 1 - u;
    weightBox.classList.toggle('is-updating', rate > 0.04);
    if (rate <= 0.04 || now - weightsAt < 160) return;
    weightsAt = now;
    weightCells.forEach(function (cell, n) {
      if (Math.random() < 0.35) return;
      weights[n] = Math.max(-1.99, Math.min(1.99, weights[n] + (Math.random() - 0.5) * 0.9 * rate));
      cell.textContent = (weights[n] < 0 ? String.fromCharCode(8722) : '+') + Math.abs(weights[n]).toFixed(2);
    });
  }

  // one dataset image at a time, each cross-fading into the next
  function showTiles(a, b, mix) {
    shown.forEach(function (n) { if (n !== a && n !== b) tiles[n].style.opacity = 0; });
    tiles[a].style.opacity = 1 - mix;
    if (b !== a) tiles[b].style.opacity = mix;
    shown = [a, b];
  }

  function stop() {
    cancelAnimationFrame(raf);
    raf = 0;
  }

  // back to the plain video, e.g. when the page is opened from disk and the canvas may not read it
  function giveUp() {
    dead = true;
    stop();
    video.style.opacity = '';
    canvas.style.display = 'none';
  }

  function tick(now) {
    var turn, a, b, mix, u;
    raf = requestAnimationFrame(tick);
    if (now - last < 1000 / FPS || video.readyState < 2) return;
    clock += Math.min(now - last, 250) / 1000;
    last = now;
    turn = still || !tiles.length ? 0 : clock / STEP;
    a = Math.floor(turn) % (tiles.length || 1);
    b = (a + 1) % (tiles.length || 1);
    mix = Math.max(0, (turn % 1) * STEP - (STEP - FADE)) / FADE;
    u = still ? 1 : progress(clock % LOOP);
    try {
      draw(u, a, b, mix);
    } catch (e) {
      giveUp();
      return;
    }
    if (tiles.length) showTiles(a, b, mix);
    if (weightCells.length && !still) showWeights(u, now);
  }

  function start() {
    if (raf || dead) return;
    last = performance.now();
    raf = requestAnimationFrame(tick);
  }

  video.addEventListener('playing', start);
  video.addEventListener('pause', stop);
  video.addEventListener('error', giveUp);
  if (!still) {
    video.style.opacity = 0;
    paintNoise();
  }
  if (!video.paused) start();
})();

// Feedback arrows: guided sampling, from "compare" back to the flow matching step, and training, from the
// 3D atoms back to the prior alongside the flow matching arrow. They are drawn here rather than in the
// stylesheet because where they run depends on where the layout has put their ends.
(function () {
  var flow = document.querySelector('.pipeline-flow');
  var pill = flow && flow.querySelector('.compare .pill');
  var caption = flow && flow.querySelector('.arrow-label');
  var track = flow && flow.querySelector('.pipeline-arrow .track');   // the flow matching arrow
  if (!pill || !caption || !track || !document.createElementNS) return;

  var NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name), key;
    for (key in attrs) node.setAttribute(key, attrs[key]);
    parent.appendChild(node);
    return node;
  }

  var svg = el('svg', { 'class': 'feedback', 'aria-hidden': 'true' }, flow);
  var marker = el('marker', {
    id: 'feedback-head', viewBox: '0 0 10 10', refX: 7, refY: 5,
    markerWidth: 11, markerHeight: 11, markerUnits: 'userSpaceOnUse', orient: 'auto'
  }, el('defs', {}, svg));
  el('path', { d: 'M0 0L10 5L0 10z' }, marker);

  function arrow(text) {
    var line = el('path', { 'class': 'feedback-line', 'marker-end': 'url(#feedback-head)' }, svg);
    var label = el('text', {}, svg);
    label.textContent = text;
    return { line: line, label: label };
  }
  var guided = arrow('guided sampling'), training = arrow('training');
  flow.classList.add('has-feedback');   // opens the lane the guided sampling arrow runs in

  // where a node sits inside the flow box
  function place(node, box) {
    var b = node.getBoundingClientRect();
    return {
      left: b.left - box.left, right: b.right - box.left, top: b.top - box.top, bottom: b.bottom - box.top,
      x: b.left + b.width / 2 - box.left, y: b.top + b.height / 2 - box.top
    };
  }

  function draw(arrow, d, x, y, anchor) {
    arrow.line.setAttribute('d', d);
    arrow.label.setAttribute('x', x);
    arrow.label.setAttribute('y', y);
    arrow.label.style.textAnchor = anchor || '';
  }

  function layout() {
    var box = flow.getBoundingClientRect(), r = 10;
    var a = place(pill, box), b = place(caption, box), t = place(track, box);
    var sx, sy, lane, edge, ex, ey;
    if (getComputedStyle(flow).flexDirection === 'column') {
      // guided sampling, stacked: down, along the bottom, up the right edge, and in beside the caption
      sx = a.x;
      sy = a.bottom + 4;
      lane = box.height - 12;
      edge = box.width - 9;
      ex = b.right + 9;
      ey = b.y;
      draw(guided, 'M' + sx + ' ' + sy + 'V' + (lane - r) + 'Q' + sx + ' ' + lane + ' ' + (sx + r) + ' ' + lane +
        'H' + (edge - r) + 'Q' + edge + ' ' + lane + ' ' + edge + ' ' + (lane - r) +
        'V' + (ey + r) + 'Q' + edge + ' ' + ey + ' ' + (edge - r) + ' ' + ey + 'H' + ex, (sx + edge) / 2, lane);

      // training, stacked: back up beside the flow matching arrow, on the side away from its caption
      ex = t.x - 30;
      draw(training, 'M' + ex + ' ' + (t.bottom - 2) + 'V' + (t.top + 3), ex - 10, t.y, 'end');
    } else {
      // guided sampling, side by side: up, along the top, and down onto the caption
      sx = a.x;
      sy = a.top - 4;
      lane = 12;
      ex = b.x;
      ey = b.top - 9;
      draw(guided, 'M' + sx + ' ' + sy + 'V' + (lane + r) + 'Q' + sx + ' ' + lane + ' ' + (sx - r) + ' ' + lane +
        'H' + (ex + r) + 'Q' + ex + ' ' + lane + ' ' + ex + ' ' + (lane + r) + 'V' + ey, (sx + ex) / 2, lane);

      // training, side by side: back along the flow matching arrow, just under it, with its label beneath
      ey = t.y + 30;
      draw(training, 'M' + (t.right - 2) + ' ' + ey + 'H' + (t.left + 3), t.x, ey + 16);
    }
  }

  layout();
  if (window.ResizeObserver) new ResizeObserver(layout).observe(flow);
  window.addEventListener('resize', layout);
  window.addEventListener('load', layout);
})();
