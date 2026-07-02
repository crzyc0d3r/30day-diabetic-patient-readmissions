/**
 * Neural Mesh Background: Pancreatic Islet Cell Edition
 *
 * A living signaling network visualized as beta cells / islet cells communicating.
 * Nodes are rendered as cells with nuclei and insulin granules. Traveling particles
 * represent secreted vesicles and signaling molecules moving between cells.
 *
 * Inspired by the animated field on https://build.nvidia.com/spark,
 * thematically aligned with the medi-watch diabetes readmission project.
 *
 * Features:
 * - Cell-like nodes (membrane + nucleus + granules)
 * - Vesicles traveling along connections (the "communication" effect)
 * - Theme-aware via --accent and data-theme
 * - Mouse interaction + pause on resize / tab hidden
 * - Respects prefers-reduced-motion
 * - Low CPU
 */

(function () {
  'use strict';

  const NeuralMeshBackground = {
    canvas: null,
    ctx: null,
    nodes: [],
    edges: [],
    particles: [],
    rafId: null,
    running: false,
    lastTime: 0,
    accentColor: '#5b9eff',
    membraneColor: '#e8e8f0',   // cell membrane stroke, theme-driven (see updateAccentColor)
    cellColor: '#f4f4ff',       // granule / vesicle highlight, theme-driven
    mouse: { x: 0, y: 0, active: false },
    reducedMotion: false,
    resizeObserver: null,

    config: {
      nodeCount: 78,
      connectionDistance: 185,
      longRangeChance: 0.018,
      particleCount: 46,
      baseOpacity: 0.29,        // overall layer opacity, kept low enough that text stays readable
      particleSpeedMin: 0.006,
      particleSpeedMax: 0.014,
      nodeDrift: 0.012,          // subtle organic movement
      glowIntensity: 0.85,
    },

    init(options = {}) {
      this.canvas = document.getElementById('neural-bg');
      if (!this.canvas) {
        console.warn('[NeuralMesh] No canvas#neural-bg found. Aborting.');
        return;
      }

      this.ctx = this.canvas.getContext('2d', { alpha: true });
      Object.assign(this.config, options);

      this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      this.updateAccentColor();
      this.setupThemeObserver();
      this.setupCanvas();
      this.buildGraph();
      this.createParticles();
      this.attachInteractions();
      this.start();

      // Gentle re-seed on long idle (keeps it feeling alive without obvious repetition)
      setInterval(() => {
        if (this.running && !this.reducedMotion) {
          this.jitterParticles(0.3);
        }
      }, 42000);
    },

    updateAccentColor() {
      const root = document.documentElement;
      const styles = getComputedStyle(root);
      const accent = styles.getPropertyValue('--accent').trim();
      if (accent) {
        this.accentColor = accent;
      }
      // Cell detail colors are theme-driven so they survive the dark(screen)→
      // light(multiply) blend switch instead of washing out. Fall back to the
      // dark-theme defaults if the tokens aren't present.
      const membrane = styles.getPropertyValue('--node-membrane').trim();
      if (membrane) this.membraneColor = membrane;
      const cell = styles.getPropertyValue('--node-cell').trim();
      if (cell) this.cellColor = cell;
    },

    setupThemeObserver() {
      const root = document.documentElement;
      const observer = new MutationObserver(() => {
        this.updateAccentColor();
      });
      observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    },

    setupCanvas() {
      // Rebuild + size the canvas, then resume animation.
      // We deliberately pause during active resize for smoothness.
      const doResize = () => {
        const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        const w = window.innerWidth;
        const h = window.innerHeight;

        this.canvas.width = Math.floor(w * dpr);
        this.canvas.height = Math.floor(h * dpr);
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';

        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Rebuild the node/edge graph so density stays reasonable at new size
        if (this.nodes.length > 0) {
          this.buildGraph();
          this.respawnParticlesForNewSize();
        }

        // Only resume the RAF loop after the user has stopped resizing
        if (!this.reducedMotion) {
          // Give particles a tiny random nudge so it doesn't look frozen when it comes back
          this.jitterParticles(0.25);
          this.resume();
        }
      };

      const onResize = () => {
        // Pause the animation the moment resizing begins.
        // This prevents janky particle motion and high CPU while the user drags the window.
        this.pause();
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(doResize, 180);
      };

      window.addEventListener('resize', onResize, { passive: true });

      // Mobile orientation changes can be abrupt, so allow more settle time
      window.addEventListener('orientationchange', () => {
        this.pause();
        setTimeout(doResize, 280);
      }, { passive: true });

      // Initial layout
      doResize();
    },

    // Procedural but aesthetically pleasing node placement
    buildGraph() {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const nodes = [];
      const edges = [];

      const { nodeCount, connectionDistance, longRangeChance } = this.config;

      // Create nodes with slight clustering for more interesting "talking" regions
      for (let i = 0; i < nodeCount; i++) {
        let x, y;
        const cluster = Math.random() < 0.55;

        if (cluster) {
          // clustered around several focal points
          const cx = 120 + Math.random() * (w - 240);
          const cy = 80 + Math.random() * (h - 160);
          const radius = 90 + Math.random() * 210;
          const angle = Math.random() * Math.PI * 2;
          x = cx + Math.cos(angle) * radius * (0.4 + Math.random() * 0.9);
          y = cy + Math.sin(angle) * radius * (0.35 + Math.random() * 0.85);
        } else {
          x = 60 + Math.random() * (w - 120);
          y = 60 + Math.random() * (h - 120);
        }

        // bias a few larger "hub" neurons (think small islet cell clusters)
        const isHub = Math.random() < 0.09;
        const r = isHub ? 2.8 + Math.random() * 1.9 : 1.15 + Math.random() * 1.35;

        // Pre-generate stable insulin granule positions + nucleus for cell-like appearance
        const granuleCount = isHub ? 8 + Math.floor(Math.random() * 5) : 3 + Math.floor(Math.random() * 3);
        const granules = [];
        for (let g = 0; g < granuleCount; g++) {
          const ang = Math.random() * Math.PI * 2;
          const dist = 0.22 + Math.random() * 0.58;
          granules.push({
            dx: Math.cos(ang) * dist,
            dy: Math.sin(ang) * dist,
            r: 0.32 + Math.random() * 0.38
          });
        }
        const nucleus = {
          dx: (Math.random() - 0.5) * 0.32,
          dy: (Math.random() - 0.5) * 0.32,
          r: isHub ? 0.52 : 0.42
        };

        nodes.push({
          x,
          y,
          r,
          vx: (Math.random() - 0.5) * this.config.nodeDrift,
          vy: (Math.random() - 0.5) * this.config.nodeDrift,
          isHub,
          pulse: 0,
          granules,
          nucleus
        });
      }

      // Build edges
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.hypot(dx, dy);

          const isLong = dist > connectionDistance * 1.65 && Math.random() < longRangeChance;
          const isNormal = dist < connectionDistance && Math.random() < 0.82;

          if (isNormal || isLong) {
            edges.push({ a: i, b: j, dist, isLong });
          }
        }
      }

      this.nodes = nodes;
      this.edges = edges;
    },

    createParticles() {
      const particles = [];
      const count = this.config.particleCount;

      for (let i = 0; i < count; i++) {
        const edgeIdx = Math.floor(Math.random() * this.edges.length);
        const edge = this.edges[edgeIdx];
        if (!edge) continue;

        particles.push({
          edgeIdx,
          t: Math.random(),                    // position along edge [0, 1]
          speed: this.config.particleSpeedMin +
                 Math.random() * (this.config.particleSpeedMax - this.config.particleSpeedMin),
          size: 0.9 + Math.random() * 1.35,
          dir: Math.random() > 0.5 ? 1 : -1,   // some travel backwards for richness
          life: 1.0,
        });
      }

      this.particles = particles;
    },

    respawnParticlesForNewSize() {
      // Keep roughly the same density after resize
      const target = this.config.particleCount;
      while (this.particles.length < target && this.edges.length > 0) {
        const edgeIdx = Math.floor(Math.random() * this.edges.length);
        this.particles.push({
          edgeIdx,
          t: Math.random(),
          speed: this.config.particleSpeedMin + Math.random() * (this.config.particleSpeedMax - this.config.particleSpeedMin),
          size: 0.9 + Math.random() * 1.35,
          dir: Math.random() > 0.5 ? 1 : -1,
          life: 1.0,
        });
      }
      // Trim if we have far too many after a shrink
      if (this.particles.length > target * 1.6) {
        this.particles.length = Math.floor(target * 1.35);
      }
    },

    jitterParticles(fraction = 0.25) {
      const n = Math.floor(this.particles.length * fraction);
      for (let i = 0; i < n; i++) {
        const p = this.particles[i];
        if (p) p.t = Math.random();
      }
    },

    attachInteractions() {
      const canvas = this.canvas;
      if (!canvas) return;

      const onMove = (e) => {
        this.mouse.x = e.clientX;
        this.mouse.y = e.clientY;
        this.mouse.active = true;

        // Occasionally emit a bright "signal" near the cursor
        if (!this.reducedMotion && this.edges.length > 0 && Math.random() < 0.14) {
          this.spawnParticleNearCursor();
        }

        // Gentle influence on nearby nodes (repulsion)
        this.applyMouseInfluence();
      };

      const onLeave = () => {
        this.mouse.active = false;
      };

      window.addEventListener('mousemove', onMove, { passive: true });
      window.addEventListener('mouseleave', onLeave);

      // Touch support (lighter)
      window.addEventListener('touchmove', (e) => {
        if (e.touches.length > 0) {
          this.mouse.x = e.touches[0].clientX;
          this.mouse.y = e.touches[0].clientY;
          this.mouse.active = true;
        }
      }, { passive: true });

      // Pause when the tab is hidden to save battery and CPU
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          this.pause();
        } else {
          this.resume();
        }
      });
    },

    spawnParticleNearCursor() {
      if (!this.edges.length) return;
      // Find a nearby edge roughly under the cursor
      let best = -1;
      let bestDist = Infinity;

      for (let i = 0; i < this.edges.length; i++) {
        const e = this.edges[i];
        const n1 = this.nodes[e.a];
        const n2 = this.nodes[e.b];
        const mx = (n1.x + n2.x) * 0.5;
        const my = (n1.y + n2.y) * 0.5;
        const d = Math.hypot(mx - this.mouse.x, my - this.mouse.y);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }

      if (best !== -1 && bestDist < 380) {
        this.particles.push({
          edgeIdx: best,
          t: 0.15 + Math.random() * 0.7,
          speed: this.config.particleSpeedMax * (0.7 + Math.random() * 0.6),
          size: 1.6 + Math.random() * 0.9,
          dir: Math.random() > 0.5 ? 1 : -1,
          life: 0.9 + Math.random() * 0.6,
        });
      }
    },

    applyMouseInfluence() {
      if (!this.mouse.active) return;
      const strength = 0.6;
      for (let i = 0; i < this.nodes.length; i++) {
        const n = this.nodes[i];
        const dx = n.x - this.mouse.x;
        const dy = n.y - this.mouse.y;
        const dist = Math.hypot(dx, dy) + 0.1;
        if (dist < 210) {
          const force = (1 - dist / 210) * strength;
          n.vx += (dx / dist) * force * 0.8;
          n.vy += (dy / dist) * force * 0.8;
        }
      }
    },

    start() {
      if (this.running) return;
      this.running = true;
      this.lastTime = performance.now();
      this.loop();
    },

    pause() {
      this.running = false;
      if (this.rafId) cancelAnimationFrame(this.rafId);
    },

    resume() {
      if (!this.running) {
        this.running = true;
        this.lastTime = performance.now();
        this.loop();
      }
    },

    loop() {
      if (!this.running) return;

      const now = performance.now();
      const dt = Math.min((now - this.lastTime) / 16.67, 2.8); // clamp huge frames
      this.lastTime = now;

      this.update(dt);
      this.draw();

      this.rafId = requestAnimationFrame(() => this.loop());
    },

    update(dt) {
      const { nodes, edges, particles, config } = this;
      const w = window.innerWidth;
      const h = window.innerHeight;

      // Subtle node drift + mouse spring back + boundary
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];

        // idle drift
        if (!this.reducedMotion) {
          n.x += n.vx * dt;
          n.y += n.vy * dt;

          // light spring toward the original position (prevents drift to infinity)
          const cx = 80 + (i % 7) * ((w - 160) / 7);
          const cy = 70 + Math.floor(i / 9) * ((h - 140) / 9);
          n.vx *= 0.978;
          n.vy *= 0.978;
          n.vx += (cx - n.x) * 0.000012 * dt;
          n.vy += (cy - n.y) * 0.000012 * dt;
        }

        // boundary bounce (soft)
        if (n.x < 40) { n.x = 40; n.vx = Math.abs(n.vx) * 0.6; }
        if (n.x > w - 40) { n.x = w - 40; n.vx = -Math.abs(n.vx) * 0.6; }
        if (n.y < 40) { n.y = 40; n.vy = Math.abs(n.vy) * 0.6; }
        if (n.y > h - 40) { n.y = h - 40; n.vy = -Math.abs(n.vy) * 0.6; }

        // decay activation pulse
        if (n.pulse > 0) n.pulse = Math.max(0, n.pulse - 0.018 * dt);
      }

      // Update particles
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        const edge = edges[p.edgeIdx];
        if (!edge) {
          particles.splice(i, 1);
          continue;
        }

        p.t += p.speed * p.dir * dt;

        // wrap or bounce at ends
        if (p.t > 1.02) {
          p.t = 1.02;
          p.dir = -1;
          // flash receiving node
          const target = nodes[edge.b];
          if (target) target.pulse = Math.min(1.6, target.pulse + 0.9);
        }
        if (p.t < -0.02) {
          p.t = -0.02;
          p.dir = 1;
          const target = nodes[edge.a];
          if (target) target.pulse = Math.min(1.6, target.pulse + 0.9);
        }

        p.life -= 0.0012 * dt;
        if (p.life <= 0) {
          // recycle
          p.edgeIdx = Math.floor(Math.random() * edges.length);
          p.t = Math.random() * 0.6 + 0.2;
          p.life = 0.7 + Math.random() * 0.9;
          p.speed = config.particleSpeedMin + Math.random() * (config.particleSpeedMax - config.particleSpeedMin);
        }
      }

      // Occasionally add a new particle for "chatter" density
      if (!this.reducedMotion && particles.length < config.particleCount * 1.15 && Math.random() < 0.06) {
        const edgeIdx = Math.floor(Math.random() * edges.length);
        particles.push({
          edgeIdx,
          t: 0.3 + Math.random() * 0.4,
          speed: config.particleSpeedMin + Math.random() * (config.particleSpeedMax - config.particleSpeedMin),
          size: 1.0 + Math.random() * 0.9,
          dir: Math.random() > 0.5 ? 1 : -1,
          life: 0.85 + Math.random() * 0.6,
        });
      }

      // Trim excess particles
      if (particles.length > config.particleCount * 1.7) {
        particles.length = Math.floor(config.particleCount * 1.5);
      }
    },

    draw() {
      const ctx = this.ctx;
      const w = window.innerWidth;
      const h = window.innerHeight;
      const { nodes, edges, particles, accentColor, membraneColor, cellColor, config } = this;

      ctx.clearRect(0, 0, w, h);

      const alpha = this.reducedMotion ? 0.22 : config.baseOpacity;
      const lineAlpha = this.reducedMotion ? 0.06 : 0.085;

      // === Draw faint connection mesh ===
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 0.7;

      for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        const n1 = nodes[e.a];
        const n2 = nodes[e.b];

        const alphaMod = e.isLong ? 0.28 : 1.0;
        ctx.globalAlpha = lineAlpha * alphaMod * alpha;

        ctx.beginPath();
        ctx.moveTo(n1.x, n1.y);
        ctx.lineTo(n2.x, n2.y);
        ctx.stroke();
      }

      // === Draw nodes as pancreatic beta cells / islet cells (with insulin granules) ===
      // This keeps the abstract "signaling network" feeling while fitting the diabetes/medi-watch theme.
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const activation = n.pulse * 0.7;

        // Soft biological halo (extracellular matrix glow)
        ctx.save();
        ctx.shadowColor = accentColor;
        ctx.shadowBlur = n.isHub ? 11 : 6;
        ctx.globalAlpha = (n.isHub ? 0.12 : 0.08) * alpha + activation * 0.18;

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r * (n.isHub ? 2.45 : 1.95) + activation * 2.3, 0, Math.PI * 2);
        ctx.fillStyle = accentColor;
        ctx.fill();
        ctx.restore();

        // Main cell body (cytoplasm)
        ctx.globalAlpha = (n.isHub ? 0.78 : 0.72) * alpha + activation * 0.25;
        ctx.fillStyle = accentColor;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + activation * 0.65, 0, Math.PI * 2);
        ctx.fill();

        // Thin cell membrane (theme-driven color)
        ctx.globalAlpha = 0.32 * alpha + activation * 0.18;
        ctx.strokeStyle = membraneColor;
        ctx.lineWidth = n.isHub ? 1.1 : 0.75;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + activation * 0.65, 0, Math.PI * 2);
        ctx.stroke();

        // Nucleus, dimmed
        ctx.globalAlpha = 0.42 * alpha + activation * 0.15;
        ctx.fillStyle = accentColor;
        const nx = n.x + (n.nucleus?.dx || 0) * n.r * 0.9;
        const ny = n.y + (n.nucleus?.dy || 0) * n.r * 0.9;
        ctx.beginPath();
        ctx.arc(nx, ny, (n.nucleus?.r || 0.4) * n.r + activation * 0.2, 0, Math.PI * 2);
        ctx.fill();

        // Insulin granules (small bright vesicles inside the beta cell)
        if (n.granules && n.granules.length) {
          ctx.fillStyle = cellColor;   // theme-driven granule highlight
          for (let g = 0; g < n.granules.length; g++) {
            const gr = n.granules[g];
            const gx = n.x + gr.dx * n.r;
            const gy = n.y + gr.dy * n.r;
            const grr = gr.r * (0.5 + activation * 0.7);
            ctx.globalAlpha = (0.38 + activation * 0.25) * alpha;   // significantly reduced brightness
            ctx.beginPath();
            ctx.arc(gx, gy, grr, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // === Draw traveling vesicles / secreted signaling packets ===
      // These represent insulin granules or signaling molecules moving between beta cells.
      ctx.shadowColor = accentColor;
      ctx.shadowBlur = 4;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const e = edges[p.edgeIdx];
        if (!e) continue;

        const n1 = nodes[e.a];
        const n2 = nodes[e.b];

        const t = Math.max(0, Math.min(1, p.t));
        const x = n1.x + (n2.x - n1.x) * t;
        const y = n1.y + (n2.y - n1.y) * t;

        const brightness = p.life * (0.65 + Math.sin(t * 11) * 0.12);

        // Small bright vesicle core (theme-driven color)
        ctx.globalAlpha = brightness * 0.65 * alpha;
        ctx.fillStyle = cellColor;
        ctx.beginPath();
        ctx.arc(x, y, p.size * 0.65, 0, Math.PI * 2);
        ctx.fill();

        // Subtle organic halo (less electric, more biological)
        ctx.globalAlpha = brightness * 0.42 * alpha;
        ctx.fillStyle = accentColor;
        ctx.beginPath();
        ctx.arc(x, y, p.size * 1.45, 0, Math.PI * 2);
        ctx.fill();

        // Short secretory tail (fading behind the vesicle)
        const tailT = Math.max(0, Math.min(1, t - 0.045 * p.dir));
        const tx = n1.x + (n2.x - n1.x) * tailT;
        const ty = n1.y + (n2.y - n1.y) * tailT;

        ctx.globalAlpha = brightness * 0.22 * alpha;
        ctx.fillStyle = accentColor;
        ctx.beginPath();
        ctx.arc(tx, ty, p.size * 0.95, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1.0;
    },
  };

  // Auto-init if the canvas exists on DOM ready
  function autoInit() {
    const canvas = document.getElementById('neural-bg');
    if (canvas) {
      NeuralMeshBackground.init();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit, { once: true });
  } else {
    // Already loaded
    setTimeout(autoInit, 0);
  }

  // Expose for manual control / debugging
  window.NeuralMeshBackground = NeuralMeshBackground;

  // Also expose a tiny helper to re-init after heavy DOM changes
  window.initNeuralBackground = () => NeuralMeshBackground.init();
})();
