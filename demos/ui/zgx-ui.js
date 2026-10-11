/* ZGX demo chrome — builds the shared bar and owns every poll.
 *
 * One rule this file will not bend: a value that was not received renders as an
 * em-dash, never as 0. Zero watts, zero degrees and zero tokens are all readings
 * an audience will believe. This project has repeatedly found meters that showed
 * a plausible number while measuring nothing — a hardcoded "102 GB / 128 GB"
 * captioned as live NVML, a launcher bar of typed-in green ticks, a token count
 * that was a chars/4 guess. The chrome does not add to that list.
 *
 * Data sources, both optional:
 *   GET  {launcher}/telemetry   host GPU/CPU — read by the BROWSER, not the
 *                               demo's backend, so containerised demos with no
 *                               GPU tooling and no host route still show it.
 *   GET  ./api/metrics          the demo's own tokens. 404 = no LLM, panel hidden.
 *   GET  {launcher}/sysinfo     device identity.
 */
(function () {
  "use strict";

  var mount = document.getElementById("zgx-chrome");
  if (!mount || window.__zgxChromeLoaded) return;
  window.__zgxChromeLoaded = true;

  var LAUNCHER = "http://" + location.hostname + ":8086";
  var DASH = "—";

  var cfg = {
    title: mount.getAttribute("data-title") || document.title || "ZGX Demo",
    subtitle: mount.getAttribute("data-subtitle") || "",
    logo: mount.getAttribute("data-logo") || "hp_logo.png",
    metrics: mount.getAttribute("data-metrics") || "/api/metrics",
    foot: mount.getAttribute("data-foot") !== "off"
  };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function meter(id, label, sub) {
    var m = el("div", "zgx-meter");
    m.appendChild(el("span", "zgx-meter-label", label));
    var v = el("span", "zgx-meter-value zgx-unknown", DASH);
    v.id = "zgx-" + id;
    m.appendChild(v);
    var s = el("span", "zgx-meter-sub", sub || "");
    s.id = "zgx-" + id + "-sub";
    m.appendChild(s);
    return m;
  }

  // ── build ────────────────────────────────────────────────────────────────
  var bar = el("div", "zgx-bar");

  var plate = el("div", "zgx-logo-plate");
  var img = document.createElement("img");
  img.src = cfg.logo;
  img.alt = "HP";
  // A demo served from a subpath may not resolve the relative logo; fall back
  // to the launcher's copy rather than showing a broken-image glyph on stage.
  img.onerror = function () {
    if (img.dataset.fellBack) { plate.style.display = "none"; return; }
    img.dataset.fellBack = "1";
    img.src = LAUNCHER + "/ui/brand/hp_logo.png";
  };
  plate.appendChild(img);
  bar.appendChild(plate);

  var titles = el("div", "zgx-titles");
  titles.appendChild(el("div", "zgx-title", cfg.title));
  if (cfg.subtitle) titles.appendChild(el("div", "zgx-subtitle", cfg.subtitle));
  bar.appendChild(titles);

  var meters = el("div", "zgx-meters");
  var tokenPanel = el("div", "zgx-meters");          // hidden until /api/metrics answers
  tokenPanel.style.display = "none";
  tokenPanel.style.flex = "0 0 auto";
  tokenPanel.appendChild(meter("tokens", "Tokens", "this session"));
  tokenPanel.appendChild(meter("tps", "Tok/s", ""));
  meters.appendChild(tokenPanel);
  meters.appendChild(el("div", "zgx-sep"));
  meters.appendChild(meter("gpu", "GPU", ""));
  meters.appendChild(meter("power", "Power", ""));
  meters.appendChild(meter("temp", "Temp", ""));
  meters.appendChild(meter("mem", "Memory", ""));
  meters.appendChild(meter("cpu", "CPU", ""));
  bar.appendChild(meters);

  bar.appendChild(el("div", "zgx-sep"));
  var dev = el("div", "zgx-device");
  var devLabel = el("div", "zgx-device-label", DASH);
  devLabel.id = "zgx-device-label";
  var devSub = el("div", "zgx-device-sub", "");
  devSub.id = "zgx-device-sub";
  dev.appendChild(devLabel);
  dev.appendChild(devSub);
  bar.appendChild(dev);

  document.body.appendChild(bar);
  document.body.classList.add("zgx-chromed");

  if (cfg.foot) {
    var foot = el("div", "zgx-foot");
    var left = el("span", "zgx-local", "All inference on this device · no cloud");
    var right = el("span", null, "");
    right.id = "zgx-foot-right";
    foot.appendChild(left);
    foot.appendChild(right);
    document.body.appendChild(foot);
    document.body.classList.add("zgx-chromed-foot");
  }

  // ── render helpers ───────────────────────────────────────────────────────
  function set(id, value, unit, digits) {
    var n = document.getElementById("zgx-" + id);
    if (!n) return;
    if (value === null || value === undefined || value === "" || Number.isNaN(value)) {
      n.textContent = DASH;
      n.classList.add("zgx-unknown");
      return;
    }
    var shown = (typeof value === "number" && digits !== undefined) ? value.toFixed(digits) : value;
    n.textContent = shown + (unit || "");
    n.classList.remove("zgx-unknown");
  }
  function setSub(id, text) {
    var n = document.getElementById("zgx-" + id + "-sub");
    if (n) n.textContent = text == null ? "" : text;
  }

  // ── hardware, from the launcher ──────────────────────────────────────────
  function pollTelemetry() {
    fetch(LAUNCHER + "/telemetry", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (t) {
        if (!t) throw new Error("no telemetry");
        set("gpu", t.util_pct, "%");
        set("power", t.power_w, " W", 0);
        set("temp", t.temp_c, "°C");
        set("mem", t.mem_used_gb, " GB", 1);
        setSub("mem", t.mem_total_gb ? "of " + t.mem_total_gb + " GB" : "");
        set("cpu", t.cpu_pct, "%", 0);
        setSub("cpu", t.ram_pct != null ? "RAM " + Math.round(t.ram_pct) + "%" : "");
        setSub("gpu", t.source && t.source !== "none" ? t.source : "");
        var fr = document.getElementById("zgx-foot-right");
        if (fr) {
          fr.textContent = [t.vendor && t.vendor !== "unknown" ? t.vendor : null,
                            t.source && t.source !== "none" ? t.source : null]
                           .filter(Boolean).join(" · ");
        }
      })
      .catch(function () {
        // Launcher down: clear every hardware meter. Leaving the last reading
        // on screen would be a stale number presented as live.
        ["gpu", "power", "temp", "mem", "cpu"].forEach(function (k) { set(k, null); setSub(k, ""); });
      });
  }

  // ── tokens, from the demo itself ─────────────────────────────────────────
  var tokensSupported = null;
  function pollMetrics() {
    if (tokensSupported === false) return;
    fetch(cfg.metrics, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (m) {
        if (!m || typeof m.total_tokens !== "number") throw new Error("no metrics");
        tokensSupported = true;
        tokenPanel.style.display = "";
        var approx = m.estimated ? "≈" : "";
        set("tokens", approx + m.total_tokens.toLocaleString());
        setSub("tokens", m.requests != null ? m.requests + " request" + (m.requests === 1 ? "" : "s") : "this session");
        set("tps", m.tps, "", 1);
      })
      .catch(function () {
        // A demo with no LLM serves no /api/metrics. Hide the panel rather than
        // show a zero that would read as "this demo processed nothing".
        if (tokensSupported === null) { tokensSupported = false; tokenPanel.style.display = "none"; }
      });
  }

  function loadIdentity() {
    fetch(LAUNCHER + "/sysinfo", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (hw) {
        if (!hw) return;
        var label = document.getElementById("zgx-device-label");
        var sub = document.getElementById("zgx-device-sub");
        label.textContent = label.title = hw.device_label || DASH;
        // Some vendors' device names run long (marketing name plus arch id); the CSS
        // ellipsizes both lines and the title keeps the full text on hover.
        sub.textContent = sub.title =
          [hw.accelerator !== "unknown" ? hw.accelerator : null, hw.arch].filter(Boolean).join(" · ");
      })
      .catch(function () {
        document.getElementById("zgx-device-label").textContent = "hardware unknown";
      });
  }

  loadIdentity();
  pollTelemetry();
  pollMetrics();
  setInterval(pollTelemetry, 2000);
  setInterval(pollMetrics, 2000);
})();
