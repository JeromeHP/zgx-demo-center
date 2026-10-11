# The ZGX demo UI standard

One chrome, every demo. The bar at the top, the meter strip and the footer are
**fixed and identical** across the suite; everything between them is the demo's
own. A demo author gets to design their demo — they do not get to redesign the
frame it sits in, because the frame is what makes eleven demos read as one
product on a stand.

## Opting in

One element, anywhere in `<body>`:

```html
<div id="zgx-chrome" data-title="NAWSCL Aircraft Classification"
     data-subtitle="Defense / Navy"></div>
```

then the injected `<script>` block (added by `scripts/sync_ui.sh`) does the rest.

## What the chrome shows

| Region | Content | Source |
|---|---|---|
| Left | HP logo, demo title, subtitle | `data-*` attributes |
| Centre | Tokens (session total) · tok/s | the demo's own `GET /api/metrics` |
| Right | GPU util · power · temp · memory · CPU | `GET :8086/telemetry` |
| Far right | Device label | `GET :8086/sysinfo` |

**Both data sources are optional.** A demo with no LLM simply serves no
`/api/metrics`, and the token panel is hidden rather than showing a fake zero. If
the Demo Center launcher is not running, every hardware meter reads `—`. Nothing
in this chrome ever renders a plausible-looking number it did not receive — that
is the one rule it will not bend.

## Why the widget runs in the browser

The hardware meters are fetched from the host's port 8086, by the browser, not by
the demo's backend. That is deliberate: ballistics, pii, pharmacy and maritime's
AMD stack are `python:3.12-slim` containers with no `nvidia-smi`, no `rocm-smi`
and no `/dev/kfd` — they cannot read GPU telemetry at all, and maritime's NVIDIA
stack is on bridge networking so it cannot even reach the host. The browser can
always reach both, so the awkward cases disappear.

## The logo

`brand/hp_logo.png` is the single canonical asset. Five demos previously shipped
**three different files** under the same name. It is used unmodified — the
artwork has an opaque white background, so the chrome places it on a light plate
rather than alpha-keying a brand-approved file.

## Theming

Tokens are CSS custom properties on `:root` (see `zgx-ui.css`). A demo may
override them to match its own palette; it may not change the layout, the meter
set, or the logo. `scripts/audit_ui_standard.sh` enforces that.
