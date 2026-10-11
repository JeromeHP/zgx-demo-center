# Debug Tools

Utilities for diagnosing Demo Center issues. Not part of the production launcher.

## iframe-test.html

Test harness for verifying whether a demo can be loaded inside an iframe served from the launcher's origin (port 8085). Useful when adding a new demo or diagnosing iframe-related issues (X-Frame-Options, CSP frame-ancestors, microphone permissions, etc.).

Usage: ensure the demo is running, then open http://<nano-ip>:8085/tools/iframe-test.html, select the demo from the dropdown, observe behavior. Detailed errors appear in the browser DevTools console.
