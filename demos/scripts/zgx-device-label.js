// Fill any [data-zgx-device] element with the machine the Demo Center detected.
//
// Every demo frontend used to hardcode "Powered by HP ZGX Nano" -- and the
// pharmacy one went further with "with NVIDIA Grace Blackwell GB10". At an
// AMD-funded show those lines sell the wrong vendor from the demo's own screen,
// and there is no way to fix them except editing HTML per event.
//
// The attribute's value is a template; "{}" is replaced by the device label.
//   <span data-zgx-device="Powered by {}">Powered by this AI Station</span>
//
// If /sysinfo cannot be reached the element keeps whatever it already says, so
// the fallback text in the markup must name NO vendor. A demo that stays
// generic is fine; a demo that confidently names the wrong silicon is not.
//
// Source of truth: demos/scripts/zgx-device-label.js. The copies inlined into
// each frontend are kept in sync by scripts/sync_device_label.sh, and checked
// by scripts/audit_hardware_strings.sh.
(function () {
  var els = document.querySelectorAll('[data-zgx-device]');
  if (!els.length) return;
  fetch('http://' + location.hostname + ':8086/sysinfo')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (hw) {
      if (!hw || !hw.device_label) return;
      for (var i = 0; i < els.length; i++) {
        var tpl = els[i].getAttribute('data-zgx-device') || '{}';
        els[i].textContent = tpl.replace('{}', hw.device_label);
      }
    })
    .catch(function () { /* offline or launcher not running: keep the neutral text */ });
})();
