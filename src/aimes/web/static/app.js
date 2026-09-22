// Live monitor: subscribe to the server-sent event stream and prepend rows.
(function () {
  const tbody = document.querySelector('[data-live-body]');
  const status = document.querySelector('[data-live-status]');
  if (!tbody || !window.EventSource) return;

  const bandClass = { SAFE: 'safe', SUSPICIOUS: 'watch', HIGH_RISK: 'high', CRITICAL: 'critical' };
  const source = new EventSource('/stream');

  source.onopen = () => { if (status) status.textContent = 'Connected. New mail appears here as it is analysed.'; };
  source.onerror = () => { if (status) status.textContent = 'Stream disconnected. Reconnecting.'; };
  source.onmessage = (event) => {
    let row;
    try { row = JSON.parse(event.data); } catch (e) { return; }
    const tr = document.createElement('tr');
    const cls = bandClass[row.band] || 'safe';
    tr.innerHTML =
      '<td class="mono">' + (row.time || '').replace('T', ' ').slice(11, 19) + '</td>' +
      '<td class="trunc">' + esc(row.sender) + '</td>' +
      '<td class="trunc"><a href="/email/' + row.id + '">' + esc(row.subject || '(no subject)') + '</a></td>' +
      '<td>' + esc(row.base_class) + '</td>' +
      '<td class="risk ' + cls + '">' + row.risk + '</td>' +
      '<td class="mono">' + row.confidence + '%</td>' +
      '<td><span class="chip ' + cls + '">' + row.band.replace('_', ' ') + '</span></td>';
    tbody.prepend(tr);
    while (tbody.rows.length > 100) tbody.deleteRow(tbody.rows.length - 1);
  };

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }
})();

// Monitor controls
document.querySelectorAll('[data-monitor]').forEach((button) => {
  button.addEventListener('click', async () => {
    const action = button.getAttribute('data-monitor');
    const out = document.querySelector('[data-monitor-status]');
    button.disabled = true;
    if (out) out.textContent = 'Working...';
    try {
      const response = await fetch('/api/monitor/' + action, { method: 'POST' });
      const body = await response.json();
      if (out) {
        out.textContent = response.ok
          ? (body.analysed !== undefined ? body.analysed + ' new email(s) analysed.' : '')
            + ' Monitor ' + (body.running ? 'running' : 'stopped') + '.'
            + (body.last_error ? ' Last error: ' + body.last_error : '')
          : 'Could not start: ' + (body.error || body.detail || response.status);
      }
    } catch (e) {
      if (out) out.textContent = 'Request failed: ' + e.message;
    } finally {
      button.disabled = false;
    }
  });
});
