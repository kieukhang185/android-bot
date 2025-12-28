const devicesEl = document.getElementById("devices");
const logsEl = document.getElementById("logs");

const startAllBtn = document.getElementById("startAll");
const stopAllBtn = document.getElementById("stopAll");
const toggleLogsBtn = document.getElementById("toggleLogs");
const clearLogsBtn = document.getElementById("clearLogs");
const autoScrollEl = document.getElementById("autoScroll");
const deviceAddEl = document.getElementById("deviceAdd");
const addDeviceBtn = document.getElementById("addDeviceBtn");
const deviceSearchEl = document.getElementById("deviceSearch");
const logsCard = document.getElementById("logsCard");
const logDeviceFilterEl = document.getElementById("logDeviceFilter");

let lastDevices = [];

/* ---------------- utils ---------------- */
function fmtTime(ms) {
  if (!ms) return new Date().toLocaleTimeString();
  return new Date(ms).toLocaleTimeString();
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[m]));
}
async function postJSON(url, body) {
  await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
}

/* ---------------- logs (filter by device) ---------------- */
function selectedLogDevice() {
  return (logDeviceFilterEl?.value || "").trim();
}

// payload format like before: { time, type, deviceId, msg/message }
function logLine(payload) {
  const deviceId = payload.deviceId || payload.device || "";
  const filter = selectedLogDevice();

  // filter: only show matching device if chosen
  if (filter && deviceId !== filter) return;

  const ts = fmtTime(payload.time || Date.now());

  // message only (as before)
  const msg = payload.msg ?? payload.message ?? "";
  const type = payload.type

  const div = document.createElement("div");
  div.className = `logline log-${type}`;
  div.innerHTML = `<span class="t">[${escapeHtml(ts)}]</span><span class="dev">${escapeHtml(deviceId)}</span>${escapeHtml(msg)}`;

  logsEl.appendChild(div);

  if (autoScrollEl?.checked) {
    logsEl.scrollTop = logsEl.scrollHeight;
  }
}

clearLogsBtn?.addEventListener("click", () => {
  logsEl.innerHTML = "";
});

toggleLogsBtn?.addEventListener("click", () => {
  logsCard.classList.toggle("logs-hidden");
  const hidden = logsCard.classList.contains("logs-hidden");
  toggleLogsBtn.textContent = hidden ? "Show" : "Hide";
});

// When changing device filter: clear current logs (keeps UI simple/clean)
logDeviceFilterEl?.addEventListener("change", () => {
  logsEl.innerHTML = "";
});

/* ---------------- devices ---------------- */
async function addDevice(deviceId) {
  await postJSON("/device/add", { deviceId });
}

function applyDeviceFilterAndRender() {
  const q = (deviceSearchEl?.value || "").toLowerCase().trim();
  const list = !q
    ? lastDevices
    : lastDevices.filter(d =>
        (d.id || "").toLowerCase().includes(q) ||
        (d.name || "").toLowerCase().includes(q)
      );

  renderDevices(list);
}

function renderDevices(devices) {
  devicesEl.innerHTML = "";

  for (const d of devices) {
    const row = document.createElement("div");
    row.className = "device";

    const left = document.createElement("div");
    left.className = "device-left";
    left.innerHTML = `
      <div class="device-id">${escapeHtml(d.id)}</div>
      <div class="device-sub">${escapeHtml(d.statusText || d.status || "")}</div>
    `;

    const right = document.createElement("div");
    right.className = "device-right";

    const status = document.createElement("span");
    status.textContent = d.running ? "🟢" : "⚪";
    status.title = d.running ? "Running" : "Stopped";

    const btn = document.createElement("button");
    btn.textContent = d.running ? "Stop" : "Start";

    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      btn.disabled = true;
      try {
        if (d.running) {
          await postJSON("/auto/stop", { deviceId: d.id });
        } else {
          await postJSON("/auto/start", { deviceId: d.id });
        }
        await fetchDevices();
      } finally {
        btn.disabled = false;
      }
    });

    // quick filter logs by clicking device row (optional but handy)
    row.addEventListener("click", () => {
      if (logDeviceFilterEl) {
        logDeviceFilterEl.value = d.id;
        logsEl.innerHTML = "";
      }
    });

    right.appendChild(status);
    right.appendChild(btn);

    row.appendChild(left);
    row.appendChild(right);
    devicesEl.appendChild(row);
  }
}

function refreshLogDeviceDropdown(devices) {
  if (!logDeviceFilterEl) return;

  const current = selectedLogDevice();
  const ids = devices.map(d => d.id);

  // Rebuild options
  logDeviceFilterEl.innerHTML = `<option value="">All devices</option>` +
    ids.map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");

  // Restore selected if still present
  if (current && ids.includes(current)) logDeviceFilterEl.value = current;
}

async function fetchDevices() {
  const res = await fetch("/devices");
  const devices = await res.json();
  lastDevices = devices;

  refreshLogDeviceDropdown(devices);
  applyDeviceFilterAndRender();
}

deviceSearchEl?.addEventListener("input", applyDeviceFilterAndRender);

/* ---------------- Start/Stop all ---------------- */
startAllBtn?.addEventListener("click", async () => {
  for (const d of lastDevices) {
    await postJSON("/auto/start", { deviceId: d.id });
  }
  fetchDevices();
});

stopAllBtn?.addEventListener("click", async () => {
  for (const d of lastDevices) {
    await postJSON("/auto/stop", { deviceId: d.id });
  }
  fetchDevices();
});

/* ---------------- SSE logs ---------------- */
const es = new EventSource("/logs");
es.onmessage = (e) => {
  try {
    const payload = JSON.parse(e.data); // same as before :contentReference[oaicite:1]{index=1}
    logLine(payload);
  } catch {
    // if server ever sends plain text
    logLine({ time: Date.now(), deviceId: "", msg: String(e.data) });
  }
};

/* ------------- adb add device ------------- */
async function onAddDevice() {
  const id = (deviceAddEl.value || "").trim();
  if (!id) return;

  addDeviceBtn.disabled = true;
  try {
    const r = await addDevice(id);
    if (r?.error) {
      alert(r.error);
      return;
    }
    await fetchDevices();
  } catch (e) {
    console.error(e);
    alert("Add device failed. Check endpoint /devices/add and payload {deviceId}.");
  } finally {
    addDeviceBtn.disabled = false;
  }
}

addDeviceBtn?.addEventListener("click", onAddDevice);

deviceAddEl?.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") onAddDevice();
});

/* ---------------- init ---------------- */
fetchDevices();
setInterval(fetchDevices, 4000);
