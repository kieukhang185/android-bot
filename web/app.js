const devicesEl = document.getElementById("devices");
const logsEl = document.getElementById("logs");
const intervalEl = document.getElementById("interval");
const reloadBtn = document.getElementById("reload");

function fmtTime(ms){ if(!ms) return "-"; return new Date(ms).toLocaleTimeString(); }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g,(m)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m])); }

function logLine(payload){
  const ts = new Date(payload.time || Date.now()).toLocaleTimeString();
  const type = payload.type || "log";
  const msg = payload.msg || payload.message || JSON.stringify(payload);

  let cls = "ok";
  if(type==="error") cls="err";
  if(type==="status") cls="warn";

  const div = document.createElement("div");
  div.className="logline";
  div.innerHTML = `<span class="t">[${ts}]</span> <span class="${cls}">${escapeHtml(payload.deviceId||"")}</span> ${escapeHtml(msg)}`;
  logsEl.appendChild(div);
  logsEl.scrollTop = logsEl.scrollHeight;
}

async function fetchDevices(){
  devicesEl.textContent="Loading...";
  const res = await fetch(`/devices`);
  const devices = await res.json();
  devicesEl.innerHTML="";
  if(!devices.length){ devicesEl.textContent="No devices found. (Check adb devices / adb connect)"; return; }

  for(const d of devices){
    const row = document.createElement("div");
    row.className="device";

    const pill = d.running
      ? `<span class="pill good">● running</span>`
      : `<span class="pill bad">● stopped</span>`;

    const statusText = d.statusText ? ` | ${escapeHtml(d.statusText)}` : "";

    row.innerHTML = `
      <div class="meta">
        <div class="id">${escapeHtml(d.id)}</div>
        <div class="sub">adb: ${escapeHtml(d.status||"")} | last: ${escapeHtml(fmtTime(d.lastEventAt))}${statusText}</div>
        <div>${pill}</div>
      </div>
      <div class="buttons">
        <button data-action="start" data-id="${escapeHtml(d.id)}">Start</button>
        <button data-action="stop" data-id="${escapeHtml(d.id)}">Stop</button>
      </div>
    `;

    row.addEventListener("click", async (e)=>{
      const btn = e.target.closest("button");
      if(!btn) return;
      const action = btn.dataset.action;
      const deviceId = btn.dataset.id;

      if(action==="start"){
        const intervalMs = Number(intervalEl.value) || 1200;
        await fetch(`/auto/start`, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ deviceId, intervalMs })});
        await fetchDevices();
      }
      if(action==="stop"){
        await fetch(`/auto/stop`, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ deviceId })});
        await fetchDevices();
      }
    });

    devicesEl.appendChild(row);
  }
}

reloadBtn.addEventListener("click", fetchDevices);

const es = new EventSource(`/logs`);
es.onmessage = (e)=>{ try{ logLine(JSON.parse(e.data)); }catch{} };

// refresh device statuses periodically
setInterval(fetchDevices, 4000);
fetchDevices();
