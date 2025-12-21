import { spawn } from "child_process";

const sessions = new Map(); // deviceId -> { proc, startedAt, lastEventAt, status }

const sseClients = new Set();

export function addSseClient(res) {
  sseClients.add(res);
  res.write(`data: ${JSON.stringify({ type: "system", msg: "connected", time: Date.now() })}\n\n`);
  res.on("close", () => sseClients.delete(res));
}

export function emitEvent(deviceId, payload) {
  const evt = { time: Date.now(), deviceId, ...payload };
  const line = `data: ${JSON.stringify(evt)}\n\n`;
  for (const c of sseClients) c.write(line);

  const s = sessions.get(deviceId);
  if (s) {
    s.lastEventAt = evt.time;
    // keep a compact status
    if (payload.type === "status") s.status = payload.msg;
    sessions.set(deviceId, s);
  }
}

export function getSessionStatuses() {
  const out = {};
  for (const [deviceId, s] of sessions.entries()) {
    out[deviceId] = {
      running: !!s.proc && !s.proc.killed,
      startedAt: s.startedAt || null,
      lastEventAt: s.lastEventAt || null,
      status: s.status || null
    };
  }
  return out;
}

function safeArgs(deviceId, intervalMs) {
  return [
    "auto_puzzle.py",
    deviceId,
    "--interval-ms",
    String(intervalMs),
    "--workdir",
    "templates",
  ];
}

// Start a long-running python process per device.
// It prints JSONL (one JSON object per line). We forward those to SSE.
export function startAuto(deviceId, intervalMs = 1200) {
  if (sessions.get(deviceId)?.proc) return { ok: true, already: true };

  const args = safeArgs(deviceId, intervalMs);

  const proc = spawn("python3", args, {
    cwd: process.cwd(), // server/
    env: process.env,
  });

  const session = { proc, startedAt: Date.now(), lastEventAt: null, status: "starting" };
  sessions.set(deviceId, session);

  emitEvent(deviceId, { type: "status", msg: `AUTO started (python pid=${proc.pid})` });

  let buf = "";
  proc.stdout.on("data", (d) => {
    buf += d.toString();
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;

      // Expect JSON per line; if not, still forward as log
      try {
        const obj = JSON.parse(line);
        emitEvent(deviceId, { type: "py", ...obj });
      } catch {
        emitEvent(deviceId, { type: "log", msg: line });
      }
    }
  });

  proc.stderr.on("data", (d) => {
    emitEvent(deviceId, { type: "error", msg: d.toString().trim() });
  });

  proc.on("close", (code, signal) => {
    emitEvent(deviceId, { type: "status", msg: `AUTO stopped (code=${code}, signal=${signal || ""})` });
    sessions.delete(deviceId);
  });

  proc.on("error", (e) => {
    emitEvent(deviceId, { type: "error", msg: `spawn error: ${e.message}` });
    sessions.delete(deviceId);
  });

  return { ok: true };
}

export function stopAuto(deviceId) {
  const s = sessions.get(deviceId);
  if (!s?.proc) return { ok: true, already: true };

  try {
    s.proc.kill("SIGTERM");
  } catch (e) {
    return { ok: false, error: e.message };
  }
  return { ok: true };
}
