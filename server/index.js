import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";

import { adbDevices, addDevice } from "./adb.js";
import { addSseClient, startAuto, stopAuto, getSessionStatuses } from "./proc_manager.js";

const app = express();
app.use(cors());
app.use(express.json());

// Serve static UI
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webDir = path.resolve(__dirname, "../web");
app.use("/", express.static(webDir));

// SSE events
app.get("/logs", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  addSseClient(res);
});

// devices + running status
app.get("/devices", async (req, res) => {
  try {
    const devices = await adbDevices();
    const statuses = getSessionStatuses();
    const merged = devices.map((d) => ({
      ...d,
      running: statuses[d.id]?.running || false,
      startedAt: statuses[d.id]?.startedAt || null,
      lastEventAt: statuses[d.id]?.lastEventAt || null,
      statusText: statuses[d.id]?.status || null,
    }));
    res.json(merged);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/device/add", async (req, res) => {
  try {
    const { deviceId } = req.body;
    if (!deviceId) return res.status(200).json({ error: "deviceId required" });

    const r = await addDevice(deviceId);
    res.json(r);
  } catch (e) {
    res.status(200).json({ error: String(e.message || e) });
  }
});

app.post("/auto/start", (req, res) => {
  const { deviceId, intervalMs } = req.body;
  if (!deviceId) return res.status(400).json({ error: "deviceId required" });
  const r = startAuto(deviceId, Number(intervalMs) || 1200);
  res.json(r);
});

app.post("/auto/stop", (req, res) => {
  const { deviceId } = req.body;
  if (!deviceId) return res.status(400).json({ error: "deviceId required" });
  const r = stopAuto(deviceId);
  res.json(r);
});

const PORT = 3000;
app.listen(PORT, () => console.log(`✅ Server running: http://localhost:${PORT}`));
