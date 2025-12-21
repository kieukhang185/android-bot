import { execFile } from "child_process";
import fs from "fs";

export function adbDevices() {
  return new Promise((resolve, reject) => {
    execFile("adb", ["devices"], (err, stdout, stderr) => {
      if (err) return reject(new Error(stderr || err.message));
      const lines = stdout.trim().split("\n").slice(1);
      const devices = lines
        .map((l) => l.trim().split(/\s+/))
        .filter((parts) => parts[0])
        .map(([id, status]) => ({ id, status }));
      resolve(devices);
    });
  });
}

// Optional: take screenshot from Node (not required if python does it)
export function adbScreencap(deviceId, outPath) {
  return new Promise((resolve, reject) => {
    const out = fs.createWriteStream(outPath);
    const adb = execFile(
      "adb",
      ["-s", deviceId, "exec-out", "screencap", "-p"],
      { encoding: "buffer" }
    );
    adb.stdout.pipe(out);
    adb.on("error", reject);
    adb.on("close", (code) => {
      if (code === 0) resolve(outPath);
      else reject(new Error(`adb screencap failed (code=${code})`));
    });
  });
}
