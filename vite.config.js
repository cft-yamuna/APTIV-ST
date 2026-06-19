import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const OUTPUT_DIR = path.resolve(process.cwd(), "output_images");
const DATA_DIR = path.resolve(process.cwd(), "data");
const REGISTRATIONS_FILE = path.join(DATA_DIR, "registrations.json");
const MAX_IMAGE_BYTES = 25 * 1024 * 1024;

function getTimestampedFilename() {
  return `photo-strip-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
}

function getLanIp() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const net of interfaces[name] || []) {
      if (net.family === "IPv4" && !net.internal) {
        return net.address;
      }
    }
  }
  return "127.0.0.1";
}

async function readJsonBody(req) {
  let body = "";

  for await (const chunk of req) {
    body += chunk;
    if (body.length > MAX_IMAGE_BYTES * 2) {
      throw new Error("Image payload is too large");
    }
  }

  return JSON.parse(body || "{}");
}

async function appendRegistration(entry) {
  await fs.mkdir(DATA_DIR, { recursive: true });

  let registrations = [];
  try {
    registrations = JSON.parse(await fs.readFile(REGISTRATIONS_FILE, "utf8"));
    if (!Array.isArray(registrations)) registrations = [];
  } catch {
    // File doesn't exist yet (or is corrupt) — start a fresh list.
    registrations = [];
  }

  registrations.push(entry);
  await fs.writeFile(REGISTRATIONS_FILE, JSON.stringify(registrations, null, 2));
}

function outputImageSaver() {
  return {
    name: "output-image-saver",
    configureServer(server) {
      // Save visitor contact details (name, email, mobile) to data/registrations.json.
      server.middlewares.use("/api/save-registration", async (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method not allowed");
          return;
        }

        try {
          const body = await readJsonBody(req);
          const name = String(body.name || "").trim();
          const email = String(body.email || "").trim();
          const mobile = String(body.mobile || "").trim();

          if (!name || !email || !mobile) {
            res.statusCode = 400;
            res.end("Missing name, email, or mobile");
            return;
          }

          if (!/^\d{10}$/.test(mobile)) {
            res.statusCode = 400;
            res.end("Mobile number must be exactly 10 digits");
            return;
          }

          await appendRegistration({ name, email, mobile, savedAt: new Date().toISOString() });

          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ ok: true }));
        } catch (error) {
          console.error("Could not save registration.", error);
          res.statusCode = 500;
          res.end("Could not save registration");
        }
      });

      // Save the generated strip and return a LAN-reachable download URL.
      server.middlewares.use("/api/save-output-image", async (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method not allowed");
          return;
        }

        try {
          const { imageData } = await readJsonBody(req);
          const match = /^data:image\/png;base64,(.+)$/.exec(imageData || "");

          if (!match) {
            res.statusCode = 400;
            res.end("Expected a PNG data URL");
            return;
          }

          const imageBuffer = Buffer.from(match[1], "base64");
          if (imageBuffer.byteLength > MAX_IMAGE_BYTES) {
            res.statusCode = 413;
            res.end("Image is too large");
            return;
          }

          await fs.mkdir(OUTPUT_DIR, { recursive: true });
          const filename = getTimestampedFilename();
          await fs.writeFile(path.join(OUTPUT_DIR, filename), imageBuffer);

          const host = req.headers.host || "";
          const port = host.includes(":") ? host.split(":")[1] : "5173";
          const downloadUrl = `http://${getLanIp()}:${port}/output/${filename}`;

          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ filename, url: downloadUrl }));
        } catch (error) {
          console.error("Could not save output image.", error);
          res.statusCode = 500;
          res.end("Could not save output image");
        }
      });

      // Serve saved strips for phone download (scanned via QR code).
      server.middlewares.use("/output/", async (req, res) => {
        try {
          const requested = decodeURIComponent((req.url || "").split("?")[0]).replace(/^\/+/, "");
          const filename = path.basename(requested);
          const filePath = path.join(OUTPUT_DIR, filename);

          const fileBuffer = await fs.readFile(filePath);
          res.setHeader("Content-Type", "image/png");
          res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
          res.setHeader("Cache-Control", "no-store");
          res.end(fileBuffer);
        } catch {
          res.statusCode = 404;
          res.end("Image not found");
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), outputImageSaver()],
});
