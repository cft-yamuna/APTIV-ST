import React, { useEffect, useRef, useState } from "react";

const isIOS =
  typeof navigator !== "undefined" &&
  (/iP(hone|ad|od)/.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1));

function triggerDownload(href, fileName) {
  const link = document.createElement("a");
  link.href = href;
  link.download = fileName || "photo-strip.png";
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export default function DownloadViewer({ imageUrl, fileName }) {
  const [status, setStatus] = useState("loading"); // loading | ready | error
  const [blobUrl, setBlobUrl] = useState("");
  const autoTriggered = useRef(false);

  // Fetch the strip as a blob, retrying a few times so a brief CDN
  // propagation delay after upload doesn't surface as a "not found" error.
  useEffect(() => {
    let cancelled = false;
    let objectUrl = "";

    async function load() {
      for (let attempt = 0; attempt < 12 && !cancelled; attempt += 1) {
        try {
          const response = await fetch(imageUrl, { cache: "no-store" });
          if (response.ok) {
            const blob = await response.blob();
            if (cancelled) return;
            objectUrl = URL.createObjectURL(blob);
            setBlobUrl(objectUrl);
            setStatus("ready");
            return;
          }
        } catch {
          // network hiccup — fall through to retry
        }
        await new Promise((resolve) => setTimeout(resolve, 800));
      }
      if (!cancelled) setStatus("error");
    }

    load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [imageUrl]);

  // Auto-download once ready. iOS Safari blocks programmatic downloads, so we
  // skip it there and rely on the visible button / long-press hint instead.
  useEffect(() => {
    if (status !== "ready" || !blobUrl || autoTriggered.current || isIOS) return;
    autoTriggered.current = true;
    triggerDownload(blobUrl, fileName);
  }, [status, blobUrl, fileName]);

  return (
    <div className="viewer">
      <div className="viewer-card">
        <h1 className="viewer-title">Your photo strip</h1>

        {status === "loading" && (
          <>
            <div className="viewer-spinner" aria-hidden="true" />
            <p className="viewer-text">Preparing your photo…</p>
          </>
        )}

        {status === "error" && (
          <>
            <p className="viewer-text">Could not load the photo. Please try scanning again.</p>
            <button className="viewer-btn" type="button" onClick={() => window.location.reload()}>
              Retry
            </button>
          </>
        )}

        {status === "ready" && (
          <>
            <img className="viewer-image" src={blobUrl} alt="Your photo strip" />
            <button className="viewer-btn" type="button" onClick={() => triggerDownload(blobUrl, fileName)}>
              Download
            </button>
            <p className="viewer-hint">
              {isIOS
                ? "Tap Download, or press and hold the photo to save it to your gallery."
                : "Your download should start automatically. Tap Download if it doesn't."}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
