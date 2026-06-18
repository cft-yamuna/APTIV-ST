import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import DownloadViewer from "./DownloadViewer.jsx";
import "./App.css";

const params = new URLSearchParams(window.location.search);
const imageUrl = params.get("img");
const fileName = params.get("name") || "photo-strip.png";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    {imageUrl ? <DownloadViewer imageUrl={imageUrl} fileName={fileName} /> : <App />}
  </StrictMode>
);
