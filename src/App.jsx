import React, { useEffect, useMemo, useRef, useState } from "react";
import * as bodySegmentation from "@tensorflow-models/body-segmentation";
import "@tensorflow/tfjs-backend-webgl";

const STAGE_WIDTH = 1920;
const STAGE_HEIGHT = 1080;
const POSE_WIDTH = 468;
const POSE_HEIGHT = 702;
const SHEET_WIDTH = 1200;
const SHEET_HEIGHT = 1800;
const FRAME_SRC = "/leadframe.png";
const PERSON_SCALE = 1;
const PERSON_BASELINE_DROP = 0.06;
const FRAME_SHADOW_CROP = 64;
const STRIP_COUNT = 3;
const BG_REMOVAL_URL = "http://127.0.0.1:8765/remove-bg";
const BACKGROUND_OPTIONS = [
  { id: "bg-1", name: "Mountains", previewSrc: "/select1.png", src: "/bg1.png" },
  { id: "bg-2", name: "Beach", previewSrc: "/select2.png", src: "/bg2.png" },
  { id: "bg-3", name: "Desert", previewSrc: "/select3.png", src: "/bg3.png" },
  { id: "bg-4", name: "Background 4", previewSrc: "/select4.png", src: "/bg4.png" },
];
const EMOJI_COUNT = 18;
const AVAILABLE_EMOJIS = Array.from({ length: EMOJI_COUNT }, (_, index) => index + 1).map((number) => ({
  id: String(number),
  src: `/emojis/${number}.png`,
  name: `Emoji ${number}`,
}));
const FIXED_EMOJI_POSITIONS = [
  { x: 0.04, y: 0.5 },
  { x: 0.96, y: 0.56 },
  { x: 0.08, y: 0.88 },
];

function setCanvasCover(ctx, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight) {
  setCanvasCoverWithFocus(ctx, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight, 0.5, 0.5);
}

function getCoverCrop(sourceWidth, sourceHeight, targetRatio, focusX = 0.5, focusY = 0.5) {
  const sourceRatio = sourceWidth / sourceHeight;
  let cropWidth = sourceWidth;
  let cropHeight = sourceHeight;
  let cropX = 0;
  let cropY = 0;

  if (sourceRatio > targetRatio) {
    cropWidth = sourceHeight * targetRatio;
    cropX = (sourceWidth - cropWidth) * focusX;
  } else {
    cropHeight = sourceWidth / targetRatio;
    cropY = (sourceHeight - cropHeight) * focusY;
  }

  return {
    x: Math.max(0, Math.min(sourceWidth - cropWidth, cropX)),
    y: Math.max(0, Math.min(sourceHeight - cropHeight, cropY)),
    width: cropWidth,
    height: cropHeight,
  };
}

function setCanvasCoverWithFocus(ctx, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight, focusX = 0.5, focusY = 0.5) {
  const crop = getCoverCrop(sourceWidth, sourceHeight, targetWidth / targetHeight, focusX, focusY);
  ctx.drawImage(source, crop.x, crop.y, crop.width, crop.height, targetX, targetY, targetWidth, targetHeight);
}

// Canvas throws away detail when a single drawImage reduces by more than half,
// which is exactly what a 1920+ px camera frame into a ~520 px slot does. Halving
// repeatedly first keeps the photo sharp instead of soft and aliased.
function drawCoverSharp(ctx, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight, focusX = 0.5, focusY = 0.5) {
  const crop = getCoverCrop(sourceWidth, sourceHeight, targetWidth / targetHeight, focusX, focusY);
  const destWidth = Math.ceil(targetWidth);
  const destHeight = Math.ceil(targetHeight);

  let stepSource = source;
  let stepX = crop.x;
  let stepY = crop.y;
  let stepWidth = crop.width;
  let stepHeight = crop.height;

  while (stepWidth > destWidth * 2 && stepHeight > destHeight * 2) {
    const nextWidth = Math.max(destWidth, Math.round(stepWidth / 2));
    const nextHeight = Math.max(destHeight, Math.round(stepHeight / 2));
    const stepCanvas = document.createElement("canvas");
    stepCanvas.width = nextWidth;
    stepCanvas.height = nextHeight;

    const stepCtx = stepCanvas.getContext("2d");
    stepCtx.imageSmoothingEnabled = true;
    stepCtx.imageSmoothingQuality = "high";
    stepCtx.drawImage(stepSource, stepX, stepY, stepWidth, stepHeight, 0, 0, nextWidth, nextHeight);

    stepSource = stepCanvas;
    stepX = 0;
    stepY = 0;
    stepWidth = nextWidth;
    stepHeight = nextHeight;
  }

  ctx.drawImage(stepSource, stepX, stepY, stepWidth, stepHeight, targetX, targetY, targetWidth, targetHeight);
}

function setCanvasContain(ctx, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight) {
  const sourceRatio = sourceWidth / sourceHeight;
  const targetRatio = targetWidth / targetHeight;
  let drawWidth = targetWidth;
  let drawHeight = targetHeight;

  if (sourceRatio > targetRatio) {
    drawHeight = targetWidth / sourceRatio;
  } else {
    drawWidth = targetHeight * sourceRatio;
  }

  ctx.drawImage(
    source,
    targetX + (targetWidth - drawWidth) / 2,
    targetY + (targetHeight - drawHeight) / 2,
    drawWidth,
    drawHeight
  );
}

function drawFrameOverlay(ctx, frameImage) {
  // leadframe.png already contains both strip columns and is transparent only where
  // the photos go, so painting it over the photos restores every printed detail —
  // including the red accent bars that overlap the slot corners.
  ctx.drawImage(frameImage, 0, 0, SHEET_WIDTH, SHEET_HEIGHT);
}

// Photo slots are the transparent cut-outs in leadframe.png, measured in its own
// 1153 x 1729 pixel space and scaled to the sheet, so SHEET_WIDTH/HEIGHT can change
// without re-measuring anything.
const FRAME_ART_WIDTH = 1153;
const FRAME_ART_HEIGHT = 1729;
const FRAME_SLOT_COLUMNS = [28, 604];
const FRAME_SLOT_ROWS = [427, 778, 1129];
const FRAME_SLOT_WIDTH = 520;
const FRAME_SLOT_HEIGHT = 323;
// The photo is painted a hair past the cut-out on every side so no white seam can
// show through the anti-aliased edge; the frame overlay hides the overdraw.
const SLOT_BLEED = 2;

function getSheetBoxes() {
  const scaleX = SHEET_WIDTH / FRAME_ART_WIDTH;
  const scaleY = SHEET_HEIGHT / FRAME_ART_HEIGHT;
  const boxes = [];

  FRAME_SLOT_COLUMNS.forEach((slotLeft) => {
    FRAME_SLOT_ROWS.forEach((slotTop, row) => {
      boxes.push({
        poseIndex: row,
        x: slotLeft * scaleX,
        y: slotTop * scaleY,
        width: FRAME_SLOT_WIDTH * scaleX,
        height: FRAME_SLOT_HEIGHT * scaleY,
      });
    });
  });

  return boxes;
}

// The slot grown by the bleed, i.e. the rectangle the photo actually fills.
function getBleedBox(box) {
  const bleedX = SLOT_BLEED * (SHEET_WIDTH / FRAME_ART_WIDTH);
  const bleedY = SLOT_BLEED * (SHEET_HEIGHT / FRAME_ART_HEIGHT);

  return {
    x: box.x - bleedX,
    y: box.y - bleedY,
    width: box.width + bleedX * 2,
    height: box.height + bleedY * 2,
  };
}

function drawEmojiAt(ctx, emojiImage, x, y, size) {
  if (!emojiImage) return;

  const imageWidth = emojiImage.naturalWidth || size;
  const imageHeight = emojiImage.naturalHeight || size;
  const ratio = imageWidth / imageHeight;
  const drawWidth = ratio >= 1 ? size : size * ratio;
  const drawHeight = ratio >= 1 ? size / ratio : size;

  ctx.save();
  ctx.shadowColor = "rgba(0, 0, 0, 0.2)";
  ctx.shadowBlur = 6;
  ctx.shadowOffsetY = 3;
  ctx.drawImage(emojiImage, x - drawWidth / 2, y - drawHeight / 2, drawWidth, drawHeight);
  ctx.restore();
}

function canvasToBlob(canvas, type = "image/png", quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("Could not encode camera frame."));
      }
    }, type, quality);
  });
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not encode photo strip."));
    reader.readAsDataURL(blob);
  });
}

// Same lossless PNG data URL as canvas.toDataURL, but encoded off the main
// thread so the UI stays responsive while the sheet is written. Falls back to
// the synchronous call if toBlob/FileReader fails, so the flow never breaks.
async function encodeCanvasToDataUrl(canvas) {
  try {
    return await blobToDataUrl(await canvasToBlob(canvas, "image/png"));
  } catch {
    return canvas.toDataURL("image/png");
  }
}

function loadImageFromBlob(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();

    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not load processed image."));
    };
    image.src = url;
  });
}

function drawBlackBackgroundReplacement(ctx, keyedCanvas, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight) {
  keyedCanvas.width = targetWidth;
  keyedCanvas.height = targetHeight;

  const keyedCtx = keyedCanvas.getContext("2d", { willReadFrequently: true });
  keyedCtx.clearRect(0, 0, targetWidth, targetHeight);
  setCanvasContain(keyedCtx, source, sourceWidth, sourceHeight, 0, 0, targetWidth, targetHeight);

  const frame = keyedCtx.getImageData(0, 0, targetWidth, targetHeight);
  const pixels = frame.data;
  const subjectBounds = { left: targetWidth, right: 0, top: targetHeight, bottom: 0 };
  const subjectScanHeight = Math.round(targetHeight * 0.82);

  for (let index = 0; index < pixels.length; index += 4) {
    const red = pixels[index];
    const green = pixels[index + 1];
    const blue = pixels[index + 2];
    const brightness = (red + green + blue) / 3;
    const colorSpread = Math.max(red, green, blue) - Math.min(red, green, blue);

    if (brightness < 42 && colorSpread < 28) {
      pixels[index + 3] = 0;
    } else if (brightness < 72 && colorSpread < 34) {
      pixels[index + 3] = Math.round(((brightness - 42) / 30) * 255);
    }

    if (pixels[index + 3] > 32) {
      const pixel = index / 4;
      const x = pixel % targetWidth;
      const y = Math.floor(pixel / targetWidth);

      if (y < subjectScanHeight) {
        subjectBounds.left = Math.min(subjectBounds.left, x);
        subjectBounds.right = Math.max(subjectBounds.right, x);
        subjectBounds.top = Math.min(subjectBounds.top, y);
        subjectBounds.bottom = Math.max(subjectBounds.bottom, y);
      }
    }
  }

  keyedCtx.putImageData(frame, 0, 0);
  drawPositionedPerson(ctx, keyedCanvas, subjectBounds, targetX, targetY, targetWidth, targetHeight);
}

function drawPositionedPerson(ctx, keyedCanvas, subjectBounds, targetX, targetY, targetWidth, targetHeight) {
  let offsetX = 0;
  let offsetY = targetHeight * PERSON_BASELINE_DROP;

  if (subjectBounds.left <= subjectBounds.right) {
    const subjectCenterX = (subjectBounds.left + subjectBounds.right) / 2;

    offsetX = targetWidth / 2 - subjectCenterX;
    offsetY += targetHeight - subjectBounds.bottom;
    offsetX = Math.max(-targetWidth * 0.18, Math.min(targetWidth * 0.18, offsetX));
    offsetY = Math.max(-targetHeight * 0.18, Math.min(targetHeight * 0.24, offsetY));
  }

  const scaledWidth = targetWidth * PERSON_SCALE;
  const scaledHeight = targetHeight * PERSON_SCALE;
  ctx.drawImage(
    keyedCanvas,
    targetX + offsetX - (scaledWidth - targetWidth) / 2,
    targetY + offsetY - (scaledHeight - targetHeight) / 2,
    scaledWidth,
    scaledHeight
  );
}

function drawProcessedPerson(ctx, keyedCanvas, image, targetX, targetY, targetWidth, targetHeight) {
  keyedCanvas.width = targetWidth;
  keyedCanvas.height = targetHeight;

  const keyedCtx = keyedCanvas.getContext("2d", { willReadFrequently: true });
  keyedCtx.clearRect(0, 0, targetWidth, targetHeight);
  keyedCtx.drawImage(image, 0, 0, targetWidth, targetHeight);

  const frame = keyedCtx.getImageData(0, 0, targetWidth, targetHeight);
  const pixels = frame.data;
  const subjectBounds = { left: targetWidth, right: 0, top: targetHeight, bottom: 0 };

  for (let index = 0; index < pixels.length; index += 4) {
    if (pixels[index + 3] > 32) {
      const pixel = index / 4;
      const x = pixel % targetWidth;
      const y = Math.floor(pixel / targetWidth);
      subjectBounds.left = Math.min(subjectBounds.left, x);
      subjectBounds.right = Math.max(subjectBounds.right, x);
      subjectBounds.top = Math.min(subjectBounds.top, y);
      subjectBounds.bottom = Math.max(subjectBounds.bottom, y);
    }
  }

  drawPositionedPerson(ctx, keyedCanvas, subjectBounds, targetX, targetY, targetWidth, targetHeight);
}

async function drawSegmentedPerson(ctx, segmenter, keyedCanvas, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight) {
  keyedCanvas.width = targetWidth;
  keyedCanvas.height = targetHeight;

  const keyedCtx = keyedCanvas.getContext("2d", { willReadFrequently: true });
  keyedCtx.clearRect(0, 0, targetWidth, targetHeight);
  setCanvasContain(keyedCtx, source, sourceWidth, sourceHeight, 0, 0, targetWidth, targetHeight);

  const segmentations = await segmenter.segmentPeople(keyedCanvas, {
    flipHorizontal: false,
    multiSegmentation: false,
    segmentBodyParts: false,
  });

  if (!segmentations.length) {
    drawBlackBackgroundReplacement(ctx, keyedCanvas, source, sourceWidth, sourceHeight, targetX, targetY, targetWidth, targetHeight);
    return;
  }

  const mask = await bodySegmentation.toBinaryMask(
    segmentations,
    { r: 255, g: 255, b: 255, a: 255 },
    { r: 0, g: 0, b: 0, a: 0 },
    false,
    0.45
  );
  const frame = keyedCtx.getImageData(0, 0, targetWidth, targetHeight);
  const framePixels = frame.data;
  const maskPixels = mask.data;
  const subjectBounds = { left: targetWidth, right: 0, top: targetHeight, bottom: 0 };

  for (let index = 0; index < framePixels.length; index += 4) {
    const alpha = maskPixels[index + 3];
    framePixels[index + 3] = alpha;

    if (alpha > 32) {
      const pixel = index / 4;
      const x = pixel % targetWidth;
      const y = Math.floor(pixel / targetWidth);
      subjectBounds.left = Math.min(subjectBounds.left, x);
      subjectBounds.right = Math.max(subjectBounds.right, x);
      subjectBounds.top = Math.min(subjectBounds.top, y);
      subjectBounds.bottom = Math.max(subjectBounds.bottom, y);
    }
  }

  keyedCtx.putImageData(frame, 0, 0);
  drawPositionedPerson(ctx, keyedCanvas, subjectBounds, targetX, targetY, targetWidth, targetHeight);
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const keyedCanvasRef = useRef(document.createElement("canvas"));
  const backgroundImageRef = useRef(null);
  const frameImageRef = useRef(null);
  const emojiImagesRef = useRef({});
  const segmenterRef = useRef(null);
  const segmenterLoadingRef = useRef(null);
  const imageBoxesRef = useRef([]);

  const [step, setStep] = useState("start");
  const [stageScale, setStageScale] = useState(1);
  const [status, setStatus] = useState({ message: "Select a background", type: "" });
  const [cameras, setCameras] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [selectedBackground, setSelectedBackground] = useState(BACKGROUND_OPTIONS[0]);
  const [cameraReady, setCameraReady] = useState(false);
  const [isCapturingSequence, setIsCapturingSequence] = useState(false);
  const [isProcessingCaptures, setIsProcessingCaptures] = useState(false);
  const [countdown, setCountdown] = useState(null);
  const [poses, setPoses] = useState([]);
  const [activeEmoji, setActiveEmoji] = useState(AVAILABLE_EMOJIS[0]);
  const [emojiPlacements, setEmojiPlacements] = useState([]);
  const [sheetUrl, setSheetUrl] = useState("");

  const captureLabel = useMemo(() => {
    if (isProcessingCaptures) return "Preparing...";
    if (isCapturingSequence) return "Capturing...";
    return "Capture";
  }, [isCapturingSequence, isProcessingCaptures]);
  const selectedBackgroundIndex = BACKGROUND_OPTIONS.findIndex((option) => option.id === selectedBackground.id);
  const carouselBackgrounds = BACKGROUND_OPTIONS.map((_, index) => BACKGROUND_OPTIONS[(selectedBackgroundIndex + index) % BACKGROUND_OPTIONS.length]);

  useEffect(() => {
    const frame = new Image();
    frame.src = FRAME_SRC;
    frameImageRef.current = frame;

    emojiImagesRef.current = AVAILABLE_EMOJIS.reduce((images, emoji) => {
      const image = new Image();
      image.src = emoji.src;
      images[emoji.id] = image;
      return images;
    }, {});

    listCameras();

    return () => {
      stopStream();
    };
  }, []);

  useEffect(() => {
    const image = new Image();
    image.src = selectedBackground.src;
    backgroundImageRef.current = image;
  }, [selectedBackground]);

  useEffect(() => {
    if (step === "capture") {
      startCamera();
      return;
    }

    stopStream();
  }, [step]);

  useEffect(() => {
    function updateStageScale() {
      const scale = Math.min(window.innerWidth / STAGE_WIDTH, window.innerHeight / STAGE_HEIGHT);
      setStageScale(scale > 0 ? scale : 1);
    }

    updateStageScale();
    window.addEventListener("resize", updateStageScale);
    return () => window.removeEventListener("resize", updateStageScale);
  }, []);

  async function getSegmenter() {
    if (segmenterRef.current) return segmenterRef.current;
    if (segmenterLoadingRef.current) return segmenterLoadingRef.current;

    updateStatus("Loading person mask");
    segmenterLoadingRef.current = bodySegmentation
      .createSegmenter(bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation, {
        runtime: "mediapipe",
        modelType: "general",
        // Served from public/ so the kiosk works fully offline (no CDN).
        solutionPath: "/mediapipe/selfie_segmentation",
      })
      .then((segmenter) => {
        segmenterRef.current = segmenter;
        return segmenter;
      })
      .catch((error) => {
        console.warn("MediaPipe segmentation failed, using black key fallback.", error);
        segmenterLoadingRef.current = null;
        updateStatus("Person mask unavailable, using fallback", "error");
        return null;
      });

    return segmenterLoadingRef.current;
  }

  function updateStatus(message, type = "") {
    setStatus({ message, type });
  }

  function stopStream() {
    if (!streamRef.current) return;
    streamRef.current.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraReady(false);
  }

  async function listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((device) => device.kind === "videoinput");
    setCameras(videoInputs);
  }

  async function startCamera(deviceId = selectedDevice) {
    if (!navigator.mediaDevices?.getUserMedia) {
      updateStatus("Camera access is not supported in this browser", "error");
      return;
    }

    stopStream();
    updateStatus("Starting camera");

    // Ask for the sharpest stream the webcam has; `ideal` quietly settles for the
    // camera's real maximum, and every extra pixel here is detail the printed slot
    // gets to keep.
    const videoConstraints = { width: { ideal: 3840 }, height: { ideal: 2160 } };
    const constraints = {
      video: deviceId ? { deviceId: { exact: deviceId }, ...videoConstraints } : videoConstraints,
      audio: false,
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      await listCameras();
      // The webcam can silently fall back below the requested 1920x1080, which caps
      // how sharp the printed strip can ever be — log what we actually got.
      const track = stream.getVideoTracks()[0];
      console.info("Camera stream:", track?.label, track?.getSettings?.());
      setCameraReady(true);
      updateStatus(`Capture ${STRIP_COUNT} photos`, "ready");
    } catch (error) {
      setCameraReady(false);
      updateStatus(error.message || "Could not start camera", "error");
    }
  }

  async function handleDeviceChange(event) {
    const deviceId = event.target.value;
    setSelectedDevice(deviceId);
    await startCamera(deviceId);
  }

  async function ensureImageLoaded(image, message) {
    if (!image) return false;
    if (image.complete && image.naturalWidth) return true;

    updateStatus(message);
    try {
      await image.decode();
      return Boolean(image.naturalWidth);
    } catch {
      return false;
    }
  }

  async function makePoseCanvas(sourceCanvas) {
    const backgroundLoaded = await ensureImageLoaded(backgroundImageRef.current, "Loading background");
    if (!backgroundLoaded) {
      updateStatus("Could not load selected background", "error");
      return null;
    }

    const width = sourceCanvas.width;
    const height = sourceCanvas.height;
    const poseCanvas = document.createElement("canvas");
    poseCanvas.width = width;
    poseCanvas.height = height;

    const ctx = poseCanvas.getContext("2d");
    const backgroundImage = backgroundImageRef.current;
    setCanvasCover(ctx, backgroundImage, backgroundImage.naturalWidth, backgroundImage.naturalHeight, 0, 0, poseCanvas.width, poseCanvas.height);

    const processedByLocalApi = await drawLocalBackgroundRemoval(ctx, sourceCanvas, width, height);
    if (processedByLocalApi) {
      console.info("[bg-removal] Using local API result from", BG_REMOVAL_URL);
      return poseCanvas;
    }

    console.info("[bg-removal] Using MediaPipe browser fallback");

    const segmenter = await getSegmenter();
    const fallbackCanvas = document.createElement("canvas");
    if (segmenter) {
      await drawSegmentedPerson(ctx, segmenter, fallbackCanvas, sourceCanvas, width, height, 0, 0, poseCanvas.width, poseCanvas.height);
    } else {
      drawBlackBackgroundReplacement(ctx, fallbackCanvas, sourceCanvas, width, height, 0, 0, poseCanvas.width, poseCanvas.height);
    }

    return poseCanvas;
  }

  async function drawLocalBackgroundRemoval(ctx, sourceCanvas, width, height) {
    try {
      console.info("[bg-removal] Sending frame to local API:", BG_REMOVAL_URL);
      updateStatus("Removing background");

      const frameBlob = await canvasToBlob(sourceCanvas);

      const formData = new FormData();
      formData.append("file", frameBlob, "camera.png");
      formData.append("model", "u2netp");
      formData.append("enhance_mode", "basic");
      formData.append("feather", "1");
      formData.append("output_format", "png");

      const response = await fetch(BG_REMOVAL_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const processedImage = await loadImageFromBlob(await response.blob());
      drawProcessedPerson(ctx, keyedCanvasRef.current, processedImage, 0, 0, width, height);
      console.info("[bg-removal] Local API completed successfully");
      return true;
    } catch (error) {
      console.warn("Local background removal failed, using browser fallback.", error);
      updateStatus("Local background removal unavailable, using fallback", "error");
      return false;
    }
  }

  function wait(milliseconds) {
    return new Promise((resolve) => {
      window.setTimeout(resolve, milliseconds);
    });
  }

  async function drawSheet(nextPoses = poses, nextPlacements = emojiPlacements) {
    const canvas = canvasRef.current;
    canvas.width = SHEET_WIDTH;
    canvas.height = SHEET_HEIGHT;

    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, SHEET_WIDTH, SHEET_HEIGHT);

    const frameLoaded = await ensureImageLoaded(frameImageRef.current, "Loading strip frame");
    const boxes = getSheetBoxes();
    imageBoxesRef.current = boxes;

    // Photos go down first and the frame is stamped over them, so every cut-out is
    // filled edge to edge and the frame's own artwork stays on top of the photo.
    boxes.forEach((box) => {
      const pose = nextPoses[box.poseIndex];
      if (!pose) return;

      const fill = getBleedBox(box);
      drawCoverSharp(
        ctx,
        pose.baseCanvas,
        pose.baseCanvas.width,
        pose.baseCanvas.height,
        fill.x,
        fill.y,
        fill.width,
        fill.height,
        0.5,
        0.5
      );
    });

    if (frameLoaded) {
      drawFrameOverlay(ctx, frameImageRef.current);
    }

    boxes.forEach((box) => {
      const placement = nextPlacements[box.poseIndex];
      if (!placement) return;
      const emojiImage = emojiImagesRef.current[placement.emoji.id];

      drawEmojiAt(
        ctx,
        emojiImage,
        box.x + box.width * placement.x,
        box.y + box.height * placement.y,
        box.width * 0.28
      );
    });

    const dataUrl = await encodeCanvasToDataUrl(canvas);
    setSheetUrl(dataUrl);
    updateStatus("Preview ready", "ready");
    return dataUrl;
  }

  function resetProject(nextStep = "start") {
    setPoses([]);
    setEmojiPlacements([]);
    setSheetUrl("");
    setStep(nextStep);
    updateStatus(nextStep === "start" ? "Tap capture to begin" : `Capture ${STRIP_COUNT} photos`, "ready");
  }

  function chooseBackground(option) {
    setSelectedBackground(option);
    setSheetUrl("");
  }

  function chooseNextBackground() {
    setSelectedBackground(BACKGROUND_OPTIONS[(selectedBackgroundIndex + 1) % BACKGROUND_OPTIONS.length]);
    setSheetUrl("");
  }

  function goToCapture() {
    setPoses([]);
    setEmojiPlacements(Array(STRIP_COUNT).fill(null));
    setSheetUrl("");
    setSelectedDevice("");
    setStep("capture");
  }

  async function goToEdit() {
    // Show the edit screen first so the controls are usable while the sheet
    // renders; the preview slot falls back to its placeholder until sheetUrl lands.
    setStep("edit");
    await drawSheet(poses, emojiPlacements);
  }

  function captureCurrentFrame() {
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) {
      updateStatus("Camera frame is not ready yet", "error");
      return null;
    }

    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = video.videoWidth;
    frameCanvas.height = video.videoHeight;
    frameCanvas.getContext("2d").drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    return frameCanvas;
  }

  async function processCapturedFrame(frameCanvas) {
    // No masking and no re-encoding: the untouched full-resolution camera canvas is
    // what the sheet samples from, so nothing is thrown away before compositing.
    return {
      id: crypto.randomUUID(),
      baseCanvas: frameCanvas,
    };
  }

  async function captureImage() {
    if (isCapturingSequence) return;

    if (!cameraReady) {
      updateStatus("Camera frame is not ready yet", "error");
      return;
    }

    setIsCapturingSequence(true);
    setIsProcessingCaptures(false);

    try {
      const processingTasks = [];

      for (let photoIndex = 0; photoIndex < STRIP_COUNT; photoIndex += 1) {
        for (let value = 3; value >= 1; value -= 1) {
          setCountdown(value);
          updateStatus(`Photo ${photoIndex + 1} in ${value}`);
          await wait(1000);
        }

        setCountdown(null);
        const frameCanvas = captureCurrentFrame();
        if (!frameCanvas) return;

        console.info(`[capture] Photo ${photoIndex + 1} frame captured; processing in background`);
        processingTasks.push(processCapturedFrame(frameCanvas));
        updateStatus(`Photo ${photoIndex + 1} captured`, "ready");
        await wait(350);
      }

      setIsCapturingSequence(false);
      setIsProcessingCaptures(true);
      updateStatus("Preparing final output");
      const processedPoses = (await Promise.all(processingTasks)).filter(Boolean);
      if (processedPoses.length !== STRIP_COUNT) {
        updateStatus("Some photos could not be processed", "error");
        return;
      }

      const nextPlacements = Array(STRIP_COUNT).fill(null);
      setPoses(processedPoses);
      setEmojiPlacements(nextPlacements);
      // Switch screens first so the controls are usable while the sheet renders.
      setStep("edit");
      const sheetDataUrl = await drawSheet(processedPoses, nextPlacements);
      saveOutputImage(sheetDataUrl);
    } finally {
      setCountdown(null);
      setIsCapturingSequence(false);
      setIsProcessingCaptures(false);
    }
  }

  async function placeEmoji(event) {
    if (!sheetUrl || !activeEmoji) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const sheetX = ((event.clientX - rect.left) / rect.width) * (SHEET_WIDTH / 2);
    const sheetY = ((event.clientY - rect.top) / rect.height) * SHEET_HEIGHT;
    const box = imageBoxesRef.current.find(
      (candidate) =>
        sheetX >= candidate.x &&
        sheetX <= candidate.x + candidate.width &&
        sheetY >= candidate.y &&
        sheetY <= candidate.y + candidate.height
    );

    if (!box) {
      updateStatus("Click inside a captured photo to place the emoji", "error");
      return;
    }

    const nextPlacements = [...emojiPlacements];
    nextPlacements[box.poseIndex] = {
      emoji: activeEmoji,
      x: Math.max(0, Math.min(1, (sheetX - box.x) / box.width)),
      y: Math.max(0, Math.min(1, (sheetY - box.y) / box.height)),
    };

    setEmojiPlacements(nextPlacements);
    await drawSheet(poses, nextPlacements);
  }

  async function selectEmojiForFixedSlot(emoji) {
    const existingIndex = emojiPlacements.findIndex((placement) => placement?.emoji.id === emoji.id);
    const nextPlacements = [...emojiPlacements];

    if (existingIndex >= 0) {
      nextPlacements[existingIndex] = null;
    } else {
      const emptyIndex = nextPlacements.findIndex((placement, index) => index < STRIP_COUNT && !placement);
      const targetIndex = emptyIndex >= 0 ? emptyIndex : STRIP_COUNT - 1;
      nextPlacements[targetIndex] = {
        emoji,
        ...FIXED_EMOJI_POSITIONS[targetIndex],
      };
    }

    setActiveEmoji(emoji);
    setEmojiPlacements(nextPlacements);
    await drawSheet(poses, nextPlacements);
  }

  async function saveOutputImage(imageData) {
    try {
      updateStatus("Saving photo strip");
      // imageData is a lossless PNG data URL straight from the canvas, so the
      // saved file is bit-for-bit identical to what was composited — no quality loss.
      const response = await fetch("/api/save-output-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageData }),
      });

      if (!response.ok) {
        throw new Error(`Save failed with status ${response.status}`);
      }

      const { filename } = await response.json();
      updateStatus(`Saved to output_images/${filename}`, "ready");
    } catch (error) {
      console.warn("Could not save the photo strip.", error);
      updateStatus("Could not save the photo strip", "error");
    }
  }

  function printSheet() {
    if (!sheetUrl) return;
    window.print();
  }

  return (
    <div className="app">
      {step === "start" && (
        <main className="start-screen">
          <div className="kiosk-stage start-stage" style={{ transform: `translate(-50%, -50%) scale(${stageScale})` }}>
            <button className="start-capture-btn" type="button" onClick={goToCapture} aria-label="Start capture">
              <img src="/start.png" alt="" />
            </button>
          </div>
        </main>
      )}

      {step === "capture" && (
        <main className="capture-screen">
          <div className="kiosk-stage capture-stage" style={{ transform: `translate(-50%, -50%) scale(${stageScale})` }}>
            <div className="capture-video-box">
              <video ref={videoRef} autoPlay playsInline muted />
              {countdown && <div className="countdown-overlay">{countdown}</div>}
            </div>
            <button
              className="capture-btn"
              type="button"
              disabled={!cameraReady || isCapturingSequence || isProcessingCaptures}
              onClick={captureImage}
              aria-label={captureLabel}
            >
              <img src="/capture.png" alt="" />
            </button>
          </div>
        </main>
      )}

      {step === "edit" && (
        <main className="editor-screen">
          <div className="kiosk-stage edit-stage" style={{ transform: `translate(-50%, -50%) scale(${stageScale})` }}>
            <div className="strip-preview-box">
              {sheetUrl ? <img src={sheetUrl} alt="Editable output preview" /> : "Preparing preview"}
            </div>

            <button className="print-btn" type="button" onClick={printSheet} aria-label="Print">
              <img src="/print.png" alt="" />
            </button>

            <button className="home-btn" type="button" onClick={() => resetProject("start")} aria-label="Home">
              <img src="/home.png" alt="" />
            </button>
          </div>
        </main>
      )}

      {sheetUrl && (
        <div className="print-output" aria-hidden="true">
          <img src={sheetUrl} alt="" />
        </div>
      )}

      <canvas ref={canvasRef} />
    </div>
  );
}
