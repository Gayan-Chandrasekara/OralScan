let session = null;
let modelUrl = "https://firebasestorage.googleapis.com/v0/b/oral-cancer-detector-1e004.firebasestorage.app/o/model.onnx?alt=media&token=d17886be-78e8-4916-a377-4ff3c6b7ef68";

// Elements
const downloadBtn = document.getElementById("downloadModel");
const progressBar = document.getElementById("modelProgress");
const progressText = document.getElementById("progressPercent");
const captureBtn = document.getElementById("capture");
const fileInput = document.getElementById("fileInput");
const imageElement = document.getElementById("capturedImage");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// ONNX model download with progress
async function downloadModelWithProgress(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Failed to fetch model");

  const contentLength = +response.headers.get("Content-Length");
  const reader = response.body.getReader();
  let receivedLength = 0;
  let chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedLength += value.length;

    const percent = Math.round((receivedLength / contentLength) * 100);
    progressBar.value = percent;
    progressText.textContent = `${percent}%`;
  }

  let chunksAll = new Uint8Array(receivedLength);
  let position = 0;
  for (let chunk of chunks) {
    chunksAll.set(chunk, position);
    position += chunk.length;
  }

  console.log("Model downloaded. Loading...");
  session = await ort.InferenceSession.create(chunksAll.buffer);
  console.log("Model loaded.");
}

// Trigger model download
downloadBtn.addEventListener("click", async () => {
  progressBar.style.display = "block";
  progressText.textContent = "0%";
  try {
    await downloadModelWithProgress(modelUrl);
    downloadBtn.disabled = true;
    progressText.textContent = "Download complete ✅";
  } catch (err) {
    console.error(err);
    progressText.textContent = "Download failed ❌";
  }
});

// Trigger file input
captureBtn.addEventListener("click", () => {
  fileInput.click();
});

// On file selected
fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file || !session) return;

  const img = await loadImage(file);
  const square = cropToSquare(img);
  const resized = await resizeImage(square, 512, 512);
  imageElement.src = resized.toDataURL();

  const inputTensor = preprocessImage(resized);
  const output = await session.run({ [session.inputNames[0]]: inputTensor });
  const mask = new Uint8Array(output[session.outputNames[0]].data);
  const [height, width] = [512, 512];
  const result = postprocessMask(mask, width, height);

  canvas.width = width;
  canvas.height = height;
  ctx.putImageData(result, 0, 0);
});

// Load image from file input
function loadImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = URL.createObjectURL(file);
  });
}

// Crop to square
function cropToSquare(image) {
  const canvas = document.createElement("canvas");
  const size = Math.min(image.width, image.height);
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const sx = (image.width - size) / 2;
  const sy = (image.height - size) / 2;
  ctx.drawImage(image, sx, sy, size, size, 0, 0, size, size);
  return canvas;
}

// Resize canvas image to 512x512
function resizeImage(sourceCanvas, width, height) {
  return new Promise((resolve) => {
    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = width;
    resizedCanvas.height = height;
    const ctx = resizedCanvas.getContext("2d");
    ctx.drawImage(sourceCanvas, 0, 0, width, height);
    resolve(resizedCanvas);
  });
}

// Preprocess image to Float32 tensor [1,3,512,512]
function preprocessImage(canvas) {
  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { data, width, height } = imgData;
  const floatData = new Float32Array(3 * width * height);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) {
      floatData[c * width * height + i] = ((data[i * 4 + c] / 255.0) - mean[c]) / std[c];
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, height, width]);
}

// Postprocess mask and overlay on canvas
function postprocessMask(mask, width, height) {
  const colorMap = {
    0: [0, 0, 0],        // background
    1: [255, 0, 0],      // lesion
    2: [0, 0, 0]         // mouth
  };

  const imageData = new ImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const label = mask[i];
    const [r, g, b] = colorMap[label] || [0, 0, 0];
    imageData.data[i * 4 + 0] = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = 128; // semi-transparent
  }
  return imageData;
}
