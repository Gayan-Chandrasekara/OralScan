const MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/oral-cancer-detector-1e004.firebasestorage.app/o/model.onnx?alt=media&token=d17886be-78e8-4916-a377-4ff3c6b7ef68";
const MODEL_KEY = "cached_model";
let session = null;

// Utility: Fetch and store model in IndexedDB
async function fetchAndCacheModel(progressCallback) {
  const response = await fetch(MODEL_URL);
  const reader = response.body.getReader();
  const contentLength = +response.headers.get('Content-Length');
  let receivedLength = 0;
  let chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedLength += value.length;
    if (progressCallback) {
      const percent = Math.floor((receivedLength / contentLength) * 100);
      progressCallback(percent);
    }
  }

  const modelArray = new Uint8Array(receivedLength);
  let position = 0;
  for (let chunk of chunks) {
    modelArray.set(chunk, position);
    position += chunk.length;
  }

  await saveModelToIndexedDB(modelArray);
  return modelArray;
}

// IndexedDB helpers
function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("ModelDB", 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("models");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveModelToIndexedDB(arrayBuffer) {
  const db = await openIndexedDB();
  const tx = db.transaction("models", "readwrite");
  tx.objectStore("models").put(arrayBuffer, MODEL_KEY);
  return tx.complete;
}

async function loadModelFromIndexedDB() {
  const db = await openIndexedDB();
  const tx = db.transaction("models", "readonly");
  const data = await tx.objectStore("models").get(MODEL_KEY);
  return data;
}

// Load ONNX model
async function loadModel() {
  const modelArrayBuffer = await loadModelFromIndexedDB();
  if (!modelArrayBuffer) throw new Error("Model not found in IndexedDB");
  session = await ort.InferenceSession.create(modelArrayBuffer);
  console.log("ONNX model loaded.");
}

// Image preprocessing
function preprocessImage(imageElement) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const minDim = Math.min(imageElement.width, imageElement.height);
  const sx = (imageElement.width - minDim) / 2;
  const sy = (imageElement.height - minDim) / 2;

  canvas.width = 512;
  canvas.height = 512;

  ctx.drawImage(imageElement, sx, sy, minDim, minDim, 0, 0, 512, 512);
  const imageData = ctx.getImageData(0, 0, 512, 512).data;

  const input = new Float32Array(1 * 3 * 512 * 512);
  for (let i = 0; i < 512 * 512; i++) {
    const r = imageData[i * 4] / 255;
    const g = imageData[i * 4 + 1] / 255;
    const b = imageData[i * 4 + 2] / 255;

    input[i] = (r - 0.485) / 0.229;
    input[i + 512 * 512] = (g - 0.456) / 0.224;
    input[i + 2 * 512 * 512] = (b - 0.406) / 0.225;
  }

  return new ort.Tensor("float32", input, [1, 3, 512, 512]);
}

// Inference + Overlay
async function runInference(imageElement) {
  const inputTensor = preprocessImage(imageElement);
  const output = await session.run({ [session.inputNames[0]]: inputTensor });
  const outputTensor = output[session.outputNames[0]];
  const prediction = outputTensor.data;
  const outputCanvas = document.getElementById("canvas");
  const ctx = outputCanvas.getContext("2d");
  const imageData = ctx.createImageData(512, 512);

  for (let i = 0; i < 512 * 512; i++) {
    const value = prediction[i * 3 + 1] > 0.5 ? 255 : 0; // Lesion class
    imageData.data[i * 4] = value;
    imageData.data[i * 4 + 1] = 0;
    imageData.data[i * 4 + 2] = 0;
    imageData.data[i * 4 + 3] = 100;
  }

  ctx.clearRect(0, 0, 512, 512);
  ctx.drawImage(imageElement, 0, 0, 512, 512);
  ctx.putImageData(imageData, 0, 0);
}

// UI Handlers
document.getElementById("downloadModel").addEventListener("click", async () => {
  const progressDiv = document.getElementById("progress");
  progressDiv.textContent = "Downloading...";
  try {
    await fetchAndCacheModel(percent => {
      progressDiv.textContent = `Downloading: ${percent}%`;
    });
    await loadModel();
    progressDiv.textContent = "Model downloaded and ready.";
  } catch (e) {
    progressDiv.textContent = "Download failed.";
    console.error(e);
  }
});

document.getElementById("capture").addEventListener("click", () => {
  document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const img = document.getElementById("capturedImage");
  img.src = URL.createObjectURL(file);
  img.onload = () => runInference(img);
});
