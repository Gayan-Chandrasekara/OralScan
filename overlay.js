const modelUrl = "https://firebasestorage.googleapis.com/v0/b/oral-cancer-detector-1e004.firebasestorage.app/o/model.onnx?alt=media&token=d17886be-78e8-4916-a377-4ff3c6b7ef68";
let session = null;
let video, canvas, ctx;

const classColors = {
  0: [0, 0, 0],       // Background
  1: [255, 0, 0],     // Lesion
  2: [0, 0, 0],       // Mouth (same as background for now)
};

window.onload = () => {
  video = document.getElementById("video");
  canvas = document.getElementById("outputCanvas");
  ctx = canvas.getContext("2d");

  loadModel();
};

async function loadModel() {
  try {
    session = await ort.InferenceSession.create(modelUrl);
    console.log("✅ Model successfully downloaded and loaded.");
    document.body.insertAdjacentHTML("beforeend", "<p style='color:green;'>Model loaded successfully ✅</p>");
    startCamera();
  } catch (error) {
    console.error("❌ Failed to load model:", error);
    document.body.insertAdjacentHTML("beforeend", `<p style='color:red;'>Error loading model ❌: ${error.message}</p>`);
  }
}

function startCamera() {
  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
    .then((stream) => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
        requestAnimationFrame(processFrame);
      };
    })
    .catch((err) => {
      console.error("❌ Failed to access camera:", err);
    });
}

async function processFrame() {
  if (!session) return requestAnimationFrame(processFrame);

  // Capture and preprocess
  const inputTensor = preprocessVideo(video);

  try {
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    const mask = argmax(output.data, output.dims[2], output.dims[3]);

    drawOverlay(video, mask);
  } catch (err) {
    console.error("❌ Inference error:", err);
  }

  requestAnimationFrame(processFrame);
}

function preprocessVideo(videoElement) {
  const size = 512;
  const tempCanvas = document.createElement("canvas");
  const ctx2 = tempCanvas.getContext("2d");

  // Crop square
  const minDim = Math.min(videoElement.videoWidth, videoElement.videoHeight);
  const sx = (videoElement.videoWidth - minDim) / 2;
  const sy = (videoElement.videoHeight - minDim) / 2;

  tempCanvas.width = size;
  tempCanvas.height = size;
  ctx2.drawImage(videoElement, sx, sy, minDim, minDim, 0, 0, size, size);

  const imageData = ctx2.getImageData(0, 0, size, size);
  const data = Float32Array.from(imageData.data);

  const input = new Float32Array(1 * 3 * size * size);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const offset = (y * size + x) * 4;
      for (let c = 0; c < 3; c++) {
        const value = data[offset + c] / 255.0;
        input[c * size * size + y * size + x] = (value - mean[c]) / std[c];
      }
    }
  }

  return new ort.Tensor("float32", input, [1, 3, size, size]);
}

function argmax(data, height, width) {
  const mask = new Uint8ClampedArray(height * width);
  const numClasses = data.length / (height * width);

  for (let i = 0; i < height * width; i++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let c = 0; c < numClasses; c++) {
      const val = data[c * height * width + i];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = c;
      }
    }
    mask[i] = maxIdx;
  }

  return mask;
}

function drawOverlay(video, mask) {
  const width = canvas.width = video.videoWidth;
  const height = canvas.height = video.videoHeight;

  const imageData = ctx.getImageData(0, 0, width, height);
  const overlay = new Uint8ClampedArray(width * height * 4);

  const scale = width / 512;

  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) {
      const idx = y * 512 + x;
      const color = classColors[mask[idx]];
      const dx = Math.floor(x * scale);
      const dy = Math.floor(y * scale);
      const destIdx = (dy * width + dx) * 4;

      overlay[destIdx] = color[0];
      overlay[destIdx + 1] = color[1];
      overlay[destIdx + 2] = color[2];
      overlay[destIdx + 3] = 100; // Alpha
    }
  }

  const overlayImage = new ImageData(overlay, width, height);
  ctx.drawImage(video, 0, 0, width, height);
  ctx.putImageData(overlayImage, 0, 0);
}
