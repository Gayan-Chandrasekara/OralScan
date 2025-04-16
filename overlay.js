const video = document.getElementById("video");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const captureBtn = document.getElementById("captureBtn");

let session = null;

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
}

function cropToSquare(imageData, width, height) {
  const size = Math.min(width, height);
  const x = (width - size) / 2;
  const y = (height - size) / 2;
  const square = ctx.getImageData(x, y, size, size);
  return square;
}

function preprocess(imageData) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 512;
  tempCanvas.height = 512;
  const tempCtx = tempCanvas.getContext("2d");

  // Draw cropped square and resize to 512x512
  const squareCanvas = document.createElement("canvas");
  squareCanvas.width = squareCanvas.height = imageData.width;
  squareCanvas.getContext("2d").putImageData(imageData, 0, 0);
  tempCtx.drawImage(squareCanvas, 0, 0, 512, 512);

  const resized = tempCtx.getImageData(0, 0, 512, 512).data;
  const input = new Float32Array(1 * 3 * 512 * 512);

  for (let i = 0; i < 512 * 512; i++) {
    input[i] = resized[i * 4] / 255;         // R
    input[i + 512 * 512] = resized[i * 4 + 1] / 255; // G
    input[i + 2 * 512 * 512] = resized[i * 4 + 2] / 255; // B
  }

  return new ort.Tensor("float32", input, [1, 3, 512, 512]);
}

function renderMask(mask) {
  const imageData = ctx.createImageData(512, 512);
  for (let i = 0; i < 512 * 512; i++) {
    const cls = mask[i];
    const color = cls === 1 ? [255, 0, 0] : cls === 2 ? [0, 0, 0] : [0, 0, 0];
    imageData.data.set([...color, 100], i * 4); // RGBA
  }
  ctx.putImageData(imageData, 0, 0);
}

async function runInference() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const fullImage = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const cropped = cropToSquare(fullImage, canvas.width, canvas.height);
  const inputTensor = preprocess(cropped);

  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  const output = results[Object.keys(results)[0]].data;

  const mask = new Uint8Array(512 * 512);
  for (let i = 0; i < 512 * 512; i++) {
    let maxVal = -Infinity, maxIdx = 0;
    for (let j = 0; j < 3; j++) {
      const val = output[j * 512 * 512 + i];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = j;
      }
    }
    mask[i] = maxIdx;
  }

  renderMask(mask);
}

captureBtn.onclick = runInference;

(async () => {
  await initCamera();
  session = await ort.InferenceSession.create("https://firebasestorage.googleapis.com/v0/b/oral-cancer-detector-1e004.appspot.com/o/model.onnx?alt=media");
})();
