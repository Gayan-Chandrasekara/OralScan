const modelUrl = "https://firebasestorage.googleapis.com/v0/b/oral-cancer-detector-1e004.firebasestorage.app/o/model.onnx?alt=media&token=d17886be-78e8-4916-a377-4ff3c6b7ef68";
let session = null;

// Class colors
const classColors = {
  0: [0, 0, 0],
  1: [255, 0, 0],
  2: [0, 0, 0]
};

async function loadModel() {
  session = await ort.InferenceSession.create(modelUrl);
  console.log("Model loaded");
  startCamera();
}

function cropToSquare(imageData) {
  const size = Math.min(imageData.width, imageData.height);
  const offsetX = (imageData.width - size) / 2;
  const offsetY = (imageData.height - size) / 2;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageData, offsetX, offsetY, size, size, 0, 0, size, size);
  return canvas;
}

function preprocess(canvas) {
  const resized = document.createElement('canvas');
  resized.width = 512;
  resized.height = 512;
  const ctx = resized.getContext('2d');
  ctx.drawImage(canvas, 0, 0, 512, 512);

  const imageData = ctx.getImageData(0, 0, 512, 512);
  const data = imageData.data;

  const input = new Float32Array(1 * 3 * 512 * 512);
  for (let i = 0; i < 512 * 512; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    input[i] = (r - 0.485) / 0.229;
    input[i + 512 * 512] = (g - 0.456) / 0.224;
    input[i + 2 * 512 * 512] = (b - 0.406) / 0.225;
  }

  return new ort.Tensor("float32", input, [1, 3, 512, 512]);
}

async function runInference(tensor) {
  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;
  return new Uint8Array(output).slice(); // Argmax should be done in model
}

function overlayMask(mask, canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(512, 512);

  for (let i = 0; i < 512 * 512; i++) {
    const classIdx = mask[i];
    const [r, g, b] = classColors[classIdx] || [0, 0, 0];
    imageData.data[i * 4] = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = 120; // Alpha
  }

  ctx.putImageData(imageData, 0, 0);
}

function startCamera() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('outputCanvas');
  navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      setInterval(async () => {
        const frame = cropToSquare(video);
        const tensor = preprocess(frame);
        const mask = await runInference(tensor);
        overlayMask(mask, canvas);
      }, 1000);
    };
  });
}

loadModel();
