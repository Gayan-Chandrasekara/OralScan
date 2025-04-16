let model, inputImage, canvas, capturedImage, fileInput, captureButton;

async function loadModel() {
    const modelUrl = "https://firebasestorage.googleapis.com/v0/b/oral-cancer-detector-1e004.firebasestorage.app/o/model.onnx?alt=media&token=d17886be-78e8-4916-a377-4ff3c6b7ef68"; // Your ONNX model URL
    model = await ort.InferenceSession.create(modelUrl);
}

function denormalize(tensor) {
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    tensor = tensor.mul(std).add(mean);
    return tensor.clamp(0, 1);
}

function overlaySegmentation(inputTensor, mask, colorMap, alpha = 0.4) {
    const colorMask = new Array(mask.length).fill(0).map(() => new Array(mask[0].length).fill([0, 0, 0]));

    for (let idx in colorMap) {
        const color = colorMap[idx];
        for (let i = 0; i < mask.length; i++) {
            for (let j = 0; j < mask[i].length; j++) {
                if (mask[i][j] === parseInt(idx)) {
                    colorMask[i][j] = color;
                }
            }
        }
    }

    let inputImage = denormalize(inputTensor).squeeze().permute([1, 2, 0]).cpu().numpy();
    inputImage = inputImage.mul(255).astype("uint8");

    let overlay = inputImage.map((row, i) =>
        row.map((pixel, j) => {
            const color = colorMask[i][j];
            return (1 - alpha) * pixel + alpha * color;
        })
    );

    return overlay;
}

// Handle the file input or photo capture
function handleCapture(event) {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async function (e) {
            const img = new Image();
            img.onload = function () {
                processImage(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

async function processImage(image) {
    const width = image.width;
    const height = image.height;

    // Crop to square logic
    const minDim = Math.min(width, height);
    const sx = (width - minDim) / 2;
    const sy = (height - minDim) / 2;

    // Create a new canvas to crop and resize the image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = minDim;
    canvas.height = minDim;

    // Draw the cropped image onto the canvas
    ctx.drawImage(image, sx, sy, minDim, minDim, 0, 0, minDim, minDim);

    // Get the cropped image as a data URL
    const croppedImageUrl = canvas.toDataURL();

    // Create a new image with the cropped data
    const croppedImage = new Image();
    croppedImage.onload = function () {
        processCroppedImage(croppedImage);
    };
    croppedImage.src = croppedImageUrl;
}

async function processCroppedImage(image) {
    const width = image.width;
    const height = image.height;

    // Convert image to tensor
    let tensor = await imageToTensor(image);

    // Resize image to 512x512 and prepare it for inference
    const resizedImage = await resizeImage(tensor, 512, 512);

    // Perform inference with the model
    const inputTensor = resizedImage.cpu().numpy();
    const output = await runInference(inputTensor);
    const predictedMask = np.argmax(output, axis=1).squeeze(0);

    // Class color mapping
    const classColors = {
        0: [0, 0, 0],    // Background
        1: [255, 0, 0],  // Lesion
        2: [0, 0, 0]     // Mouth
    };

    // Generate the overlay
    const overlayResult = overlaySegmentation(inputTensor, predictedMask, classColors, 0.4);

    // Display the result
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    const overlayImageData = ctx.createImageData(image.width, image.height);
    overlayImageData.data.set(overlayResult.flat());
    ctx.putImageData(overlayImageData, 0, 0);

    capturedImage.src = image.src;
    capturedImage.style.display = "block";
}

// Helper to load image as a tensor
async function imageToTensor(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return tf.browser.fromPixels(imageData);
}

// Resize image to the target size
async function resizeImage(imageTensor, targetWidth, targetHeight) {
    return tf.image.resizeBilinear(imageTensor, [targetWidth, targetHeight]);
}

// Run inference on the model
async function runInference(inputTensor) {
    const inputName = model.inputNames[0];
    const outputName = model.outputNames[0];
    const result = await model.run({ [inputName]: inputTensor });
    return result[outputName];
}

// Initialize the application
window.onload = function () {
    fileInput = document.getElementById("fileInput");
    captureButton = document.getElementById("capture");
    capturedImage = document.getElementById("capturedImage");
    canvas = document.getElementById("canvas");

    loadModel();

    captureButton.addEventListener("click", () => {
        fileInput.click();  // Trigger file input when capture button is clicked
    });

    fileInput.addEventListener("change", handleCapture);
};
