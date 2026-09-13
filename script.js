let session;

async function loadModel() {
    session = await ort.InferenceSession.create('./model/yolo26n-cls.onnx', { executionProviders: ['wasm'] });
}

loadModel();

document.getElementById('imageInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const canvas = document.getElementById('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        const tensor = preprocess(img);
        const feeds = { images: tensor };
        const results = await session.run(feeds);
        console.log(results);
    };
});

function preprocess(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 640, 640);
    const imgData = ctx.getImageData(0, 0, 640, 640);
    const data = imgData.data;

    const red = new Float32Array(640 * 640);
    const green = new Float32Array(640 * 640);
    const blue = new Float32Array(640 * 640);

    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        red[idx] = data[i] / 255.0;
        green[idx] = data[i + 1] / 255.0;
        blue[idx] = data[i + 2] / 255.0;
    }

    const inputData = new Float32Array(3 * 640 * 640);
    inputData.set(red, 0);
    inputData.set(green, 640 * 640);
    inputData.set(blue, 2 * 640 * 640);

    return new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
}
