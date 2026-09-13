let session;
let video;
let canvas;
let ctx;
let isProcessing = false;

document.getElementById('startBtn').addEventListener('click', async () => {
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('progress').style.display = 'block';
    await init();
});

async function init() {
    video = document.createElement('video');
    video.autoplay = true;
    video.playsInline = true;

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 640 } });
    video.srcObject = stream;
    
    await new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve();
        };
    });

    const progressDiv = document.getElementById('progress');
    
    try {
        const response = await fetch('./model/yolo26n-cls.onnx');
        const contentLength = response.headers.get('content-length');
        
        if (!contentLength) {
            progressDiv.textContent = "Загрузка модели...";
            const buffer = await response.arrayBuffer();
            progressDiv.textContent = "Инициализация...";
            session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] });
        } else {
            const total = parseInt(contentLength, 10);
            let loaded = 0;
            const reader = response.body.getReader();
            const chunks = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                loaded += value.length;
                const percent = Math.round((loaded / total) * 100);
                progressDiv.textContent = `Загрузка модели: ${percent}%`;
            }

            const buffer = new Uint8Array(loaded);
            let position = 0;
            for (let chunk of chunks) {
                buffer.set(chunk, position);
                position += chunk.length;
            }

            progressDiv.textContent = "Инициализация модели...";
            session = await ort.InferenceSession.create(buffer.buffer, { executionProviders: ['wasm'] });
        }
        
        progressDiv.style.display = 'none';
        detectFrame();
    } catch (err) {
        progressDiv.textContent = "Ошибка загрузки: " + err.message;
        console.error(err);
    }
}

async function detectFrame() {
    if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
        isProcessing = true;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const tensor = preprocess(canvas);
        const feeds = { images: tensor };
        
        try {
            const results = await session.run(feeds);
            console.log(results);
        } catch (err) {
            console.error(err);
        }
        
        isProcessing = false;
    }

    requestAnimationFrame(detectFrame);
}

function preprocess(canvasElement) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 640;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvasElement, 0, 0, 640, 640);
    
    const imgData = tempCtx.getImageData(0, 0, 640, 640);
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
