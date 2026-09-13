let session;
let video;
let canvas;
let ctx;
let isProcessing = false;

// Список классов (замените на свои, если у вашей модели другая номенклатура)
const classNames = ["Класс 0", "Класс 1", "Класс 2"]; 

document.getElementById('downloadBtn').addEventListener('click', async () => {
    const downloadBtn = document.getElementById('downloadBtn');
    const progressDiv = document.getElementById('progress');
    const startBtn = document.getElementById('startBtn');

    downloadBtn.style.display = 'none';
    progressDiv.style.display = 'block';

    try {
        const response = await fetch('./model/yolo26n-cls.onnx');
        const contentLength = response.headers.get('content-length');
        
        let buffer;
        if (!contentLength) {
            progressDiv.textContent = "Скачивание модели...";
            buffer = await response.arrayBuffer();
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
                progressDiv.textContent = `Скачивание: ${percent}%`;
            }

            buffer = new Uint8Array(loaded);
            let position = 0;
            for (let chunk of chunks) {
                buffer.set(chunk, position);
                position += chunk.length;
            }
            buffer = buffer.buffer;
        }

        progressDiv.textContent = "Инициализация...";
        session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] });
        
        progressDiv.style.display = 'none';
        startBtn.style.display = 'inline-block';
    } catch (err) {
        progressDiv.textContent = "Ошибка: " + err.message;
        console.error(err);
    }
});

document.getElementById('startBtn').addEventListener('click', async () => {
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('canvas').style.display = 'inline-block';
    await initCamera();
});

async function initCamera() {
    video = document.createElement('video');
    video.autoplay = true;
    video.playsInline = true;

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: { ideal: 640 }, 
            height: { ideal: 640 } 
        } 
    });
    video.srcObject = stream;
    
    await new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve();
        };
    });

    detectFrame();
}

async function detectFrame() {
    if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
        isProcessing = true;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        ctx.filter = 'grayscale(100%)';
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.filter = 'none';

        const tensor = preprocess(canvas);
        const feeds = { images: tensor };
        
        try {
            const results = await session.run(feeds);
            const outputName = session.outputNames[0];
            const output = results[outputName];
            const data = output.data;

            let maxProb = -1;
            let maxClassId = -1;
            for (let i = 0; i < data.length; i++) {
                if (data[i] > maxProb) {
                    maxProb = data[i];
                    maxClassId = i;
                }
            }

            const className = classNames[maxClassId] || `Class ${maxClassId}`;
            
            ctx.fillStyle = 'lime';
            ctx.font = 'bold 22px Arial';
            ctx.fillText(`${className}: ${(maxProb * 100).toFixed(1)}%`, 20, 40);

        } catch (err) {
            console.error(err);
        }
        
        await new Promise(resolve => setTimeout(resolve, 80));
        isProcessing = false;
    }

    requestAnimationFrame(detectFrame);
}

function preprocess(canvasElement) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 640;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.filter = 'grayscale(100%)';
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
