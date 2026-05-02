import os
import asyncio
from typing import Optional, Set
from contextlib import asynccontextmanager

import numpy as np
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from engine.resnet_engine import ResNetEngine
from audio.capture import MicrophoneCapture, FileCapture
from audio.ring_buffer import AudioRingBuffer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "..", "frontend", "dist")
MODELS_DIR = os.path.join(BASE_DIR, "models")

class AppState:
    def __init__(self):
        self.engine: Optional[ResNetEngine] = None
        self.clients: Set[WebSocket] = set()

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH", "./models/best_resnet18_gtzan.pth")
    # 读取环境变量，默认值为 0.2，以匹配训练时的架构
    dropout_val = float(os.environ.get("MODEL_DROPOUT", "0.2")) 
    
    state.engine = ResNetEngine(model_path=model_path, dropout=dropout_val)
    print(f"[Engine] Loaded from {model_path} on {state.engine.device} with dropout={dropout_val}")
    yield
    for ws in state.clients:
        await ws.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

# 必须挂载 models，否则前端无法直接获取 ONNX 模型
if os.path.exists(MODELS_DIR):
    app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")

@app.get("/")
async def root():
    index = os.path.join(DIST_DIR, "index.html")
    return FileResponse(index) if os.path.exists(index) else {"message": "Music Genre API"}

@app.get("/api/genres")
async def genres():
    return {"genres": state.engine.genres}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    import uuid
    ext = os.path.splitext(file.filename)[1]
    tmp_path = f"/tmp/{uuid.uuid4().hex}{ext}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    return {"file_path": tmp_path, "filename": file.filename}

# ==================== WebSocket: Unified Inference ====================
@app.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.clients.add(websocket)

    capture: Optional[object] = None
    buffer: Optional[AudioRingBuffer] = None
    infer_task: Optional[asyncio.Task] = None
    patch_probs: list =[]
    stream_mode = False

    try:
        while True:
            msg = await websocket.receive_json()
            cmd = msg.get("command")

            if cmd == "start_stream":
                patch_probs =[]
                stream_mode = True

            elif cmd == "audio_patch" and stream_mode:
                audio = np.array(msg["data"], dtype=np.float32)
                sr = msg.get("sr", 22050)
                req_id = msg.get("request_id")

                # 将 CPU 密集的重采样和推理放进线程池以防止阻塞主线程 (Event Loop)
                def process_patch(audio_data, current_sr):
                    if current_sr != state.engine.sample_rate:
                        audio_data = librosa.resample(
                            audio_data.astype(np.float32),
                            orig_sr=current_sr,
                            target_sr=state.engine.sample_rate,
                        )

                    ps = state.engine.patch_samples
                    if len(audio_data) < ps:
                        audio_data = np.pad(audio_data, (0, ps - len(audio_data)))
                    elif len(audio_data) > ps:
                        audio_data = audio_data[:ps]

                    return state.engine.predict(audio_data)

                loop = asyncio.get_event_loop()
                probs = await loop.run_in_executor(None, process_patch, audio, sr)
                
                patch_probs.append(probs)
                top5_idx = np.argsort(probs)[::-1][:5]
                top5 = [
                    {"genre": state.engine.genres[i], "probability": float(probs[i])}
                    for i in top5_idx
                ]

                await websocket.send_json(
                    {
                        "type": "patch",
                        "request_id": req_id,
                        "timestamp": msg.get("timestamp", 0),
                        "probabilities": probs.tolist(),
                        "top5": top5,
                    }
                )

            elif cmd == "stop":
                if capture:
                    capture.stop()
                if infer_task and not infer_task.done():
                    infer_task.cancel()

                if patch_probs:
                    mean = np.mean(patch_probs, axis=0)
                    final_idx = np.argsort(mean)[::-1][:5]
                    final_top5 = [
                        {"genre": state.engine.genres[i], "probability": float(mean[i])}
                        for i in final_idx
                    ]
                    await websocket.send_json(
                        {
                            "type": "final",
                            "top5": final_top5,
                            "distribution": {
                                g: float(mean[i])
                                for i, g in enumerate(state.engine.genres)
                            },
                        }
                    )
                else:
                    await websocket.send_json({"type": "stopped"})

                stream_mode = False
                patch_probs =[]

    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(websocket)
        if capture:
            capture.stop()
        if infer_task and not infer_task.done():
            infer_task.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)