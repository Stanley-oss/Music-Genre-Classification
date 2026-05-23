# File: ./backend/main.py
import os
import json
import struct
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
    dropout_val = float(os.environ.get("MODEL_DROPOUT", "0.2"))

    state.engine = ResNetEngine(model_path=model_path, dropout=dropout_val)
    print(
        f"[Engine] Loaded from {model_path} on {state.engine.device} with dropout={dropout_val}"
    )
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
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(DIST_DIR, "assets")),
        name="assets",
    )

if os.path.exists(MODELS_DIR):
    app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")


@app.get("/")
async def root():
    index = os.path.join(DIST_DIR, "index.html")
    return (
        FileResponse(index) if os.path.exists(index) else {"message": "Music Genre API"}
    )


@app.get("/api/genres")
async def genres():
    return {"genres": state.engine.genres}


# ==================== WebSocket: High-Performance Binary Inference ====================
@app.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.clients.add(websocket)

    patch_probs: list = []
    stream_mode = False

    try:
        while True:
            # 放弃使用 receive_json()，改为基础的 receive() 兼顾纯文本与二进制字节
            msg = await websocket.receive()

            # --- 处理 JSON 控制信令 ---
            if "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                cmd = payload.get("command")

                if cmd == "start_stream":
                    patch_probs = []
                    stream_mode = True

                elif cmd == "stop":
                    if patch_probs:
                        mean = np.mean(patch_probs, axis=0)
                        final_idx = np.argsort(mean)[::-1][:5]
                        final_top5 = [
                            {
                                "genre": state.engine.genres[i],
                                "probability": float(mean[i]),
                            }
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
                    patch_probs = []

            # --- 处理二进制高速流 (Binary Audio Payload) ---
            elif "bytes" in msg and msg["bytes"]:
                if not stream_mode:
                    continue

                raw_bytes = msg["bytes"]
                if len(raw_bytes) <= 16:
                    continue

                # 按照前段制定的协议解包 16 Bytes Header
                # <I: little-endian unsigned int (4 bytes)
                # <d: little-endian double float (8 bytes)
                req_id = int.from_bytes(raw_bytes[0:4], byteorder="little")
                sr = int.from_bytes(raw_bytes[4:8], byteorder="little")
                timestamp = struct.unpack("<d", raw_bytes[8:16])[0]

                # 从第 16 字节开始零拷贝读取 Float32 音频数组
                # 使用 .copy() 保证数据安全，防止底层内存池覆写
                audio = np.frombuffer(raw_bytes, dtype="<f4", offset=16).copy()

                def process_patch(audio_data, current_sr):
                    if current_sr != state.engine.sample_rate:
                        audio_data = librosa.resample(
                            audio_data,
                            orig_sr=current_sr,
                            target_sr=state.engine.sample_rate,
                        )
                    ps = state.engine.patch_samples
                    if len(audio_data) < ps:
                        audio_data = np.pad(audio_data, (0, ps - len(audio_data)))
                    elif len(audio_data) > ps:
                        audio_data = audio_data[:ps]

                    return state.engine.predict(audio_data)

                # 提交给线程池避免阻塞
                loop = asyncio.get_event_loop()
                probs = await loop.run_in_executor(None, process_patch, audio, sr)

                patch_probs.append(probs)
                top5_idx = np.argsort(probs)[::-1][:5]
                top5 = [
                    {"genre": state.engine.genres[i], "probability": float(probs[i])}
                    for i in top5_idx
                ]

                # 返回依然使用极小的 JSON
                await websocket.send_json(
                    {
                        "type": "patch",
                        "request_id": req_id,
                        "timestamp": timestamp,
                        "probabilities": probs.tolist(),
                        "top5": top5,
                    }
                )

    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(websocket)
