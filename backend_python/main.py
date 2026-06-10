from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import tempfile
import os

from predict_cnn import predict_audio

app = FastAPI()

# =====================================
# ENABLE CORS
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# ROOT
# =====================================

@app.get("/")
def root():

    return {
        "status": "Baby Cry Detection API Running"
    }

# =====================================
# PREDICT
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # create temporary wav file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        content = await file.read()

        temp_audio.write(content)

        temp_path = temp_audio.name

    try:

        # predict
        result = predict_audio(temp_path)

        return result

    finally:

        # delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)