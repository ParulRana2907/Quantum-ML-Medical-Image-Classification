
import logging
import sys
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import joblib


# --------------------------------------------------
# 1. CONFIGURE LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("quantum_ml")


# --------------------------------------------------
# 2. INITIALIZE FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="Quantum ML Pneumonia Classification",
    description="Chest X-ray image classification using ML",
    version="1.0"
)


# --------------------------------------------------
# 3. LOAD SAVED MODEL
# --------------------------------------------------

bundle = joblib.load("model/best_model_pipeline.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
pca = bundle["pca"]

IMG_SIZE = bundle["image_size"]
class_names = bundle["class_names"]

logger.info("Model loaded successfully: %s", type(model).__name__)


# --------------------------------------------------
# 4. HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Pneumonia Classification</title>
    </head>
    <body style="font-family: Arial; text-align:center; padding:40px;">
        <h1>Chest X-Ray Classification</h1>
        <p>Upload an X-ray image for an experimental model prediction.</p>

        <form action="/predict-page" method="post"
              enctype="multipart/form-data">
            <input type="file" name="file"
                   accept="image/jpeg,image/png" required>
            <button type="submit">Predict</button>
        </form>

        <p>Academic project only; not a medical diagnosis.</p>
    </body>
    </html>
    """


# --------------------------------------------------
# 5. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__
    }


# --------------------------------------------------
# 6. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image, dtype=np.float32) / 255.0
    flattened = image_array.reshape(1, -1)

    scaled = scaler.transform(flattened)
    reduced = pca.transform(scaled)

    return reduced


# --------------------------------------------------
# 7. PREDICTION LOGGING HELPER
# --------------------------------------------------

def log_prediction(endpoint, prediction, confidence, file_size, content_type):

    logger.info(
        "PREDICTION_SUCCESS | endpoint=%s | prediction=%s "
        "| confidence=%.2f%% | image_size_bytes=%d | content_type=%s",
        endpoint,
        prediction,
        confidence,
        file_size,
        content_type
    )


# --------------------------------------------------
# 8. PREDICTION API
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(
            "INVALID_FILE_TYPE | endpoint=/predict | content_type=%s",
            file.content_type
        )

        raise HTTPException(
            status_code=415,
            detail="Please upload a JPG or PNG image"
        )

    contents = await file.read()

    try:
        processed = preprocess_image(contents)

        prediction = int(model.predict(processed)[0])

        probabilities = model.predict_proba(processed)[0]
        confidence = float(np.max(probabilities))

        result = class_names[prediction]

        log_prediction(
            endpoint="/predict",
            prediction=result,
            confidence=confidence * 100,
            file_size=len(contents),
            content_type=file.content_type
        )

        return {
            "prediction": result,
            "confidence": round(confidence * 100, 2)
        }

    except (UnidentifiedImageError, OSError):

        logger.warning(
            "INVALID_IMAGE | endpoint=/predict | image_size_bytes=%d",
            len(contents)
        )

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image"
        )

    except Exception:

        logger.exception("PREDICTION_ERROR | endpoint=/predict")

        raise HTTPException(
            status_code=500,
            detail="Unable to process this image"
        )


# --------------------------------------------------
# 9. BROWSER PREDICTION RESULT
# --------------------------------------------------

@app.post("/predict-page", response_class=HTMLResponse)
async def predict_page(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:

        logger.warning(
            "INVALID_FILE_TYPE | endpoint=/predict-page | content_type=%s",
            file.content_type
        )

        return HTMLResponse(
            "<h3>Please upload a JPG or PNG image.</h3>",
            status_code=415
        )

    try:
        contents = await file.read()

        processed = preprocess_image(contents)

        prediction = int(model.predict(processed)[0])
        result = class_names[prediction]

        probabilities = model.predict_proba(processed)[0]
        confidence = float(np.max(probabilities))

        log_prediction(
            endpoint="/predict-page",
            prediction=result,
            confidence=confidence * 100,
            file_size=len(contents),
            content_type=file.content_type
        )

        return f"""
        <html>
        <body style="font-family:Arial;text-align:center;padding:40px;">
            <h1>Prediction Result</h1>
            <h2>{result}</h2>
            <p>Confidence: {confidence * 100:.2f}%</p>
            <p>Experimental machine learning output only.</p>
            <p>This result is not a medical diagnosis.</p>
            <a href="/">Try another image</a>
        </body>
        </html>
        """

    except (UnidentifiedImageError, OSError):

        logger.warning(
            "INVALID_IMAGE | endpoint=/predict-page | image_size_bytes=%d",
            len(contents)
        )

        return HTMLResponse(
            "<h3>Invalid image. Please upload a valid JPG or PNG.</h3>",
            status_code=400
        )

    except Exception:

        logger.exception("PREDICTION_ERROR | endpoint=/predict-page")

        return HTMLResponse(
            "<h3>Unable to process this image.</h3>",
            status_code=500
        )
