
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import joblib
import io

# Initialize FastAPI
app = FastAPI(
    title="Quantum ML Pneumonia Classification",
    description="Chest X-ray image classification using ML",
    version="1.0"
)

# Load saved model pipeline
bundle = joblib.load("model/best_model_pipeline.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
pca = bundle["pca"]
IMG_SIZE = bundle["image_size"]
class_names = bundle["class_names"]


# Home page
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
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Predict</button>
        </form>
        <p>Academic project only; not a medical diagnosis.</p>
    </body>
    </html>
    """


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__
    }


# Image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image, dtype=np.float32) / 255.0
    flattened = image_array.reshape(1, -1)

    scaled = scaler.transform(flattened)
    reduced = pca.transform(scaled)

    return reduced


# Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
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

        return {
            "prediction": class_names[prediction],
            "confidence": round(confidence * 100, 2)
        }

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image"
        )

    except Exception as e:
        print("PREDICTION ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="Unable to process this image"
        )


# Browser prediction result
@app.post("/predict-page", response_class=HTMLResponse)
async def predict_page(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        return HTMLResponse(
            "<h3>Please upload a JPG or PNG image.</h3>",
            status_code=415
        )

    try:
        contents = await file.read()
        processed = preprocess_image(contents)

        prediction = int(model.predict(processed)[0])
        result = class_names[prediction]

        return f"""
        <html>
        <body style="font-family:Arial;text-align:center;padding:40px;">
            <h1>Prediction Result</h1>
            <h2>{result}</h2>
            <p>Experimental machine learning output only.</p>
            <p>This result is not a medical diagnosis.</p>
            <a href="/">Try another image</a>
        </body>
        </html>
        """

    except (UnidentifiedImageError, OSError):
        return HTMLResponse(
            "<h3>Invalid image. Please upload a valid JPG or PNG.</h3>",
            status_code=400
        )

    except Exception as e:
        print("PREDICTION ERROR:", repr(e))
        return HTMLResponse(
            "<h3>Unable to process this image.</h3>",
            status_code=500
        )
