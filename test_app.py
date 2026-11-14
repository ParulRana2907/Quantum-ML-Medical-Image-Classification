
from fastapi.testclient import TestClient
from PIL import Image
import io

from app import app

client = TestClient(app)


# Test 1: Home page
def test_home_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "Chest X-Ray Classification" in response.text


# Test 2: Health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test 3: Valid image prediction
def test_valid_image_prediction():
    image = Image.new("RGB", (100, 100), color="white")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                image_bytes,
                "image/png"
            )
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [
        "NORMAL",
        "PNEUMONIA"
    ]
    assert "confidence" in response.json()


# Test 4: Missing file validation
def test_missing_file():
    response = client.post("/predict")

    # FastAPI should reject a missing required file
    assert response.status_code == 422


# Test 5: Invalid file type
def test_invalid_file_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "document.pdf",
                b"sample content",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 415