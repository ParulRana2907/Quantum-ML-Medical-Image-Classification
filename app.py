
import logging
import sys
import io
import base64
import html

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import joblib


# ==================================================
# 1. CONFIGURATION AND LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("quantum_ml")


# ==================================================
# 2. INITIALIZE FASTAPI
# ==================================================

app = FastAPI(
    title="Quantum ML Pneumonia Classification",
    description=(
        "Experimental chest X-ray classification "
        "using classical and quantum machine learning research."
    ),
    version="1.0"
)


# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

bundle = joblib.load("model/best_model_pipeline.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
pca = bundle["pca"]

IMG_SIZE = bundle["image_size"]
class_names = bundle["class_names"]

logger.info(
    "Model loaded successfully: %s",
    type(model).__name__
)


# ==================================================
# 4. PROJECT INFORMATION
# ==================================================

PROJECT_TITLE = (
    "Quantum Machine Learning for "
    "Medical Image Classification"
)

PROJECT_SUBTITLE = (
    "An experimental study of classical and quantum "
    "machine learning techniques for chest X-ray classification."
)

# These values are from the project's recorded experiments.
# They describe the evaluated models, not live predictions.

SVM_METRICS = {
    "accuracy": 90.05,
    "precision": 92.26,
    "recall": 89.86,
    "f1": 91.04
}

RF_METRICS = {
    "accuracy": 88.74
}

QSVC_METRICS = {
    "accuracy": 55.00,
    "training_samples": 80,
    "validation_samples": 40,
    "qubits": 6,
    "repetitions": 2
}


# ==================================================
# 5. GLOBAL WEBSITE STYLING
# ==================================================

STYLE = """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap'
);

:root {
    --bg: #07111f;
    --panel: #101f32;
    --panel2: #142941;
    --cyan: #48d9e8;
    --blue: #438cff;
    --text: #f3f8ff;
    --muted: #9aadc2;
    --border: rgba(157,190,220,.15);
    --green: #54d6a0;
    --orange: #ffbd69;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    scroll-behavior: smooth;
}

body {
    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(57,136,255,.16),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 25%,
            rgba(72,217,232,.09),
            transparent 30%
        ),
        var(--bg);

    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
}

a {
    color: inherit;
    text-decoration: none;
}

.container {
    width: min(1160px, 91%);
    margin: auto;
}


/* NAVIGATION */

.navbar {
    position: sticky;
    top: 0;
    z-index: 10;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 17px 5%;

    background: rgba(7,17,31,.91);
    backdrop-filter: blur(15px);

    border-bottom: 1px solid var(--border);
}

.brand {
    display: flex;
    align-items: center;
    gap: 11px;

    font-family: 'Manrope', sans-serif;
    font-size: 17px;
    font-weight: 800;
}

.brand-icon {
    height: 42px;
    width: 42px;

    display: grid;
    place-items: center;

    border-radius: 13px;

    background: linear-gradient(
        135deg,
        var(--cyan),
        #72a9ff
    );

    color: #07111f;
    font-size: 22px;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 24px;
}

.nav-links a {
    color: var(--muted);
    font-size: 12px;
    transition: .2s;
}

.nav-links a:hover {
    color: var(--cyan);
}

.nav-tag {
    color: var(--cyan);

    border: 1px solid rgba(72,217,232,.3);
    background: rgba(72,217,232,.07);

    border-radius: 30px;
    padding: 8px 13px;

    font-size: 11px;
    font-weight: 800;
}


/* HERO */

.hero {
    padding: 70px 10px 36px;
    text-align: center;
}

.eyebrow {
    color: var(--cyan);

    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;

    font-size: clamp(32px, 5vw, 53px);
    line-height: 1.15;

    letter-spacing: -1.8px;
    margin: 18px 0;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #48d9e8,
        #75aaff
    );

    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.subtitle {
    color: var(--muted);

    font-size: 15px;
    line-height: 1.8;

    max-width: 710px;
    margin: auto;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 9px;

    margin-top: 23px;
}

.hero-tag {
    color: #b9d7ed;

    border: 1px solid var(--border);
    background: rgba(255,255,255,.035);

    padding: 8px 12px;
    border-radius: 30px;

    font-size: 11px;
}


/* SECTION */

.section {
    margin: 35px 0 48px;
    scroll-margin-top: 95px;
}

.section-heading {
    margin-bottom: 20px;
}

.section-heading h2 {
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    margin-bottom: 7px;
}

.section-heading p {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
}


/* CARDS */

.grid {
    display: grid;
    gap: 17px;
}

.grid-2 {
    grid-template-columns: repeat(2, 1fr);
}

.grid-3 {
    grid-template-columns: repeat(3, 1fr);
}

.grid-4 {
    grid-template-columns: repeat(4, 1fr);
}

.card {
    background: linear-gradient(
        145deg,
        rgba(20,40,64,.96),
        rgba(13,29,47,.96)
    );

    border: 1px solid var(--border);
    border-radius: 19px;

    padding: 23px;

    box-shadow: 0 15px 45px rgba(0,0,0,.13);

    transition: transform .25s, border-color .25s;
}

.card:hover {
    transform: translateY(-3px);
    border-color: rgba(72,217,232,.32);
}

.card-icon {
    font-size: 25px;
    margin-bottom: 13px;
}

.card h3 {
    font-size: 15px;
    margin-bottom: 9px;
}

.card p {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.8;
}


/* STATUS */

.status-row {
    display: flex;
    align-items: center;
    gap: 9px;
}

.status-dot {
    height: 9px;
    width: 9px;

    border-radius: 50%;
    background: var(--green);

    box-shadow: 0 0 12px rgba(84,214,160,.6);
}

.status-text {
    color: var(--green);
    font-size: 12px;
    font-weight: 700;
}


/* METRIC CARDS */

.metric-card {
    position: relative;
    overflow: hidden;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
}

.metric-value {
    color: var(--cyan);

    font-family: 'Manrope', sans-serif;
    font-size: 32px;
    font-weight: 800;

    margin: 10px 0 5px;
}

.metric-note {
    color: var(--muted);
    font-size: 10px;
}


/* UPLOAD AREA */

.upload-grid {
    display: grid;
    grid-template-columns: 1.1fr .9fr;
    gap: 22px;
}

.panel {
    background: linear-gradient(
        145deg,
        rgba(20,40,64,.96),
        rgba(13,29,47,.96)
    );

    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 28px;
}

.panel-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.panel-heading h2 {
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
}

.step-number {
    height: 34px;
    width: 34px;

    display: grid;
    place-items: center;

    border-radius: 10px;

    background: rgba(72,217,232,.12);
    color: var(--cyan);

    font-weight: 800;
}

.panel-description {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.8;
    margin-bottom: 21px;
}

.drop-zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    min-height: 220px;

    border: 2px dashed rgba(72,217,232,.4);
    background: rgba(72,217,232,.035);

    border-radius: 17px;
    cursor: pointer;

    padding: 25px;
    text-align: center;

    transition: .25s;
}

.drop-zone:hover,
.drop-zone.dragging {
    background: rgba(72,217,232,.09);
    border-color: var(--cyan);
}

.upload-icon {
    height: 58px;
    width: 58px;

    display: grid;
    place-items: center;

    border-radius: 17px;

    background: rgba(72,217,232,.12);
    color: var(--cyan);

    font-size: 27px;
    margin-bottom: 15px;
}

.drop-zone strong {
    font-size: 15px;
}

.drop-zone span {
    color: var(--muted);
    font-size: 11px;
    margin-top: 9px;
}

#fileInput {
    display: none;
}

#fileName {
    color: var(--cyan);
    font-size: 11px;
    margin-top: 12px;
    word-break: break-all;
}

.preview-container {
    display: none;
    margin-top: 17px;
    text-align: center;
}

.preview-container img {
    max-height: 210px;
    max-width: 100%;

    object-fit: contain;

    border-radius: 12px;
    border: 1px solid var(--border);

    background: white;
    padding: 5px;
}

.preview-caption {
    color: var(--muted);
    font-size: 11px;
    margin-top: 8px;
}


/* BUTTON */

.primary-btn {
    width: 100%;

    border: 0;
    padding: 16px;

    margin-top: 21px;

    border-radius: 12px;

    background: linear-gradient(
        100deg,
        #35c9dc,
        #438cff
    );

    color: #07111f;

    font-family: inherit;
    font-size: 14px;
    font-weight: 800;

    cursor: pointer;
    transition: .25s;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 25px rgba(57,136,255,.25);
}

.primary-btn:disabled {
    opacity: .65;
    cursor: wait;
}


/* WORKFLOW */

.workflow-item {
    text-align: center;
    padding: 20px 13px;
}

.workflow-number {
    color: var(--cyan);
    font-size: 11px;
    font-weight: 800;
}

.workflow-item h3 {
    font-size: 13px;
    margin: 11px 0 7px;
}

.workflow-item p {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.7;
}


/* MODEL COMPARISON */

.model-name {
    font-family: 'Manrope', sans-serif;
    font-size: 18px;
    margin-bottom: 9px;
}

.model-score {
    color: var(--cyan);
    font-size: 29px;
    font-weight: 800;
    margin: 12px 0;
}

.model-badge {
    display: inline-block;

    color: var(--cyan);
    background: rgba(72,217,232,.08);

    border: 1px solid rgba(72,217,232,.23);
    border-radius: 30px;

    padding: 6px 10px;
    font-size: 10px;
    font-weight: 800;
}

.model-details {
    margin-top: 14px;
    border-top: 1px solid var(--border);
    padding-top: 13px;
}

.model-details div {
    display: flex;
    justify-content: space-between;
    gap: 10px;

    padding: 6px 0;

    color: var(--muted);
    font-size: 11px;
}

.model-details strong {
    color: var(--text);
}


/* PROGRESS BAR */

.bar-row {
    margin: 15px 0;
}

.bar-label {
    display: flex;
    justify-content: space-between;

    color: var(--muted);
    font-size: 11px;

    margin-bottom: 7px;
}

.bar-track {
    height: 9px;
    border-radius: 20px;
    background: rgba(255,255,255,.08);
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(
        90deg,
        #35c9dc,
        #438cff
    );
}


/* TECH STACK */

.tech-pill {
    display: inline-block;

    color: #c8e2f5;
    background: rgba(255,255,255,.045);

    border: 1px solid var(--border);
    border-radius: 9px;

    padding: 10px 13px;
    margin: 5px;

    font-size: 11px;
    font-weight: 700;
}


/* DISCLAIMER */

.disclaimer {
    border: 1px solid rgba(255,190,90,.22);
    background: rgba(255,190,90,.055);

    color: #e6c99b;

    padding: 17px 19px;
    border-radius: 13px;

    font-size: 11px;
    line-height: 1.8;

    margin: 28px 0 40px;
}


/* RESULT PAGE */

.result-card {
    max-width: 760px;
    margin: 35px auto 45px;
    text-align: center;
}

.result-label {
    display: inline-block;

    border: 1px solid rgba(72,217,232,.3);
    color: var(--cyan);
    background: rgba(72,217,232,.08);

    border-radius: 30px;
    padding: 8px 16px;

    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

.result-value {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(28px, 5vw, 42px);
    margin: 16px 0;
}

.result-image {
    max-width: 100%;
    max-height: 260px;

    object-fit: contain;

    padding: 7px;
    background: white;

    border-radius: 14px;
    margin: 20px auto;
}

.confidence-number {
    color: var(--cyan);
    font-size: 28px;
    font-weight: 800;
}

.confidence-track {
    width: 100%;
    height: 11px;

    border-radius: 20px;
    background: rgba(255,255,255,.1);

    overflow: hidden;
    margin: 14px 0;
}

.confidence-fill {
    height: 100%;
    border-radius: 20px;

    background: linear-gradient(
        90deg,
        #35c9dc,
        #438cff
    );
}

.back-link {
    display: inline-block;

    margin-top: 23px;

    color: #07111f;
    background: linear-gradient(
        100deg,
        #35c9dc,
        #438cff
    );

    padding: 13px 23px;
    border-radius: 11px;

    font-weight: 800;
    font-size: 12px;
}


/* FOOTER */

.footer {
    border-top: 1px solid var(--border);

    text-align: center;

    padding: 25px 15px;

    color: #71869d;
    font-size: 11px;
    line-height: 1.8;
}


/* RESPONSIVE */

@media(max-width: 850px) {

    .grid-4 {
        grid-template-columns: repeat(2, 1fr);
    }

    .grid-3 {
        grid-template-columns: repeat(2, 1fr);
    }

    .upload-grid {
        grid-template-columns: 1fr;
    }

    .nav-links {
        display: none;
    }

}

@media(max-width: 520px) {

    .grid-2,
    .grid-3,
    .grid-4 {
        grid-template-columns: 1fr;
    }

    .navbar {
        padding: 14px 5%;
    }

    .brand {
        font-size: 14px;
    }

    .hero {
        padding-top: 45px;
    }

    .panel {
        padding: 20px;
    }

}

</style>
"""


# ==================================================
# 6. SHARED WEBSITE COMPONENTS
# ==================================================

def website_header():

    return """
    <header class="navbar">

        <a href="/" class="brand">

            <div class="brand-icon">✚</div>

            <span>
                QUANTUM<span style="color:#48d9e8">.MED AI</span>
            </span>

        </a>


        <nav class="nav-links">

            <a href="#overview">Overview</a>

            <a href="#methodology">Methodology</a>

            <a href="#performance">Performance</a>

            <a href="#upload">Analyze</a>

        </nav>


        <div class="nav-tag">
            AI RESEARCH PROJECT
        </div>

    </header>
    """


def website_footer():

    return """
    <footer class="footer">

        <strong style="color:#48d9e8">
            QUANTUM.MED AI
        </strong>

        <br>

        Quantum Machine Learning for Medical Image Classification

        <br><br>

        Developed using Python · FastAPI · Scikit-learn · PCA

        <br>

        Experimental academic research project

    </footer>
    """


def page_shell(content):

    return (
        """
        <!DOCTYPE html>
        <html lang="en">

        <head>

            <meta charset="UTF-8">

            <meta name="viewport"
                  content="width=device-width, initial-scale=1.0">

            <meta name="description"
                  content="Experimental chest X-ray image classification research dashboard">

            <title>Quantum.Med AI</title>

            __STYLE__

        </head>

        <body>

            __HEADER__

            __CONTENT__

            __FOOTER__

        </body>

        </html>
        """
        .replace("__STYLE__", STYLE)
        .replace("__HEADER__", website_header())
        .replace("__CONTENT__", content)
        .replace("__FOOTER__", website_footer())
    )


# ==================================================
# 7. HOME PAGE
# ==================================================

@app.get("/", response_class=HTMLResponse)
def home():

    content = """

    <main class="container">


        <!-- HERO SECTION -->

        <section class="hero">

            <div class="eyebrow">
                INTELLIGENT MEDICAL IMAGE RESEARCH
            </div>


            <h1>

                Quantum Machine Learning

                <br>

                <span class="gradient-text">
                    For Medical Image Classification
                </span>

            </h1>


            <p class="subtitle">

                An experimental research project exploring
                classical and quantum machine learning
                approaches for chest X-ray image classification.

                Combining image preprocessing, dimensionality
                reduction and intelligent classification.

            </p>


            <div class="hero-tags">

                <span class="hero-tag">✦ Machine Learning</span>

                <span class="hero-tag">⚛ Quantum ML Research</span>

                <span class="hero-tag">🩻 Chest X-Ray Images</span>

                <span class="hero-tag">📊 Model Evaluation</span>

            </div>

        </section>



        <!-- PROJECT OVERVIEW -->

        <section class="section" id="overview">

            <div class="section-heading">

                <h2>Project Overview</h2>

                <p>
                    An overview of the research implementation
                    and the currently loaded prediction model.
                </p>

            </div>


            <div class="grid grid-3">


                <div class="card">

                    <div class="card-icon">🧠</div>

                    <h3>Research Objective</h3>

                    <p>
                        Explore machine learning and quantum
                        machine learning approaches for
                        classification of chest X-ray images.
                    </p>

                </div>


                <div class="card">

                    <div class="card-icon">🩻</div>

                    <h3>Image Classification</h3>

                    <p>
                        The dataset contains two image
                        categories: NORMAL and PNEUMONIA.
                    </p>

                </div>


                <div class="card">

                    <div class="card-icon">⚙️</div>

                    <h3>Prediction Pipeline</h3>

                    <p>
                        The deployed application uses a
                        saved classical machine learning
                        pipeline for image prediction.
                    </p>

                </div>


            </div>

        </section>



        <!-- SYSTEM STATUS -->

        <section class="section">

            <div class="section-heading">

                <h2>System Information</h2>

                <p>
                    Information about the deployed
                    classification application.
                </p>

            </div>


            <div class="grid grid-4">


                <div class="card">

                    <div class="status-row">

                        <span class="status-dot"></span>

                        <span class="status-text">
                            MODEL LOADED
                        </span>

                    </div>

                    <h3 style="margin-top:16px">
                        Prediction Model
                    </h3>

                    <p>
                        __MODEL_NAME__
                    </p>

                </div>


                <div class="card">

                    <div class="card-icon">🖼️</div>

                    <h3>Input Resolution</h3>

                    <p>
                        __IMG_SIZE__ × __IMG_SIZE__ pixels
                    </p>

                </div>


                <div class="card">

                    <div class="card-icon">📉</div>

                    <h3>PCA Components</h3>

                    <p>
                        __PCA_COMPONENTS__ transformed features
                    </p>

                </div>


                <div class="card">

                    <div class="card-icon">🔢</div>

                    <h3>Input Features</h3>

                    <p>
                        __INPUT_FEATURES__ pixel features before PCA
                    </p>

                </div>


            </div>

        </section>



        <!-- METHODOLOGY -->

        <section class="section" id="methodology">

            <div class="section-heading">

                <h2>Research Methodology</h2>

                <p>
                    The major stages involved in the
                    implemented classical prediction pipeline.
                </p>

            </div>


            <div class="grid grid-3">


                <div class="card workflow-item">

                    <div class="workflow-number">
                        STEP 01
                    </div>

                    <div class="card-icon">📂</div>

                    <h3>Dataset Collection</h3>

                    <p>
                        Chest X-ray images belonging to
                        NORMAL and PNEUMONIA categories.
                    </p>

                </div>


                <div class="card workflow-item">

                    <div class="workflow-number">
                        STEP 02
                    </div>

                    <div class="card-icon">🖼️</div>

                    <h3>Image Preprocessing</h3>

                    <p>
                        Grayscale conversion, resizing to
                        64 × 64 and pixel normalization.
                    </p>

                </div>


                <div class="card workflow-item">

                    <div class="workflow-number">
                        STEP 03
                    </div>

                    <div class="card-icon">📐</div>

                    <h3>Feature Scaling</h3>

                    <p>
                        StandardScaler transforms
                        the flattened image features.
                    </p>

                </div>


                <div class="card workflow-item">

                    <div class="workflow-number">
                        STEP 04
                    </div>

                    <div class="card-icon">📉</div>

                    <h3>Dimensionality Reduction</h3>

                    <p>
                        PCA reduces the feature representation
                        to six principal components.
                    </p>

                </div>


                <div class="card workflow-item">

                    <div class="workflow-number">
                        STEP 05
                    </div>

                    <div class="card-icon">🤖</div>

                    <h3>Model Classification</h3>

                    <p>
                        The selected SVM RBF model
                        classifies the transformed features.
                    </p>

                </div>


                <div class="card workflow-item">

                    <div class="workflow-number">
                        STEP 06
                    </div>

                    <div class="card-icon">📊</div>

                    <h3>Prediction Output</h3>

                    <p>
                        The application displays the
                        predicted class and model score.
                    </p>

                </div>


            </div>

        </section>



        <!-- CLASSICAL MODEL PERFORMANCE -->

        <section class="section" id="performance">

            <div class="section-heading">

                <h2>Classical ML Performance</h2>

                <p>
                    Recorded evaluation metrics for
                    the selected classical SVM model.
                </p>

            </div>


            <div class="grid grid-4">


                <div class="card metric-card">

                    <div class="metric-label">
                        Accuracy
                    </div>

                    <div class="metric-value">
                        90.05%
                    </div>

                    <div class="metric-note">
                        Correct predictions
                    </div>

                </div>


                <div class="card metric-card">

                    <div class="metric-label">
                        Precision
                    </div>

                    <div class="metric-value">
                        92.26%
                    </div>

                    <div class="metric-note">
                        Positive predictive performance
                    </div>

                </div>


                <div class="card metric-card">

                    <div class="metric-label">
                        Recall
                    </div>

                    <div class="metric-value">
                        89.86%
                    </div>

                    <div class="metric-note">
                        Positive cases identified
                    </div>

                </div>


                <div class="card metric-card">

                    <div class="metric-label">
                        F1 Score
                    </div>

                    <div class="metric-value">
                        91.04%
                    </div>

                    <div class="metric-note">
                        Harmonic mean of precision and recall
                    </div>

                </div>


            </div>


            <div class="card" style="margin-top:17px">

                <h3>Selected Classical Model</h3>

                <div class="model-score">
                    SVM — RBF Kernel
                </div>

                <p>
                    The selected classical model uses an
                    RBF kernel with C = 1.
                </p>

                <div class="model-details">

                    <div>
                        <span>Accuracy</span>
                        <strong>90.05%</strong>
                    </div>

                    <div>
                        <span>Precision</span>
                        <strong>92.26%</strong>
                    </div>

                    <div>
                        <span>Recall</span>
                        <strong>89.86%</strong>
                    </div>

                    <div>
                        <span>F1 Score</span>
                        <strong>91.04%</strong>
                    </div>

                </div>

            </div>

        </section>



        <!-- MODEL COMPARISON -->

        <section class="section">

            <div class="section-heading">

                <h2>Classical vs Quantum Experiments</h2>

                <p>
                    A separate view of the recorded
                    experimental results from the project.
                </p>

            </div>


            <div class="grid grid-3">


                <div class="card">

                    <span class="model-badge">
                        CLASSICAL ML
                    </span>

                    <div class="model-name" style="margin-top:15px">
                        SVM RBF
                    </div>

                    <div class="model-score">
                        90.05%
                    </div>

                    <p>
                        Selected classical model
                        using six PCA components.
                    </p>

                    <div class="model-details">

                        <div>
                            <span>Kernel</span>
                            <strong>RBF</strong>
                        </div>

                        <div>
                            <span>C parameter</span>
                            <strong>1</strong>
                        </div>

                        <div>
                            <span>F1 Score</span>
                            <strong>91.04%</strong>
                        </div>

                    </div>

                </div>


                <div class="card">

                    <span class="model-badge">
                        CLASSICAL ML
                    </span>

                    <div class="model-name" style="margin-top:15px">
                        Random Forest
                    </div>

                    <div class="model-score">
                        88.74%
                    </div>

                    <p>
                        Classical ensemble model
                        used for comparison.
                    </p>

                    <div class="model-details">

                        <div>
                            <span>Estimators</span>
                            <strong>100</strong>
                        </div>

                        <div>
                            <span>Maximum depth</span>
                            <strong>15</strong>
                        </div>

                        <div>
                            <span>Accuracy</span>
                            <strong>88.74%</strong>
                        </div>

                    </div>

                </div>


                <div class="card">

                    <span class="model-badge">
                        QUANTUM ML EXPERIMENT
                    </span>

                    <div class="model-name" style="margin-top:15px">
                        QSVC
                    </div>

                    <div class="model-score">
                        55.00%
                    </div>

                    <p>
                        Separate experimental quantum
                        classification evaluation.
                    </p>

                    <div class="model-details">

                        <div>
                            <span>Qubits</span>
                            <strong>6</strong>
                        </div>

                        <div>
                            <span>Feature map</span>
                            <strong>ZZFeatureMap</strong>
                        </div>

                        <div>
                            <span>Repetitions</span>
                            <strong>2</strong>
                        </div>

                        <div>
                            <span>Training samples</span>
                            <strong>80</strong>
                        </div>

                        <div>
                            <span>Validation samples</span>
                            <strong>40</strong>
                        </div>

                    </div>

                </div>


            </div>


            <div class="disclaimer">

                <strong>Research comparison note:</strong>

                The quantum experiment used a smaller sample
                of 80 training and 40 validation images.
                Its result is not directly comparable with
                the classical model evaluation because the
                evaluation sample sizes and experimental
                setups differ.

                The quantum experiment is displayed as a
                separate research result and is not used
                by the website prediction endpoint.

            </div>

        </section>



        <!-- TECHNOLOGY STACK -->

        <section class="section">

            <div class="section-heading">

                <h2>Technology Stack</h2>

                <p>
                    Main tools and technologies used
                    in the project implementation.
                </p>

            </div>


            <div class="card">

                <span class="tech-pill">Python</span>

                <span class="tech-pill">FastAPI</span>

                <span class="tech-pill">NumPy</span>

                <span class="tech-pill">Pillow</span>

                <span class="tech-pill">Scikit-learn</span>

                <span class="tech-pill">PCA</span>

                <span class="tech-pill">SVM</span>

                <span class="tech-pill">Qiskit</span>

                <span class="tech-pill">QSVC</span>

                <span class="tech-pill">MLflow</span>

                <span class="tech-pill">Docker</span>

                <span class="tech-pill">GitHub Actions</span>

                <span class="tech-pill">Render</span>

            </div>

        </section>



        <!-- UPLOAD SECTION -->

        <section class="section" id="upload">

            <div class="section-heading">

                <h2>Try the Classification System</h2>

                <p>
                    Upload a chest X-ray image to obtain
                    an experimental output from the
                    deployed classical model.
                </p>

            </div>


            <div class="upload-grid">


                <div class="panel">

                    <div class="panel-heading">

                        <div class="step-number">01</div>

                        <h2>Upload X-Ray Image</h2>

                    </div>


                    <p class="panel-description">

                        Select a JPG or PNG image.
                        You can also drag and drop the file
                        into the upload area.

                    </p>


                    <form
                        id="uploadForm"
                        action="/predict-page"
                        method="post"
                        enctype="multipart/form-data"
                    >


                        <label
                            class="drop-zone"
                            id="dropZone"
                            for="fileInput"
                        >

                            <div class="upload-icon">
                                ↑
                            </div>


                            <strong>
                                Drag & Drop Your Image
                            </strong>


                            <span>
                                or click to browse files
                            </span>


                            <span>
                                Supported formats: JPG, JPEG, PNG
                            </span>


                            <input
                                type="file"
                                id="fileInput"
                                name="file"
                                accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                                required
                            >


                            <div id="fileName"></div>

                        </label>



                        <div
                            class="preview-container"
                            id="previewContainer"
                        >

                            <img
                                id="imagePreview"
                                alt="Selected X-ray preview"
                            >

                            <div class="preview-caption">
                                Selected image preview
                            </div>

                        </div>



                        <button
                            type="submit"
                            class="primary-btn"
                            id="submitBtn"
                        >

                            ✦ &nbsp; Analyze X-Ray Image

                        </button>


                    </form>

                </div>



                <div class="panel">

                    <div class="panel-heading">

                        <div class="step-number">02</div>

                        <h2>Analysis Information</h2>

                    </div>


                    <p class="panel-description">

                        What happens after you upload an image?

                    </p>


                    <div class="card" style="margin-bottom:13px">

                        <div class="card-icon">🖼️</div>

                        <h3>Image Processing</h3>

                        <p>
                            The uploaded image is converted
                            into grayscale and resized to
                            the model's expected resolution.
                        </p>

                    </div>


                    <div class="card" style="margin-bottom:13px">

                        <div class="card-icon">📉</div>

                        <h3>Feature Transformation</h3>

                        <p>
                            The saved StandardScaler and
                            PCA transformations process
                            the image features.
                        </p>

                    </div>


                    <div class="card">

                        <div class="card-icon">🤖</div>

                        <h3>Model Output</h3>

                        <p>
                            The saved classical classifier
                            generates the predicted class
                            and model confidence score.
                        </p>

                    </div>

                </div>


            </div>

        </section>



        <!-- LIMITATIONS -->

        <section class="section">

            <div class="section-heading">

                <h2>Research Limitations & Future Scope</h2>

                <p>
                    Current limitations and possible
                    directions for future development.
                </p>

            </div>


            <div class="grid grid-2">


                <div class="card">

                    <div class="card-icon">⚠️</div>

                    <h3>Current Limitations</h3>

                    <p>
                        Experimental dataset and model results
                        may not generalize to other clinical
                        environments. The quantum experiment
                        used a smaller evaluation sample.
                        The system has not been clinically validated.
                    </p>

                </div>


                <div class="card">

                    <div class="card-icon">🚀</div>

                    <h3>Future Scope</h3>

                    <p>
                        Future work may investigate larger
                        and more diverse datasets, improved
                        quantum feature representations,
                        additional model comparisons and
                        rigorous external validation.
                    </p>

                </div>


            </div>

        </section>



        <!-- DISCLAIMER -->

        <div class="disclaimer">

            <strong>⚠ Academic Research Disclaimer:</strong>

            This application is developed exclusively
            for educational and experimental purposes.

            Its output is not a medical diagnosis,
            does not establish the presence or absence
            of disease, and must not be used for
            treatment or clinical decisions.

            The displayed model confidence score is
            not a clinically validated probability.

            Consult a qualified healthcare professional
            for medical concerns.

        </div>


    </main>


    <!-- UPLOAD INTERACTION SCRIPT -->

    <script>

    const input = document.getElementById("fileInput");

    const dropZone = document.getElementById("dropZone");

    const fileName = document.getElementById("fileName");

    const preview = document.getElementById("imagePreview");

    const previewContainer =
        document.getElementById("previewContainer");

    const form = document.getElementById("uploadForm");

    const submitBtn = document.getElementById("submitBtn");


    function showFile(file) {

        if (!file) {
            return;
        }

        const allowedTypes = [
            "image/jpeg",
            "image/png"
        ];


        if (!allowedTypes.includes(file.type)) {

            alert("Please select a JPG or PNG image.");

            input.value = "";

            fileName.textContent = "";

            previewContainer.style.display = "none";

            preview.removeAttribute("src");

            return;
        }


        fileName.textContent = "Selected: " + file.name;


        const reader = new FileReader();


        reader.onload = function(event) {

            preview.src = event.target.result;

            previewContainer.style.display = "block";

        };


        reader.readAsDataURL(file);

    }


    // NORMAL FILE SELECTION

    input.addEventListener("change", function() {

        showFile(this.files[0]);

    });


    // DRAG AND DROP

    ["dragenter", "dragover"].forEach(function(eventName) {

        dropZone.addEventListener(eventName, function(event) {

            event.preventDefault();

            dropZone.classList.add("dragging");

        });

    });


    ["dragleave", "drop"].forEach(function(eventName) {

        dropZone.addEventListener(eventName, function(event) {

            event.preventDefault();

            dropZone.classList.remove("dragging");

        });

    });


    dropZone.addEventListener("drop", function(event) {

        const files = event.dataTransfer.files;

        if (files.length > 0) {

            input.files = files;

            showFile(files[0]);

        }

    });


    // FORM SUBMISSION

    form.addEventListener("submit", function(event) {

        if (!input.files.length) {

            event.preventDefault();

            alert("Please select an X-ray image first.");

            return;

        }


        submitBtn.disabled = true;

        submitBtn.textContent = "Processing Image...";

    });

    </script>

    """

    content = (
        content
        .replace("__MODEL_NAME__", html.escape(type(model).__name__))
        .replace("__IMG_SIZE__", str(IMG_SIZE))
        .replace(
            "__PCA_COMPONENTS__",
            str(pca.n_components_)
        )
        .replace(
            "__INPUT_FEATURES__",
            str(IMG_SIZE * IMG_SIZE)
        )
    )

    return HTMLResponse(page_shell(content))


# ==================================================
# 8. HEALTH CHECK
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": type(model).__name__
    }


# ==================================================
# 9. IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = image.convert("L")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    flattened = image_array.reshape(
        1, -1
    )

    scaled = scaler.transform(flattened)

    reduced = pca.transform(scaled)

    return reduced


# ==================================================
# 10. PREDICTION LOGGING
# ==================================================

def log_prediction(
    endpoint,
    prediction,
    confidence,
    file_size,
    content_type
):

    logger.info(
        "PREDICTION_SUCCESS | endpoint=%s | prediction=%s "
        "| confidence=%.2f%% | image_size_bytes=%d "
        "| content_type=%s",

        endpoint,
        prediction,
        confidence,
        file_size,
        content_type
    )


# ==================================================
# 11. PREDICTION API
# ==================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:

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


        prediction = int(
            model.predict(processed)[0]
        )


        probabilities = model.predict_proba(
            processed
        )[0]


        confidence = float(
            np.max(probabilities)
        )


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
            "confidence": round(
                confidence * 100,
                2
            )
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

        logger.exception(
            "PREDICTION_ERROR | endpoint=/predict"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process this image"
        )


# ==================================================
# 12. BROWSER PREDICTION RESULT PAGE
# ==================================================

@app.post(
    "/predict-page",
    response_class=HTMLResponse
)
async def predict_page(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:

        logger.warning(
            "INVALID_FILE_TYPE | endpoint=/predict-page | content_type=%s",
            file.content_type
        )

        return HTMLResponse(
            "<h3>Please upload a JPG or PNG image.</h3>",
            status_code=415
        )


    contents = await file.read()


    try:

        processed = preprocess_image(contents)


        prediction = int(
            model.predict(processed)[0]
        )


        result = str(
            class_names[prediction]
        )


        probabilities = model.predict_proba(
            processed
        )[0]


        confidence = float(
            np.max(probabilities)
        )


        log_prediction(
            endpoint="/predict-page",
            prediction=result,
            confidence=confidence * 100,
            file_size=len(contents),
            content_type=file.content_type
        )


        safe_result = html.escape(result)


        confidence_percent = max(
            0.0,
            min(100.0, confidence * 100.0)
        )


        encoded_image = base64.b64encode(
            contents
        ).decode("ascii")


        result_content = """

        <main class="container">


            <section class="hero">

                <div class="eyebrow">
                    IMAGE ANALYSIS COMPLETE
                </div>


                <h1>

                    Prediction

                    <span class="gradient-text">
                        Result
                    </span>

                </h1>


                <p class="subtitle">

                    The saved classical machine learning
                    model has generated an experimental
                    classification output.

                </p>

            </section>



            <section class="card result-card">


                <div class="result-label">
                    CLASSICAL MODEL OUTPUT
                </div>


                <h2 class="result-value">
                    __PREDICTION__
                </h2>


                <p class="panel-description">

                    Predicted class generated by the
                    trained classification pipeline.

                </p>



                <img

                    class="result-image"

                    src="data:__MIME__;base64,__IMAGE_DATA__"

                    alt="Uploaded X-ray image"

                >



                <h3 style="font-size:14px;margin:15px 0">

                    Model Confidence Score

                </h3>


                <div class="confidence-number">
                    __CONFIDENCE__%
                </div>


                <div class="confidence-track">

                    <div
                        class="confidence-fill"
                        style="width:__CONFIDENCE__%"
                    >
                    </div>

                </div>


                <p class="panel-description">

                    This score represents the output
                    of the model's predict_proba method.
                    It is not a clinically validated
                    probability.

                </p>


                <a href="/" class="back-link">

                    ← Return to Dashboard

                </a>


            </section>



            <div class="disclaimer">

                <strong>⚠ Important Research Disclaimer:</strong>

                This is an experimental academic project.
                The output is not a medical diagnosis and
                must not be used for treatment or clinical
                decisions.

                Consult a qualified healthcare professional
                for medical concerns.

            </div>


        </main>

        """


        result_content = (

            result_content

            .replace(
                "__PREDICTION__",
                safe_result
            )

            .replace(
                "__MIME__",
                file.content_type
            )

            .replace(
                "__IMAGE_DATA__",
                encoded_image
            )

            .replace(
                "__CONFIDENCE__",
                f"{confidence_percent:.2f}"
            )

        )


        return HTMLResponse(
            page_shell(result_content)
        )


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

        logger.exception(
            "PREDICTION_ERROR | endpoint=/predict-page"
        )

        return HTMLResponse(
            "<h3>Unable to process this image.</h3>",
            status_code=500
        )
