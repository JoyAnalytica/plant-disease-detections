
"use strict";

// ==========================================
// 1. BACKEND API URL
// ==========================================

// Local testing:
const API_BASE_URL = "http://127.0.0.1:8000";

// After deploying backend on Render, replace the URL above.
// Example:
// const API_BASE_URL = "https://your-backend-name.onrender.com";


const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ALLOWED_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp"
];

const CLASS_LABELS = {
    Healthy: "Healthy",
    Powdery: "Powdery Mildew",
    Rust: "Rust Disease"
};


// ==========================================
// 2. SELECT HTML ELEMENTS
// ==========================================

const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");

const uploadContent = document.getElementById("uploadContent");
const previewContent = document.getElementById("previewContent");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");

const chooseButton = document.getElementById("chooseButton");
const changeImageButton = document.getElementById("changeImageButton");

const analyzeButton = document.getElementById("analyzeButton");
const analyzeButtonText = document.getElementById("analyzeButtonText");
const buttonArrow = document.getElementById("buttonArrow");

const errorMessage = document.getElementById("errorMessage");

const emptyState = document.getElementById("emptyState");
const resultsContent = document.getElementById("resultsContent");

const predictedClassElement = document.getElementById("predictedClass");
const predictedNameBn = document.getElementById("predictedNameBn");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const resultStatusBn = document.getElementById("resultStatusBn");
const resultStatusIcon = document.getElementById("resultStatusIcon");

const probabilityList = document.getElementById("probabilityList");

const symptomsBn = document.getElementById("symptomsBn");
const symptomsEn = document.getElementById("symptomsEn");
const treatmentBn = document.getElementById("treatmentBn");
const treatmentEn = document.getElementById("treatmentEn");

const resetButton = document.getElementById("resetButton");


// ==========================================
// 3. STATE
// ==========================================

let selectedFile = null;
let previewUrl = null;
let isAnalyzing = false;


// ==========================================
// 4. HELPER FUNCTIONS
// ==========================================

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
}

function setLoading(loading) {
    isAnalyzing = loading;
    analyzeButton.disabled = loading || !selectedFile;

    analyzeButton.classList.toggle("is-loading", loading);

    analyzeButtonText.textContent = loading
        ? "Analyzing your leaf..."
        : "Analyze leaf with AI";

    buttonArrow.textContent = loading ? "⟳" : "→";
}

function releasePreviewUrl() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
}


// ==========================================
// 5. VALIDATE AND SELECT IMAGE
// ==========================================

function handleFile(file) {
    if (isAnalyzing) {
        return;
    }

    clearError();

    if (!file) {
        return;
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
        showError("Please choose a JPG, PNG, or WEBP image.");
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        showError("Image is too large. Please choose a file under 10 MB.");
        return;
    }

    if (file.size === 0) {
        showError("The selected image is empty. Please choose another file.");
        return;
    }

    selectedFile = file;

    releasePreviewUrl();
    previewUrl = URL.createObjectURL(file);

    imagePreview.src = previewUrl;
    imagePreview.alt = `Preview of ${file.name}`;
    fileName.textContent = file.name;

    uploadContent.hidden = true;
    previewContent.hidden = false;

    analyzeButton.disabled = false;

    // Hide old result after selecting a new image.
    emptyState.hidden = false;
    resultsContent.hidden = true;

    confidenceBar.style.width = "0%";
}

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    if (file) {
        handleFile(file);
    }
});

chooseButton.addEventListener("click", function (event) {
    event.stopPropagation();

    if (!isAnalyzing) {
        imageInput.click();
    }
});

changeImageButton.addEventListener("click", function () {
    if (!isAnalyzing) {
        imageInput.click();
    }
});

dropZone.addEventListener("click", function (event) {
    if (
        !isAnalyzing &&
        !event.target.closest("button")
    ) {
        imageInput.click();
    }
});

dropZone.addEventListener("keydown", function (event) {
    if (
        !isAnalyzing &&
        (event.key === "Enter" || event.key === " ")
    ) {
        event.preventDefault();
        imageInput.click();
    }
});


// ==========================================
// 6. DRAG AND DROP
// ==========================================

["dragenter", "dragover"].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
        event.preventDefault();
        event.stopPropagation();

        if (!isAnalyzing) {
            dropZone.classList.add("drag-over");
        }
    });
});

["dragleave", "dragend"].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
    });
});

dropZone.addEventListener("drop", function (event) {
    event.preventDefault();
    event.stopPropagation();

    dropZone.classList.remove("drag-over");

    if (isAnalyzing) {
        return;
    }

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);

        // Allow choosing the same file again later.
        imageInput.value = "";
    }
});


// ==========================================
// 7. CREATE SAFE LIST ITEMS
// ==========================================

function renderList(container, items) {
    container.replaceChildren();

    if (!Array.isArray(items)) {
        return;
    }

    items.forEach(function (item) {
        const li = document.createElement("li");
        li.textContent = String(item);
        container.appendChild(li);
    });
}


// ==========================================
// 8. DISPLAY CLASS PROBABILITIES
// ==========================================

function renderProbabilities(probabilities, predictedClass) {
    probabilityList.replaceChildren();

    const entries = Object.entries(probabilities || {});

    entries.sort(function (a, b) {
        return Number(b[1]) - Number(a[1]);
    });

    entries.forEach(function ([className, rawScore]) {
        const score = Number(rawScore);

        if (!Number.isFinite(score)) {
            return;
        }

        const safeScore = Math.max(0, Math.min(100, score));

        const row = document.createElement("div");
        row.className = "probability-row";

        if (className === predictedClass) {
            row.classList.add("is-predicted");
        }

        const label = document.createElement("div");
        label.className = "probability-label";

        const name = document.createElement("span");
        name.textContent = CLASS_LABELS[className] || className;

        const value = document.createElement("strong");
        value.textContent = `${safeScore.toFixed(2)}%`;

        label.append(name, value);

        const track = document.createElement("div");
        track.className = "probability-track";

        const fill = document.createElement("div");
        fill.className = "probability-fill";
        fill.style.width = `${safeScore}%`;

        track.appendChild(fill);
        row.append(label, track);

        probabilityList.appendChild(row);
    });
}


// ==========================================
// 9. DISPLAY API RESULT
// ==========================================

function displayResult(data) {
    const className = data.predicted_class;

    predictedClassElement.textContent =
        CLASS_LABELS[className] || className || "Unknown";

    predictedNameBn.textContent =
        data.disease_name_bn || "তথ্য পাওয়া যায়নি";

    resultStatusBn.textContent =
        data.status_bn || "";

    symptomsBn.textContent =
        data.symptoms_bn || "লক্ষণ সম্পর্কিত তথ্য পাওয়া যায়নি।";

    symptomsEn.textContent =
        data.symptoms_en || "";

    const confidence = Number(data.confidence);

    if (!Number.isFinite(confidence)) {
        throw new Error("The API returned an invalid confidence score.");
    }

    const safeConfidence = Math.max(
        0,
        Math.min(100, confidence)
    );

    confidenceValue.textContent =
        `${safeConfidence.toFixed(2)}%`;

    confidenceBar.style.width =
        `${safeConfidence}%`;

    // Change icon according to the predicted class.
    if (className === "Healthy") {
        resultStatusIcon.textContent = "✓";
    } else if (className === "Powdery") {
        resultStatusIcon.textContent = "✿";
    } else if (className === "Rust") {
        resultStatusIcon.textContent = "!";
    } else {
        resultStatusIcon.textContent = "•";
    }

    renderProbabilities(
        data.all_probabilities,
        className
    );

    renderList(
        treatmentBn,
        data.treatment_bn
    );

    renderList(
        treatmentEn,
        data.treatment_en
    );

    emptyState.hidden = true;
    resultsContent.hidden = false;

    document.getElementById("resultPanel").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// ==========================================
// 10. CALL FASTAPI /predict ENDPOINT
// ==========================================

async function analyzeLeaf() {
    if (!selectedFile || isAnalyzing) {
        return;
    }

    clearError();
    setLoading(true);

    try {
        const formData = new FormData();

        // Must match the FastAPI parameter name: file
        formData.append("file", selectedFile);

        const response = await fetch(
            `${API_BASE_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                "The server returned an unreadable response."
            );
        }

        if (!response.ok) {
            const detail = data.detail;

            throw new Error(
                typeof detail === "string"
                    ? detail
                    : `Prediction failed (HTTP ${response.status}).`
            );
        }

        if (!data.success || !data.predicted_class) {
            throw new Error(
                "The server response is missing prediction information."
            );
        }

        displayResult(data);

    } catch (error) {
        console.error("LeafCare AI error:", error);

        if (error instanceof TypeError) {
            showError(
                "Could not connect to the prediction API. " +
                "Check that your backend is running and API_BASE_URL is correct."
            );
        } else {
            showError(
                error.message || "Something went wrong. Please try again."
            );
        }
    } finally {
        setLoading(false);
    }
}

analyzeButton.addEventListener("click", analyzeLeaf);


// ==========================================
// 11. RESET / ANALYZE ANOTHER IMAGE
// ==========================================

resetButton.addEventListener("click", function () {
    if (isAnalyzing) {
        return;
    }

    selectedFile = null;
    imageInput.value = "";

    releasePreviewUrl();

    imagePreview.removeAttribute("src");
    fileName.textContent = "";

    uploadContent.hidden = false;
    previewContent.hidden = true;

    emptyState.hidden = false;
    resultsContent.hidden = true;

    confidenceBar.style.width = "0%";

    probabilityList.replaceChildren();
    treatmentBn.replaceChildren();
    treatmentEn.replaceChildren();

    clearError();

    analyzeButton.disabled = true;

    document.getElementById("detect").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
});


// ==========================================
// 12. CLEAN UP PREVIEW WHEN LEAVING
// ==========================================

window.addEventListener("beforeunload", function () {
    releasePreviewUrl();
});


// ==========================================
// 13. READY
// ==========================================

console.log("LeafCare AI frontend is ready.");
console.log("Prediction API:", `${API_BASE_URL}/predict`);
