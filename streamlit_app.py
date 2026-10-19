
import io
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# =====================================
# 1. PAGE SETTINGS
# =====================================
st.set_page_config(
    page_title="LeafCare AI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 LeafCare AI")
st.subheader("Plant Disease Detection using CNN")
st.write("পাতার ছবি আপলোড করে সম্ভাব্য রোগ শনাক্ত করো।")


# =====================================
# 2. MODEL DOWNLOAD AND LOADING
# =====================================
MODEL_PATH = Path("plant_disease_cnn.keras")
FILE_ID = "1Vaq6fmPjsGWgtab3YetketrztJDmiP4L"


@st.cache_resource
def load_cnn_model():
    if not MODEL_PATH.exists():
        downloaded = gdown.download(
            id=FILE_ID,
            output=str(MODEL_PATH),
            quiet=False
        )

        if not downloaded or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check Google Drive access."
            )

    return tf.keras.models.load_model(MODEL_PATH)


# =====================================
# 3. CLASS SETTINGS
# =====================================
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

CLASS_NAMES_BN = {
    "Healthy": "সুস্থ পাতা",
    "Powdery": "পাউডারি মিলডিউ রোগ",
    "Rust": "রাস্ট বা মরিচা রোগ"
}


# =====================================
# 4. DISEASE INFORMATION
# =====================================
DISEASE_INFO = {
    "Healthy": {
        "name_bn": "সুস্থ পাতা",
        "status_bn": "পাতাটি সুস্থ শ্রেণিতে শনাক্ত হয়েছে।",
        "symptoms_bn": (
            "মডেলটি ছবির পাতাকে সুস্থ হিসেবে শনাক্ত করেছে। "
            "তবে শুধু AI prediction দিয়ে গাছের সম্পূর্ণ সুস্থতা "
            "নিশ্চিত করা যায় না।"
        ),
        "symptoms_en": (
            "The model classified the leaf as healthy. "
            "An AI prediction cannot guarantee complete plant health."
        ),
        "treatment_bn": [
            "গাছের প্রয়োজন অনুযায়ী পানি ও আলো দাও।",
            "মাটিতে যেন অতিরিক্ত পানি জমে না থাকে।",
            "নিয়মিত পাতা পরীক্ষা করো।"
        ],
        "treatment_en": [
            "Provide suitable water and light.",
            "Ensure good soil drainage.",
            "Inspect leaves regularly."
        ]
    },

    "Powdery": {
        "name_bn": "পাউডারি মিলডিউ রোগ",
        "status_bn": "পাতায় পাউডারি মিলডিউ রোগের লক্ষণ থাকতে পারে।",
        "symptoms_bn": (
            "পাতার ওপর সাদা পাউডারের মতো ছত্রাকের আস্তরণ দেখা যেতে পারে। "
            "আক্রান্ত পাতা হলুদ হতে পারে এবং গাছের বৃদ্ধি ব্যাহত হতে পারে।"
        ),
        "symptoms_en": (
            "White powder-like fungal growth may appear on leaves. "
            "Affected leaves may turn yellow and plant growth may be reduced."
        ),
        "treatment_bn": [
            "আক্রান্ত পাতা সাবধানে সরিয়ে ফেলো।",
            "গাছের চারপাশে পর্যাপ্ত বাতাস চলাচলের ব্যবস্থা রাখো।",
            "অতিরিক্ত নাইট্রোজেন সার ব্যবহার এড়িয়ে চলো।",
            "প্রয়োজনে গাছের উপযোগী অনুমোদিত ছত্রাকনাশক "
            "লেবেলের নির্দেশনা অনুযায়ী ব্যবহার করো।"
        ],
        "treatment_en": [
            "Carefully remove affected leaves.",
            "Ensure adequate air circulation around the plant.",
            "Avoid excessive nitrogen fertilizer.",
            "If necessary, use an approved fungicide suitable "
            "for the plant and follow the product label."
        ]
    },

    "Rust": {
        "name_bn": "রাস্ট বা মরিচা রোগ",
        "status_bn": "পাতায় রাস্ট রোগের লক্ষণ থাকতে পারে।",
        "symptoms_bn": (
            "পাতার ওপর হলুদ, কমলা, লালচে বা বাদামি রঙের "
            "ছোট ছোট দাগ দেখা যেতে পারে। পাতার নিচের দিকে "
            "মরিচার গুঁড়ার মতো স্পোরও দেখা যেতে পারে।"
        ),
        "symptoms_en": (
            "Small yellow, orange, reddish, or brown spots may appear "
            "on leaves. Rust-colored powdery spores may also develop "
            "on the undersides of leaves."
        ),
        "treatment_bn": [
            "আক্রান্ত পাতা সংগ্রহ করে গাছ থেকে সরিয়ে ফেলো।",
            "গাছের মধ্যে পর্যাপ্ত দূরত্ব রাখো, যাতে বাতাস চলাচল করতে পারে।",
            "পাতা দীর্ঘ সময় ভেজা রাখা এড়িয়ে চলো।",
            "প্রয়োজনে স্থানীয় কৃষি বিশেষজ্ঞের পরামর্শ নাও।"
        ],
        "treatment_en": [
            "Collect and remove affected leaves.",
            "Provide adequate spacing for air circulation.",
            "Avoid keeping foliage wet for extended periods.",
            "Consult a local agricultural expert if necessary."
        ]
    }
}


# =====================================
# 5. IMAGE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "পাতার ছবি নির্বাচন করো",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("ছবির সাইজ 10 MB-এর কম হতে হবে।")
        st.stop()

    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error("সঠিক ছবির ফাইল আপলোড করো।")
        st.stop()

    st.image(
        image,
        caption="আপলোড করা পাতার ছবি",
        use_container_width=True
    )

    # =====================================
    # 6. DISEASE PREDICTION
    # =====================================
    if st.button("🔍 Detect Plant Disease", type="primary"):

        try:
            with st.spinner("CNN মডেল দিয়ে পরীক্ষা করা হচ্ছে..."):

                model = load_cnn_model()

                resized_image = image.resize((225, 225))

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                image_array = image_array / 255.0

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                probabilities = np.asarray(
                    predictions[0],
                    dtype=np.float64
                )

                if probabilities.shape != (3,):
                    st.error(
                        "মডেলের output shape তিনটি class-এর "
                        "সঙ্গে মিলছে না।"
                    )
                    st.stop()

                if not np.all(np.isfinite(probabilities)):
                    st.error("মডেলের prediction-এ অবৈধ মান পাওয়া গেছে।")
                    st.stop()

                # Convert logits to probabilities when necessary.
                # This assumes the model outputs either softmax
                # probabilities or three-class logits.
                if (
                    np.any(probabilities < 0)
                    or not np.isclose(
                        probabilities.sum(),
                        1.0,
                        atol=0.01
                    )
                ):
                    shifted = probabilities - probabilities.max()
                    exp_values = np.exp(shifted)
                    probabilities = exp_values / exp_values.sum()

                else:
                    probabilities = (
                        probabilities / probabilities.sum()
                    )

                index = int(np.argmax(probabilities))

                predicted_class = CLASS_NAMES[index]

                confidence = float(
                    probabilities[index]
                ) * 100

            # =====================================
            # 7. DISPLAY PREDICTION
            # =====================================
            st.success(
                f"Prediction: {CLASS_NAMES_BN[predicted_class]}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

           

            # =====================================
            # 8. CLASS PROBABILITIES
            # =====================================
            st.subheader("📊 সব Class-এর Probability")

            for name, prob in zip(CLASS_NAMES, probabilities):
                st.write(
                    f"{CLASS_NAMES_BN[name]}: {prob * 100:.2f}%"
                )
                st.progress(float(prob))

            # =====================================
            # 9. DISEASE INFORMATION
            # =====================================
            disease = DISEASE_INFO[predicted_class]

            st.subheader("🔎 রোগের লক্ষণ ও বিবরণ")

            st.write(disease["status_bn"])

            st.markdown("**বাংলায় বিস্তারিত:**")
            st.write(disease["symptoms_bn"])

            st.markdown("**Symptoms in English:**")
            st.write(disease["symptoms_en"])

            # =====================================
            # 10. TREATMENT AND PREVENTION
            # =====================================
            st.subheader("💡 প্রাথমিক পরামর্শ")

            for advice in disease["treatment_bn"]:
                st.write(f"- {advice}")

            with st.expander("English: Treatment and Prevention"):
                for advice in disease["treatment_en"]:
                    st.write(f"- {advice}")

        except Exception as error:
            st.error("Prediction চালানো যায়নি।")
            st.code(str(error))


# =====================================
# 11. FOOTER
# =====================================
st.divider()

st.caption(
    "LeafCare AI | CNN-based Plant Disease Detection"
)


