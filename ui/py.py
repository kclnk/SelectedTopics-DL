# ============================================================
# app.py
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Plant Leaf Disease Classification",
    layout="wide"
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# RICE CNN MODEL
# ============================================================

class RiceLeafCNN(nn.Module):

    def __init__(self, num_classes):
        super(RiceLeafCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# ============================================================
# CLASS NAMES
# ============================================================

rice_classes = [
    "Bacterialblight",
    "Blast",
    "Brownspot",
    "Tungro"
]

mango_classes = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# ============================================================
# TRANSFORMS
# ============================================================

rice_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mango_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD RICE MODEL
# ============================================================

@st.cache_resource
def load_rice_model():

    model = RiceLeafCNN(
        num_classes=len(rice_classes)
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/best_rice_leaf_cnn.pth",
            map_location=device
        )
    )

    model.eval()

    return model

# ============================================================
# LOAD MANGO MODEL
# ============================================================

@st.cache_resource
def load_mango_model():

    model = models.efficientnet_b1(pretrained=False)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(

        nn.Dropout(0.4),

        nn.Linear(in_features, 512),
        nn.ReLU(),

        nn.Dropout(0.3),

        nn.Linear(512, len(mango_classes))
    )

    model.load_state_dict(
        torch.load(
            "models/best_mango_effb1.pth",
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(
    image,
    model,
    transform,
    class_names
):

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score

# ============================================================
# HEADER
# ============================================================

st.title("🌿 Plant Leaf Disease Classification")

st.markdown("""
This application uses deep learning models to classify
plant leaf diseases from uploaded images.

Models included:
- Rice Leaf Disease Classification (Custom CNN)
- Mango Leaf Disease Classification (EfficientNet-B1)
""")

# ============================================================
# MODEL SELECTION
# ============================================================

st.subheader("Select Model")

col1, col2 = st.columns(2)

with col1:

    rice_button = st.button(
        "🌾 Rice Leaf Model",
        use_container_width=True
    )

with col2:

    mango_button = st.button(
        "🥭 Mango Leaf Model",
        use_container_width=True
    )

# Session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if rice_button:
    st.session_state.selected_model = "rice"

if mango_button:
    st.session_state.selected_model = "mango"

# ============================================================
# MODEL LOADING
# ============================================================

selected_model = st.session_state.selected_model

if selected_model == "rice":

    st.success("Rice Leaf Disease Model Selected")

    model = load_rice_model()

    transform = rice_transform

    class_names = rice_classes

elif selected_model == "mango":

    st.success("Mango Leaf Disease Model Selected")

    model = load_mango_model()

    transform = mango_transform

    class_names = mango_classes

else:

    st.info("Please select a model to continue.")

    st.stop()

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload Leaf Images")

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ============================================================
# PREDICTION SECTION
# ============================================================

if uploaded_files:

    st.subheader("Prediction Results")

    cols = st.columns(3)

    for idx, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")

        predicted_class, confidence = predict_image(
            image=image,
            model=model,
            transform=transform,
            class_names=class_names
        )

        with cols[idx % 3]:

            st.image(
                image,
                width=350
            )

            st.markdown(
                f"""
                ### Prediction
                **{predicted_class}**

                ### Confidence
                **{confidence:.2f}%**
                """
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
### Developed For
Selected Topics: Deep Learning Course Project

Using:
- Custom CNN
- EfficientNet-B1
- PyTorch
- Streamlit
""")