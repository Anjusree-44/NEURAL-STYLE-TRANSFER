import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import io

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
def load_image(img_data, max_size=256):  # ⬅️ faster with smaller images
    image = Image.open(img_data).convert("RGB")
    size = max_size if max(image.size) > max_size else max(image.size)
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

# Convert tensor to PIL image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return Image.fromarray((image * 255).astype("uint8"))

# Feature extractor
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# Style transfer engine
def run_style_transfer(content, style, steps=100, style_weight=1e2, content_weight=1e4):
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    for _ in range(steps):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_loss / (target_feature.shape[1] ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return target

# Load VGG
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.1
}

# ========== Streamlit UI ==========

st.set_page_config(page_title="Fast Style Transfer", layout="centered")
st.title("🎨 Neural Style Transfer (Optimized)")
st.write("Upload a **content image** and a **style image**, and generate a stylized image fast.")

col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
with col2:
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_img = load_image(content_file)
    style_img = load_image(style_file)

    st.subheader("Input Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(content_file, caption="Content Image", use_container_width=True)
    with col2:
        st.image(style_file, caption="Style Image", use_container_width=True)

    if st.button("🎨 Generate Stylized Image"):
        with st.spinner("Processing..."):
            output = run_style_transfer(content_img, style_img, steps=100)
            final_image = im_convert(output)
            st.image(final_image, caption="Stylized Output", use_container_width=True)

            # Download button
            buf = io.BytesIO()
            final_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button("⬇️ Download Stylized Image", byte_im, file_name="stylized.jpg", mime="image/jpeg")
