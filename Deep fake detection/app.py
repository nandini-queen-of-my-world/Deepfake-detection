import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# Initialize models
model1 = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)
checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model1.load_state_dict(checkpoint['model_state_dict'])
model1.to(DEVICE)
model1.eval()

model2 = models.efficientnet_b0(pretrained=True)
model2.classifier[1] = torch.nn.Linear(model2.classifier[1].in_features, 1)
model2.to(DEVICE)
model2.eval()

model3 = models.resnet50(pretrained=True)
model3.fc = torch.nn.Linear(model3.fc.in_features, 1)
model3.to(DEVICE)
model3.eval()

model4 = models.densenet121(pretrained=True)
model4.classifier = torch.nn.Linear(model4.classifier.in_features, 1)
model4.to(DEVICE)
model4.eval()

model5 = models.mobilenet_v2(pretrained=True)
model5.classifier[1] = torch.nn.Linear(model5.classifier[1].in_features, 1)
model5.to(DEVICE)
model5.eval()

# Preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(BytesIO(image_data)).convert('RGB')

    face = mtcnn(image)
    if face is None:
        return jsonify({'error': 'No face detected'}), 400

    face = face.unsqueeze(0)  # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    # Convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0

    # InceptionResnetV1 prediction
    with torch.no_grad():
        output1 = torch.sigmoid(model1(face).squeeze(0))
        prediction1 = "real" if output1.item() < 0.5 else "fake"
        real_prediction1 = 1 - output1.item()
        fake_prediction1 = output1.item()

    # EfficientNet prediction
    face_transformed = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output2 = torch.sigmoid(model2(face_transformed).squeeze(0))
        prediction2 = "real" if output2.item() < 0.5 else "fake"
        real_prediction2 = 1 - output2.item()
        fake_prediction2 = output2.item()

    # ResNet50 prediction
    with torch.no_grad():
        output3 = torch.sigmoid(model3(face_transformed).squeeze(0))
        prediction3 = "real" if output3.item() < 0.5 else "fake"
        real_prediction3 = 1 - output3.item()
        fake_prediction3 = output3.item()

    # DenseNet prediction
    with torch.no_grad():
        output4 = torch.sigmoid(model4(face_transformed).squeeze(0))
        prediction4 = "real" if output4.item() < 0.5 else "fake"
        real_prediction4 = 1 - output4.item()
        fake_prediction4 = output4.item()

    # MobileNetV2 prediction
    with torch.no_grad():
        output5 = torch.sigmoid(model5(face_transformed).squeeze(0))
        prediction5 = "real" if output5.item() < 0.5 else "fake"
        real_prediction5 = 1 - output5.item()
        fake_prediction5 = output5.item()

    # Majority voting for final decision
    predictions = [prediction1, prediction2, prediction3, prediction4, prediction5]
    final_decision = "real" if predictions.count("real") > predictions.count("fake") else "fake"
    
    final_decision_html = f"<div style='background-color: {'#28a745' if final_decision == 'real' else '#dc3545'}; color: white; padding: 10px; text-align: center;'><b>{final_decision.upper()}</b></div>"

    confidences = {
        'InceptionResnetV1': {'real': real_prediction1, 'fake': fake_prediction1},
        'EfficientNet': {'real': real_prediction2, 'fake': fake_prediction2},
        'ResNet50': {'real': real_prediction3, 'fake': fake_prediction3},
        'DenseNet121': {'real': real_prediction4, 'fake': fake_prediction4},
        'MobileNetV2': {'real': real_prediction5, 'fake': fake_prediction5}
    }

    return jsonify({
        'confidences': confidences,
        'final_decision': final_decision_html
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
