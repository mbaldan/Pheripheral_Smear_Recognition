import torch
import torchvision.transforms as transforms
from PIL import Image
import json

def load_model_and_indices(model_path, indices_path='class_indices_torch.json', device='cpu'):
    """Loads the trained model and class indices."""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(json.load(open(indices_path, 'r'))))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    with open(indices_path, 'r') as f:
        class_indices = json.load(f)
    return model, class_indices

def predict_image(model, img_path, class_indices, device):
    """Predicts the class of a single image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
        confidence = torch.softmax(output, dim=1)[0][predicted_idx].item()

    reverse_class_indices = {v: k for k, v in class_indices.items()}
    predicted_class = reverse_class_indices[predicted_idx]

    return predicted_class, confidence

if __name__ == "__main__":
    model_path = 'blood_cell_resnet_torch.pth'
    img_path = 'path/to/your/test_image.jpg'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, class_indices = load_model_and_indices(model_path, device=device)
    predicted_class, confidence = predict_image(model, img_path, class_indices, device)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")