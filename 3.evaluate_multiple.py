import os
import evaluate_single_image_resnet_torch
import json

def evaluate_multiple_images(model_path, image_dir, device='cpu', class_indices_path='class_indices_torch.json'):
    """Evaluates multiple images in a directory and provides aggregated results."""
    model, class_indices = evaluate_single_image_resnet_torch.load_model_and_indices(model_path, device=device, indices_path=class_indices_path)
    results = {}

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            predicted_class, confidence = evaluate_single_image_resnet_torch.predict_image(model, img_path, class_indices, device)
            results[filename] = {'predicted_class': predicted_class, 'confidence': confidence}

    class_counts = {}
    for result in results.values():
        predicted_class = result['predicted_class']
        if predicted_class in class_counts:
            class_counts[predicted_class] += 1
        else:
            class_counts[predicted_class] = 1

    return results, class_counts

if __name__ == "__main__":
    model_path = 'blood_cell_resnet_torch.pth'
    image_dir = 'path/to/your/test_images_dir'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results, class_counts = evaluate_multiple_images(model_path, image_dir, device=device)

    print("Individual image results