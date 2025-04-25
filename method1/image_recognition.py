import cv2
import numpy as np
import time
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


model = ResNet50(weights='imagenet')

def recognize_image(image_path):
    #timing
    start_time = time.time()
    
    # get the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    #resize
    img = cv2.resize(img, (224, 224))
    

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    time_taken = time.time() - start_time
    
    #format
    results = {
        "image_path": str(image_path),
        "predictions": [],
        "time_taken_seconds": time_taken
    }
    
    for i, (_, label, prob) in enumerate(decoded_predictions, 1):
        results["predictions"].append({
            "rank": i,
            "label": label,
            "probability": f"{prob:.2%}"
        })
    
    return results

def print_results(results):
    print("\n=== Image Recognition Results ===")
    print(f"Image Path: {results['image_path']}")
    print(f"Time Taken: {results['time_taken_seconds']:.3f} seconds")
    print("\nTop Predictions:")
    for pred in results["predictions"]:
        print(f"  {pred['rank']}. {pred['label']}: {pred['probability']}")
    print("===============================\n")

def process_image_folder(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Folder {folder_path} does not exist or is not a directory")
    
    image_files = [f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No images found in {folder_path} or its subfolders")
    

    for image_path in image_files:
        try:
            results = recognize_image(image_path)
            print_results(results)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    try:

        image_folder = 'image'
        process_image_folder(image_folder)
    except Exception as e:
        print(f"Error: {str(e)}")