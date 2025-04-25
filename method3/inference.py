import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

import time

if __name__ == "__main__":
    images = ["bird1.JPEG", "bird2.JPEG", "boat1.JPEG", "boat2.JPEG", "dog1.JPEG", 
              "dog2.JPEG", "fish1.JPEG", "fish1.JPEG", "fish2.JPEG", "human1.JPEG", 
              "human2.JPEG"]
    IMAGE_SIZE = 384
    pretrained_model = "ram_plus_swin_large_14m.pth"

    total_start = time.time()
    for image in images:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = get_transform(image_size=IMAGE_SIZE)
        model = ram_plus(pretrained=pretrained_model,
                                image_size=IMAGE_SIZE,
                                vit='swin_l')
        model.eval()

        model = model.to(device)

        image = transform(Image.open(image)).unsqueeze(0).to(device)

        start_time = time.time()
        res = inference(image, model)
        end_time = time.time()

        predicted_result = "undefined"
        if "bird" in res[0]:
            predicted_result = "bird"
        elif "boat" in res[0]:
            predicted_result = "boat"
        elif "dog" in res[0]:
            predicted_result = "dog"
        elif "fish" in res[0]:
            predicted_result = "fish"
        elif "shark" in res[0]:
            predicted_result = "shark"
        elif "human" in res[0]:
            predicted_result = "human"
        elif "boy" in res[0]:
            predicted_result = "boy"
        elif "girl" in res[0]:
            predicted_result = "girl"

        inference_time = end_time - start_time
        print(f"Predicted Result: {predicted_result}")
        print(f"Time Used: {inference_time:.4f} Second")

    total_end = time.time()
    print(f"Total Time Used: {(total_end - total_start):.4f} Second")