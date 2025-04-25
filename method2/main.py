import boto3
import time
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import argparse
import json
from credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# Configuration
IMAGE_CLASSES = ['human', 'dog', 'boat', 'bird', 'fish']  # 5 ImageNet classes
REKOGNITION_PRICING = 0.001  # $0.001 per image (AWS reference pricing)

# Initialize Rekognition
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
rekognition = session.client('rekognition')

# 1. Get local image paths (dynamic count)
def get_image_paths(class_name, image_dir):
    class_dir = os.path.join(image_dir, class_name)
    print(f"Listing images in {class_dir}")
    image_paths = []
    if os.path.exists(class_dir):
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, file))
        print(f"Found {len(image_paths)} images for class {class_name}: {image_paths}")
    else:
        print(f"Directory not found: {class_dir}")
    if not image_paths:
        print(f"Warning: No images found for class {class_name}")
    return image_paths

# 2. Image recognition function (local images)
def detect_labels(image_path):
    start_time = time.time()
    try:
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        response = rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=70
        )
        latency = time.time() - start_time
        labels = [label['Name'].lower() for label in response['Labels']]
        return labels, latency
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return [], 0

# 3. Accuracy calculation (simple matching)
def calculate_accuracy(predicted_labels, true_class):
    return 1 if true_class.lower() in predicted_labels else 0

# 4. Benchmarking (batch images, dynamic count)
def run_benchmarks(image_dir):
    results = []
    total_images = 0
    total_start = time.time()

    for class_name in IMAGE_CLASSES:
        image_paths = get_image_paths(class_name, image_dir)
        if not image_paths:
            continue
        total_images += len(image_paths)
        for image_path in image_paths:
            labels, latency = detect_labels(image_path)
            accuracy = calculate_accuracy(labels, class_name)
            results.append({
                'Class': class_name,
                'Image': image_path,
                'Accuracy': accuracy,
                'Latency': latency,
                'Cost': REKOGNITION_PRICING
            })

    total_time = time.time() - total_start
    throughput = total_images / total_time if total_time > 0 else 0

    # Summarize results
    df = pd.DataFrame(results)
    avg_accuracy = df['Accuracy'].mean() if not df.empty else 0
    avg_latency = df['Latency'].mean() if not df.empty else 0
    total_cost = df['Cost'].sum() if not df.empty else 0

    print(f"Average Accuracy: {avg_accuracy:.2%}")
    print(f"Average Latency: {avg_latency:.2f} seconds")
    print(f"Throughput: {throughput:.2f} images/second")
    print(f"Total Cost: ${total_cost:.3f}")

    return df, throughput

# 5. Single image test
def test_single_image(image_path, true_class):
    labels, latency = detect_labels(image_path)
    accuracy = calculate_accuracy(labels, true_class)
    print(f"Image: {image_path}")
    print(f"True Class: {true_class}")
    print(f"Predicted Labels: {labels}")
    print(f"Accuracy: {accuracy:.0%}")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Cost: ${REKOGNITION_PRICING:.3f}")
    return accuracy, latency, REKOGNITION_PRICING

# 6. Scalability test (simulated concurrency)
def test_scalability(image_path):
    concurrency_levels = [1, 10, 50, 100] # 100 is the maximum number of concurrent users
    latencies = []
    if not os.path.exists(image_path):
        print(f"Error: Sample image {image_path} not found for scalability test")
        return concurrency_levels, [0] * len(concurrency_levels)

    for users in concurrency_levels:
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=users) as executor:
            futures = [executor.submit(detect_labels, image_path) for _ in range(users)]
            for future in futures:
                future.result()
        latency = (time.time() - start_time) / users
        latencies.append(latency)
    return concurrency_levels, latencies

# 7. Generate charts
def plot_results(df, throughput, concurrency_levels, latencies):
    # Bar chart: Latency, throughput, cost
    metrics = {
        'Average Latency (seconds)': df['Latency'].mean() if not df.empty else 0,
        'Throughput (images/second)': throughput,
        'Total Cost ($)': df['Cost'].sum() if not df.empty else 0
    }
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red'])
    plt.title('Rekognition Benchmark Results')
    plt.savefig('bar_metrics.png')
    plt.close()

    # Line chart: Latency under high concurrency
    plt.figure(figsize=(10, 6))
    plt.plot(concurrency_levels, latencies, marker='o', color='blue')
    plt.title('Average Latency Under High Concurrency')
    plt.xlabel('Number of Concurrent Users')
    plt.ylabel('Average Latency (seconds)')
    plt.grid(True)
    plt.savefig('line_concurrency.png')
    plt.close()

    # Pie chart: Cost distribution by class
    if not df.empty:
        class_costs = df.groupby('Class')['Cost'].sum()
        plt.figure(figsize=(8, 8))
        plt.pie(class_costs, labels=class_costs.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        plt.title('Cost Distribution by Class')
        plt.savefig('pie_costs.png')
        plt.close()
    else:
        print("No data for pie chart")

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS Rekognition Image Recognition")
    parser.add_argument('--image_url', type=str, help="Path to a single image (e.g., E:\\Desktop\\CS4296\\imagenet\\bird\\n01531178_10679.JPEG)")
    parser.add_argument('--image_dir', type=str, help="Directory containing images (e.g., E:\\Desktop\\CS4296\\imagenet)")
    parser.add_argument('--true_class', type=str, help="True class for single image (e.g., bird)")
    args = parser.parse_args()

    if args.image_url:
        # Single image test
        if not args.true_class:
            print("Error: --true_class is required when using --image_url")
        elif args.true_class not in IMAGE_CLASSES:
            print(f"Error: True class must be one of {IMAGE_CLASSES}")
        else:
            test_single_image(args.image_url, args.true_class)
    elif args.image_dir:
        # Batch image benchmarking (dynamic count)
        df, throughput = run_benchmarks(args.image_dir)
        # Scalability test (using the first bird image)
        sample_image = get_image_paths('bird', args.image_dir)[0] if get_image_paths('bird', args.image_dir) else None
        concurrency_levels, latencies = test_scalability(sample_image) if sample_image else ([1, 10, 50, 100], [0] * 4)
        # Generate charts
        plot_results(df, throughput, concurrency_levels, latencies)
    else:
        print("Error: Please provide --image_url or --image_dir")