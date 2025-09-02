import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List

from torchvision import datasets, transforms

from ai_edge_litert.interpreter import Interpreter

CLASS_LABELS = "imagenet_classes.txt"

IMAGENETV2_FOLDER = "/scratch/msc25f15/datasets/imagenetv2-top-images/imagenetv2-top-images-format-val/"


dataset = datasets.ImageFolder(root=IMAGENETV2_FOLDER)

idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()} # ImageFolder creates some internal ids for the classes

filename_to_label = {path.split('/')[-1]: idx_to_class[label] for path, label in dataset.samples}

print(filename_to_label["9676ff2db28e84a86ad81e4fe98b536439ca2b5a.jpeg"])

def print_top5_labels(output_data):
    with open(CLASS_LABELS) as f:
        labels = [line.strip() for line in f]

        # Get top 5 predictions and their indices
        top5_indices = np.argsort(output_data[0])[::-1][:5]  # sort in descending order
        top5_confidences = output_data[0][top5_indices]      # get corresponding confidence values

        # Print top 5 predicted labels with their confidence scores
        top5_string = "Top 5 Predictions:\n" + "\n".join([f"{i+1}: {labels[top5_indices[i]]} (Confidence: {top5_confidences[i]:.4f})"
                                  for i in range(5)]) + "\n"
        print(top5_string)
        return top5_string


def get_top5_labels(output_data: np.ndarray) -> List[str]:
    with open(CLASS_LABELS) as f:
        labels = [line.strip() for line in f]
        top5_indices = np.argsort(output_data[0])[::-1][:5]
        top5_labels = [labels[i] for i in top5_indices]
        return top5_labels


def map_index_to_label(index: int) -> str:
    with open(CLASS_LABELS) as f:
        labels = [line.strip() for line in f]
        if 0 <= index < len(labels):
            return labels[index]
        else:
            raise ValueError(f"Index {index} is out of bounds.")

def check_topN_match(image_name:str, out_data: np.ndarray, N:int = 5) -> bool:
    """
    Takes two output data array as returned from tflite interpreter.
    Compares if the correct label is in the top N classifications.
    """
    correct_classes = dict()
    with open(TEST_SET_CLASSIFICATIONS) as f:
       lines = f.readlines()
       for line in lines: 
            split_line = line.strip().split(" ")
            correct_classes[split_line[0]] = int(split_line[1])
    
    truth = correct_classes.get(image_name, None)
    topN_indices = np.argsort(out_data[0])[::-1][:N]
    print(f"Top {N} indices for {image_name}: {topN_indices}")
    print(f"Truth for {image_name}: {truth} {map_index_to_label(truth)}")
    print(f"Actual result for {image_name}: {map_index_to_label(topN_indices[0])}")
    if truth is not None:
        return truth in topN_indices
    else:
        raise ValueError(f"Image name {image_name} not found in the test set classifications.")

def main():
    images = list(Path("images").glob("*.bmp"))
    interpreter = Interpreter("resnet18.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input details:", input_details)
    print("Output details:", output_details)

    for image_path in images:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        input_data = np.asarray(image, dtype=np.float32) / 255.0  # Shape: [224, 224, 3]
        input_data = np.transpose(input_data, (2, 0, 1))  # Shape: [3, 224, 224]
        input_data = np.expand_dims(input_data, axis=0)  # Shape: [1, 3, 224, 224]


        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(check_topN_match(image_path.stem, output_data, N=1))
        # top5_string = print_top5_labels(output_data)
        top5_labels = get_top5_labels(output_data)

        # get cva6_result
        with open(Path("results") / f"{image_path.stem}_results.txt", "r") as f:
            lines = f.readlines()
            top1_result_cva6 = lines[0].strip().split(": ")[1].split("(")[0].strip()
            if top1_result_cva6 != top5_labels[0]:
                for line in lines:
                    if not line.startswith("["):
                        print(line.strip())
                print_top5_labels(output_data)
                print("Top-1 Result CVA6:", top1_result_cva6)
                print("Top-1 Result desktop:", top5_labels[0])


if __name__ == "__main__":
    main()