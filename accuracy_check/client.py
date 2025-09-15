


# The client that connects to the server running on the remote system with tflite
# The client is responsible for sending a new .bmp image to the remote that has the correct size (224x224) and for retrieving the output
# The server can be stopped by sending a file "STOP.bmp"

import os
from paramiko import SSHClient
from scp import SCPClient
from PIL import Image
import time
from pathlib import Path
from torchvision import datasets
import numpy as np

SERVER_ADDRESS = "172.31.182.93"
SERVER_INPUT_FOLDER = "/root/resnet/data/images"
SERVER_OUTPUT_FOLDER = "/root/resnet/data/outputs"

IMAGENETV2_FOLDER = "/scratch/msc25f15/datasets/imagenetv2-top-images/imagenetv2-top-images-format-val/"
BMP_FOLDER = "images"
dataset = datasets.ImageFolder(root=IMAGENETV2_FOLDER)

idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()} # ImageFolder creates some internal ids for the classes

filename_to_label = {path.split('/')[-1]: idx_to_class[label] for path, label in dataset.samples}

# print(filename_to_label["9676ff2db28e84a86ad81e4fe98b536439ca2b5a.jpeg"])


def resize_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    return image


def send_new_image(image_path):
    # Connect to the server and send the new image
    pass

def retrieve_output():
    pass

def main():
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(SERVER_ADDRESS, username="root")
    outputs_to_retrieve = []
    with SCPClient(ssh.get_transport()) as scp:
        # First upload to a tmp file so that we can rename it later which is atomic

        number_of_images = 1
        # Select at random
        random_indices = np.random.choice(len(dataset.samples), number_of_images, replace=False)

        images = [dataset.samples[i] for i in random_indices]
        # print(images)
        for image, id in images:
                image = Path(image)
                print(f"Processing image: {image}")
                resized = resize_image(image)
                resized.save(BMP_FOLDER + f"/{image.stem}.bmp", format='BMP')
                resized_image_path = Path(BMP_FOLDER + f"/{image.stem}.bmp")
                resized_image = resized_image_path.name
                print(f"Sent image to {SERVER_INPUT_FOLDER}/{resized_image}.tmp")
                scp.put(resized_image_path, SERVER_INPUT_FOLDER + f"/{resized_image}.tmp")
                ssh.exec_command(f"mv {SERVER_INPUT_FOLDER}/{resized_image}.tmp {SERVER_INPUT_FOLDER}/{resized_image}")
                print("Waiting for results...")
                time.sleep(60)
                outputs_to_retrieve.append(resized_image_path.stem)
                for otr in outputs_to_retrieve:
                    try:
                        print(f"Retrieving output for {otr}")
                        scp.get(SERVER_OUTPUT_FOLDER + f"/{otr}_results.txt", f"results/{otr}_results.txt")
                        outputs_to_retrieve.remove(otr)
                    except Exception as e:
                        print(f"Error retrieving output for {otr}: {e}")

        # Main function to run the client
        # scp.put("STOP.bmp", SERVER_INPUT_FOLDER + "/STOP.bmp")



if __name__ == "__main__":
    main()