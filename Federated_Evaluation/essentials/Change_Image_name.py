import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the dataset using torchvision.datasets.ImageFolder
root = "/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/Augmented"
dataset = torchvision.datasets.ImageFolder(root=root,transform = transform)
#print(dataset.imgs[0][0])

# Create the output directory to save the augmented images
output_directory = '/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/Augemented_Images_4_new/'
os.makedirs(output_directory, exist_ok=True)


count = 473
# Apply augmentation for the desired class and save the augmented images
for idx, (image, target) in enumerate(dataset):
	image_filename = dataset.imgs[idx][0].split('/')[-1]
	temp = image_filename.split('.')[0]
	count +=1
	print(temp)  # Extract the image filename
	augmented_image = image
	# Save the augmented image
	augmented_image_filename = f"IDRiD_{count}.jpg"
	augmented_image_path = os.path.join(output_directory, augmented_image_filename)
	torchvision.utils.save_image(augmented_image, augmented_image_path)
	print("saved")

