import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

# Define the transformation for your dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Custom augmentation for a specific class
class_augmentation = transforms.Compose([
	transforms.RandomVerticalFlip(p=0.7),
	transforms.RandomRotation((-10,10)),
])

# Load the CSV file using pandas
csv_file = '/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/2. Groundtruths/TrainingLabels.csv'
df = pd.read_csv(csv_file)

# Extract the desired class and image titles from the DataFrame
desired_class = 4
desired_images = df.loc[df['Retinopathy grade'] == desired_class, 'Image name'].tolist()
print(desired_images)
# Load the dataset using torchvision.datasets.ImageFolder
root = "/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/1. Original Images/Train_Set/"
dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
#print(dataset.imgs[0][0])

# Create the output directory to save the augmented images
output_directory = '/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/Augemented_Images_4/'
os.makedirs(output_directory, exist_ok=True)


# Apply augmentation for the desired class and save the augmented images
for idx, (image, target) in enumerate(dataset):
	image_filename = dataset.imgs[idx][0].split('/')[-1]
	temp = image_filename.split('.')[0]
	print(temp)  # Extract the image filename
	if temp in desired_images:
		augmented_image = class_augmentation(image)
		# Save the augmented image
		augmented_image_filename = f"{idx}_newvertical_augmented.jpg"
		augmented_image_path = os.path.join(output_directory, augmented_image_filename)
		torchvision.utils.save_image(augmented_image, augmented_image_path)
		print("saved")
