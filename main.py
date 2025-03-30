import os
import csv

# Define the path to your dataset directory
dataset_dir = '.'  # Adjust this to your actual path if necessary
csv_file = 'plant_stages_labels.csv'

# Prepare CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "label"])  # CSV header

    # Walk through each plant type folder
    for plant_type in os.listdir(dataset_dir):
        plant_path = os.path.join(dataset_dir, plant_type)
        
        if os.path.isdir(plant_path):
            # Process each image file in the plant type folder
            for image_file in os.listdir(plant_path):
                if image_file.endswith((".jpeg", ".jpg", ".png")):
                    # Full path to image relative to the dataset directory
                    image_path = os.path.join(plant_type, image_file)
                    
                    # Extract stage number from the image file name (assuming 'stage_X')
                    stage_number = image_file.split('_')[1].split('.')[0]
                    
                    # Create a label in the format 'plant_type_stageNumber' (e.g., 'okra_1')
                    label = f"{plant_type}_{stage_number}"
                    
                    # Write the image path and label to the CSV
                    writer.writerow([image_path, label])

print(f"CSV file '{csv_file}' created successfully!")
