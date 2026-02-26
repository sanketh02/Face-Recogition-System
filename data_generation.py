import os
import cv2
import random
import numpy as np
import albumentations as A


# AUGMENTATION PIPELINE
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.Rotate(limit=20, p=0.7),
        A.GaussianBlur(p=0.3),
        A.RandomGamma(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.05,
                           rotate_limit=15,
                           p=0.5),
        A.CLAHE(p=0.3),
    ])


# MAIN FUNCTION
def augment_dataset(base_path="dataset", target_count=20):
    
    augmentation = get_augmentation_pipeline()
    
    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"\nProcessing: {person_name}")

        images = [img for img in os.listdir(person_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        current_count = len(images)
        print(f"Existing images: {current_count}")

        if current_count >= target_count:
            print("Already has sufficient images.")
            continue

        image_paths = [os.path.join(person_path, img) for img in images]

        while current_count < target_count:
            img_path = random.choice(image_paths)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = augmentation(image=image)
            augmented_image = augmented['image']

            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            new_filename = f"aug_{current_count}.jpg"
            save_path = os.path.join(person_path, new_filename)

            cv2.imwrite(save_path, augmented_image)

            current_count += 1

        print(f"Total images after augmentation: {current_count}")

    print("\nDataset augmentation completed successfully ✅")


# RUN FUNCTION
if __name__ == "__main__":
    augment_dataset(base_path="dataset", target_count=20)