import os
import shutil

dataset_dir = "data/Indian Food Images/Indian Food Images"
min_images = 40

def filter_classes():
    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} not found!")
        return

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Total initial classes: {len(classes)}")

    dropped_classes = []
    kept_classes = []

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) < min_images:
            dropped_classes.append((cls, len(images)))
            shutil.rmtree(cls_dir)
        else:
            kept_classes.append((cls, len(images)))

    print(f"\nKept {len(kept_classes)} classes.")
    print(f"Dropped {len(dropped_classes)} classes with fewer than {min_images} images:")
    for cls, count in dropped_classes:
        print(f"  - {cls} ({count} images)")

if __name__ == "__main__":
    filter_classes()
