"""Script to generate the completed notebook from the unmodified original.

Fills in all 10 tasks, defines batch_size=32 and n_epochs=5, corrects Colab-specific paths,
guards dataset extraction from accidental re-deletion, and appends environment version logging.
"""

import json
from pathlib import Path

ORIGINAL_NB_PATH = Path("notebooks/original/Final_Project_Classification_and_Captioning.ipynb")
COMPLETED_NB_PATH = Path("notebooks/completed/Final_Project_Classification_and_Captioning.ipynb")


def main():
    with open(ORIGINAL_NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # 1. Disable !pip install cell execution in completed notebook
    for c in cells:
        src = "".join(c.get("source", []))
        if "!pip install" in src:
            c["source"] = [
                "# Note: Dependencies are pre-installed in the virtual environment.\n",
                "# Uncomment if running in an isolated external cloud kernel.\n",
            ] + ["# " + line for line in c["source"]]

    # 2. Config cell: fill batch_size and n_epochs
    for c in cells:
        src = "".join(c.get("source", []))
        if "#Set the batch size,epochs" in src:
            c["source"] = [
                "# Set the batch size,epochs\n",
                "batch_size = 32\n",
                "n_epochs = 5\n",
                "img_rows, img_cols = 224, 224\n",
                "input_shape = (img_rows, img_cols, 3)\n",
            ]

    # 3. Data download cell: guard against deleting existing dataset
    for c in cells:
        src = "".join(c.get("source", []))
        if "url = \"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud" in src:
            c["source"] = [
                "import tarfile\n",
                "import urllib.request\n",
                "import os\n",
                "import shutil\n",
                "\n",
                "# URL of the tar file\n",
                "url = \"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar\"\n",
                "\n",
                "# Define the path to save the file\n",
                "tar_filename = \"aircraft_damage_dataset_v1.tar\"\n",
                "extracted_folder = \"aircraft_damage_dataset_v1\"\n",
                "\n",
                "# Guard extraction: only download and extract if not already extracted and valid\n",
                "if not os.path.exists(extracted_folder) or len(os.listdir(extracted_folder)) == 0:\n",
                "    if not os.path.exists(tar_filename):\n",
                "        print(f\"Downloading {tar_filename}...\")\n",
                "        urllib.request.urlretrieve(url, tar_filename)\n",
                "        print(f\"Downloaded {tar_filename}.\")\n",
                "    print(f\"Extracting {tar_filename}...\")\n",
                "    with tarfile.open(tar_filename, \"r\") as tar_ref:\n",
                "        tar_ref.extractall()\n",
                "    print(f\"Extracted {tar_filename} successfully.\")\n",
                "else:\n",
                "    print(f\"Valid extracted folder '{extracted_folder}' already exists. Reusing existing dataset.\")\n",
            ]

    # 4. Task 1: valid_generator
    # Look for task 1 code cell
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 1: Create a `valid_generator`" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 1: Create valid_generator\n",
                "valid_generator = valid_datagen.flow_from_directory(\n",
                "    directory=valid_dir,\n",
                "    class_mode='binary',\n",
                "    seed=seed_value,\n",
                "    batch_size=batch_size,\n",
                "    shuffle=False,\n",
                "    target_size=(img_rows, img_cols)\n",
                ")\n",
            ]

    # 5. Task 2: test_generator
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 2: Create a `test_generator`" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 2: Create test_generator\n",
                "test_generator = test_datagen.flow_from_directory(\n",
                "    directory=test_dir,\n",
                "    class_mode='binary',\n",
                "    seed=seed_value,\n",
                "    batch_size=batch_size,\n",
                "    shuffle=False,\n",
                "    target_size=(img_rows, img_cols)\n",
                ")\n",
            ]

    # 6. Task 3: Load VGG16
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 3: Load the pre-trained model VGG16" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 3: Load pre-trained VGG16\n",
                "base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))\n",
            ]

    # 7. Task 4: Compile model
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 4: Compile the model" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 4: Compile the model\n",
                "model.compile(\n",
                "    optimizer=Adam(learning_rate=0.0001),\n",
                "    loss='binary_crossentropy',\n",
                "    metrics=['accuracy']\n",
                ")\n",
            ]

    # 8. Task 5: Train model
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 5: Model Training" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 5: Train the model\n",
                "history = model.fit(\n",
                "    train_generator,\n",
                "    epochs=n_epochs,\n",
                "    validation_data=valid_generator\n",
                ")\n",
            ]

    # 9. Task 6: Plot accuracy curves
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 6: Plot accuracy curves for training and validation sets" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 6: Plot accuracy curves for training and validation sets\n",
                "plt.figure(figsize=(5, 5))\n",
                "plt.plot(train_history['accuracy'], label='Training Accuracy')\n",
                "plt.plot(train_history['val_accuracy'], label='Validation Accuracy')\n",
                "plt.title('Accuracy Curve')\n",
                "plt.xlabel('Epochs')\n",
                "plt.ylabel('Accuracy')\n",
                "plt.legend()\n",
                "plt.show()\n",
            ]

    # 10. Task 7: Visualizing results
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 7: Visualizing the results" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 7: Visualizing the results\n",
                "test_model_on_image(test_generator, model, index_to_plot=1)\n",
            ]

    # 11. Task 8: Helper function generate_text
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 8: Implement a Helper Function" in prev and c["cell_type"] == "code":
            c["source"] = [
                "# Task 8: Implement helper function to use the custom Keras layer\n",
                "def generate_text(image_path, task):\n",
                "    blip_layer = BlipCaptionSummaryLayer(processor, model)\n",
                "    return blip_layer(image_path, task)\n",
            ]

    # 12. Task 9: Colab path replacement & real caption
    for idx, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "/content/sample_image.jpg" in src and c["cell_type"] == "code":
            c["source"] = [
                "# Task 9: Generate a caption for the test image using BLIP\n",
                "# Compatibility change: Replace Colab sample image path with the actual test dataset image\n",
                "image_path = tf.constant(\"aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg\")\n",
                "caption_output = generate_text(image_path, tf.constant(\"caption\"))\n",
                "print(\"Caption:\", caption_output.numpy().decode(\"utf-8\"))\n",
                "caption_output\n",
            ]

    # 13. Task 10: Generate summary
    for idx, c in enumerate(cells):
        prev = "".join(cells[idx - 1].get("source", [])) if idx > 0 else ""
        if "Task 10: Generate a summary of an image using BLIP pretrained model" in prev and c["cell_type"] == "code":
            # The next code cell is image_path, and the one after is the write your code cell
            target_cell = cells[idx + 1] if idx + 1 < len(cells) and cells[idx + 1]["cell_type"] == "code" else None
            if target_cell and "# Write your code here" in "".join(target_cell.get("source", [])):
                target_cell["source"] = [
                    "# Task 10: Generate summary for the image\n",
                    "summary_output = generate_text(image_path, tf.constant(\"summary\"))\n",
                    "print(\"Summary:\", summary_output.numpy().decode(\"utf-8\"))\n",
                    "summary_output\n",
                ]

    # 14. Compatibility documentation cell
    compat_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Compatibility and Environment Notes\n",
            "\n",
            "1. **Task 9 Path Fix**: The original lab template contained a hardcoded Google Colab path `'/content/sample_image.jpg'` which does not exist locally. It has been replaced with the verified test dataset image `'aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg'` as defined in the preceding task cell.\n",
            "2. **Hyperparameter Filling**: Batch size was set to 32 and epochs to 5 per markdown instructions.\n",
            "3. **Dataset Re-download Guard**: The download cell was enhanced to prevent destructive deletion of an existing valid dataset.\n",
        ],
    }

    # 15. Environment version cell at the end
    env_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import sys\n",
            "import tensorflow as tf\n",
            "import keras\n",
            "import torch\n",
            "import transformers\n",
            "import PIL\n",
            "import pandas as pd\n",
            "import matplotlib\n",
            "\n",
            "print('=== Runtime Environment ===')\n",
            "print('Python:', sys.version)\n",
            "print('TensorFlow:', tf.__version__)\n",
            "print('Keras:', keras.__version__)\n",
            "print('PyTorch:', torch.__version__)\n",
            "print('Transformers:', transformers.__version__)\n",
            "print('Pillow:', PIL.__version__)\n",
            "print('Pandas:', pd.__version__)\n",
            "print('Matplotlib:', matplotlib.__version__)\n",
        ],
    }

    cells.append(compat_cell)
    cells.append(env_cell)

    COMPLETED_NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPLETED_NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated completed notebook at {COMPLETED_NB_PATH}")


if __name__ == "__main__":
    main()
