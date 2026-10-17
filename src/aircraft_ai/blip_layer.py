"""Custom Keras Layer wrapping HuggingFace BLIP model as specified in the academic notebook."""

from pathlib import Path
from typing import Any, Union

from PIL import Image

from aircraft_ai import config
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.blip_layer")

# Lazy reference to global BLIP singleton processor and model
_GLOBAL_PROCESSOR = None
_GLOBAL_MODEL = None


def get_blip_components():
    """Retrieve or lazily initialize the singleton processor and PyTorch model for BLIP."""
    global _GLOBAL_PROCESSOR, _GLOBAL_MODEL
    if _GLOBAL_PROCESSOR is not None and _GLOBAL_MODEL is not None:
        return _GLOBAL_PROCESSOR, _GLOBAL_MODEL

    from aircraft_ai.captioning import get_blip_model
    _GLOBAL_PROCESSOR, _GLOBAL_MODEL = get_blip_model()
    return _GLOBAL_PROCESSOR, _GLOBAL_MODEL


def set_blip_components(processor, model):
    """Set global BLIP components explicitly."""
    global _GLOBAL_PROCESSOR, _GLOBAL_MODEL
    _GLOBAL_PROCESSOR = processor
    _GLOBAL_MODEL = model


# Subclassing tf.keras.layers.Layer matching notebook specifications exactly
def create_blip_layer_class():
    """Factory to create BlipCaptionSummaryLayer with tensorflow import isolated."""
    import tensorflow as tf

    class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
        """Custom Keras layer wrapping BLIP processor and model for captioning and summarization.
        
        NOTE: This layer is an academic demonstration wrapping PyTorch inference inside
        tf.py_function. It is NOT trainable, NOT exportable to a TensorFlow SavedModel,
        and cannot participate in gradient tape calculations.
        """

        def __init__(self, processor, model, **kwargs):
            super().__init__(**kwargs)
            self.processor = processor
            self.model = model

        def call(self, image_path, task=None):
            # Support both direct two-argument call `layer(path, task)` and tuple `layer((path, task))`
            # for Keras 3 compatibility where __call__ can pass inputs as single arg or positional args
            if task is None and isinstance(image_path, (tuple, list)):
                img_p, tsk = image_path[0], image_path[1]
            else:
                img_p, tsk = image_path, task

            return tf.py_function(self.process_image, [img_p, tsk], tf.string)

        def process_image(self, image_path, task):
            """Perform image loading, preprocessing, and text generation via BLIP."""
            try:
                # Decode the image path from the TensorFlow tensor to a Python string
                if hasattr(image_path, "numpy"):
                    image_path_str = image_path.numpy().decode("utf-8")
                elif isinstance(image_path, bytes):
                    image_path_str = image_path.decode("utf-8")
                else:
                    image_path_str = str(image_path)

                # Open the image using PIL and convert it to RGB format
                image = Image.open(image_path_str).convert("RGB")

                # Decode task string
                if hasattr(task, "numpy"):
                    task_str = task.numpy().decode("utf-8")
                elif isinstance(task, bytes):
                    task_str = task.decode("utf-8")
                else:
                    task_str = str(task)

                # Set the appropriate prompt based on the task
                if task_str == "caption":
                    prompt = config.CAPTION_PROMPT  # "This is a picture of"
                else:
                    prompt = config.SUMMARY_PROMPT  # "This is a detailed photo showing"

                # Prepare inputs for the BLIP model
                inputs = self.processor(images=image, text=prompt, return_tensors="pt")

                # Generate text output using the BLIP model
                output = self.model.generate(**inputs)

                # Decode the output into a readable string
                result = self.processor.decode(output[0], skip_special_tokens=True)
                return result
            except Exception as e:
                # Handle errors during image processing or text generation (academic notebook sentinel)
                print(f"Error: {e}")
                return "Error processing image"

    return BlipCaptionSummaryLayer


def generate_text(image_path: Any, task: Any) -> Any:
    """Implement the exact Task 8 helper function specified in the academic notebook.
    
    Accepts tf.Tensor or strings, passes through BlipCaptionSummaryLayer,
    and returns a tf.string tensor.
    """
    import tensorflow as tf

    processor, model = get_blip_components()
    BlipCaptionSummaryLayer = create_blip_layer_class()
    blip_layer = BlipCaptionSummaryLayer(processor, model)

    # Ensure tensors
    if not isinstance(image_path, tf.Tensor):
        image_path = tf.constant(str(image_path))
    if not isinstance(task, tf.Tensor):
        task = tf.constant(str(task))

    return blip_layer(image_path, task)
