"""Aircraft Damage Classification and Captioning Package.

Modules:
- config: Configuration settings and environment variables.
- logging_setup: Application logger configuration.
- errors: Domain-specific typed exceptions.
- data: Dataset downloading, extraction, verification, and leakage detection.
- classifier: VGG16 model building and compilation.
- train: Training orchestrator with real metric recording.
- evaluate: Truncated academic evaluation and comprehensive full-test evaluation.
- inference: Real image inference with preprocessing parity.
- blip_layer: Custom Keras Layer BlipCaptionSummaryLayer and generate_text.
- captioning: HuggingFace BLIP processor and generation wrapper.
- api: FastAPI web service serving static assets and REST endpoints.
"""

__version__ = "1.0.0"
