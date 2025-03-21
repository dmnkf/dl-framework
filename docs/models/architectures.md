# Model Architectures

This section provides detailed documentation for all model architectures in the project. Our models are organized into several categories:

- [**Self-Supervised Learning**](#self-supervised-learning): Pre-training architectures for learning representations without labels
- [**Encoders**](#encoders): Architectures for extracting representations from data
- [**Classifiers**](#classifiers): Architectures for classifying data and modeling downstream tasks

## Base Components

The foundational interfaces and base classes that models build upon.

::: src.utils.model_weights
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true
        members: [PretrainedWeightsMixin]
        show_docstring_description: true

Our models implement a sophisticated weights management system through the `PretrainedWeightsMixin`. This system is designed with several key principles:

1. **Robustness**: Gracefully handle different weight file formats and structures
2. **Flexibility**: Support partial loading and key matching
3. **Safety**: Validate shapes and provide meaningful errors
4. **Transparency**: Detailed logging of the loading process

The mixin provides intelligent weight loading with features like:
- Automatic nested state dict extraction
- Configurable missing key tolerance
- Shape validation
- Detailed loading statistics
- Extensible key matching logic

!!! tip "Usage Example"
    ```python
    class MyModel(nn.Module, PretrainedWeightsMixin):
        def __init__(self):
            super().__init__()
            # ... model definition ...

        def load_my_weights(self, path):
            # Load with 20% missing key tolerance
            self.load_pretrained_weights(path, strict=False, missing_key_threshold=0.2)
    ```

!!! note "Design Philosophy"
    The mixin is designed to solve common issues in deep learning weight management:
    
    - **Versioning**: Models evolve, but weights should remain usable
    - **Flexibility**: Support both exact and partial loading
    - **Debugging**: Clear feedback about what was loaded
    - **Safety**: Prevent silent failures with shape mismatches
    - **Extensibility**: Easy to customize key matching logic

::: src.models.encoder_interface
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

The encoder interface abstracts the data preprocessing pipeline and provides a unified interface for all encoders in the project that can be used to extract features. This is mainly useful for the generation of representations and was specifically designed for this.

## Self-Supervised Learning

Our self-supervised learning implementations are based on state-of-the-art approaches adapted for medical data.

::: src.models.mae
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true
        members: [MAE]
        show_docstring_description: true
        docstring_section_style: spacy

!!! note "Implementation Credits"
    Our MAE implementation is based on:
    
    - Original paper: ["Masked Autoencoders Are Scalable Vision Learners"](https://arxiv.org/abs/2111.06377) by He et al.
    - Code adapted from [Turgut et al.'s MAE implementation](https://github.com/oetu/mae/tree/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65), which is a fork of [Facebook Research's MAE](https://github.com/facebookresearch/mae)
    - Adapted for time series data with 1D signal masking and ECG-specific components

::: src.models.mae_lit
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

::: src.models.sim_clr
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

!!! note "Implementation Credits"
    Our SimCLR implementation is based on:
    
    - Original paper: ["A Simple Framework for Contrastive Learning of Visual Representations"](https://arxiv.org/abs/2002.05709) by Chen et al.
    - Code adapted from [MMCL-ECG-CMR](https://github.com/oetu/MMCL-ECG-CMR) by Turgut et al.

## Encoders

Model definitions mainly designed to obtain representations.

::: src.models.ecg_encoder
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

!!! note "Implementation Credits"
    The classifier also bases on the MAE implementation by Turgut et al. as outlined in the [MAE](#src.models.mae) section.


::: src.models.cmr_encoder
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

!!! note "Implementation Credits"
    The encoder also bases on the MMCL-ECG-CMR implementation by Turgut et al. as outlined in the [MMCL-ECG-CMR](#src.models.mmcl_ecg_cmr) section.


## Classifiers

Model definitions mainly designed to classify data. Generally, these models can be extended to perform any kind of downstream task.

::: src.models.ecg_classifier
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

!!! note "Implementation Credits"
    The classifier also bases on the MAE implementation by Turgut et al. as outlined in the [MAE](#src.models.mae) section.

::: src.models.cmr_classifier
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true

!!! note "Implementation Credits"
    The classifier also bases on the SimCLR implementation by Turgut et al. as outlined in the [SimCLR](#src.models.sim_clr) section.

## Utility Models

Helper models for evaluation and feature extraction. These models are not part of the main model architecture and are not meant to be used directly but rather as building blocks for other models.

::: src.models.linear_classifier
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true