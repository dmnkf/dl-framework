# Model Artifacts Generation

This guide explains how to generate and analyze model artifacts using our artifact generation pipeline. These artifacts help understand and visualize how models process and interpret data.

## Overview

The artifact generation system provides three main types of outputs:

1. **Embeddings**: Feature vectors extracted from the model's encoder
2. **GradCAM Visualizations**: Class activation maps showing which regions influence model decisions
3. **Attention Maps**: Visualizations of the model's attention patterns (for transformer-based models)

## Basic Usage

```bash
# Generate all artifacts for a model
rye run generate_artifacts experiment=embeddings/my_experiment \
    ckpt_path=path/to/model.ckpt

# Generate artifacts for specific splits
rye run generate_artifacts experiment=embeddings/my_experiment \
    ckpt_path=path/to/model.ckpt splits=[train,val]

# Generate artifacts for a specific record
rye run generate_artifacts experiment=embeddings/my_experiment \
    ckpt_path=path/to/model.ckpt record_id=patient_123
```

## Configuration

Artifact generation is configured through Hydra, similar to training. Configurations are stored in `configs/experiment/embeddings/`:

```yaml
# configs/experiment/embeddings/default.yaml
defaults:
  - model: mae_vit
  - data: cmr_acdc
  - paths: default

model:
  # Model-specific settings
  backbone: vit_base_patch16
  
data:
  # Data loading settings
  batch_size: 32
  num_workers: 4

output_dir: ${paths.artifact_dir}/${experiment}
```

## Artifact Types

### Embeddings

Embeddings are feature vectors extracted from the model's encoder layer. They represent high-level features learned by the model and can be used for:

- Similarity analysis
- Clustering
- Downstream tasks
- Model interpretation

Embeddings are saved as PyTorch tensors and can be loaded using:

```python
embeddings = torch.load('path/to/embeddings.pt')
```

### GradCAM Visualizations

GradCAM (Gradient-weighted Class Activation Mapping) highlights regions in the input that are important for the model's predictions. This is particularly useful for:

- Understanding model decisions
- Identifying relevant features
- Validating model behavior
- Debugging model predictions

GradCAM images are generated for each class and saved as PNG files.

### Attention Maps

For transformer-based models, attention maps visualize how different parts of the input attend to each other. These visualizations help:

- Understand attention patterns
- Analyze model behavior
- Debug attention mechanisms
- Validate model architecture

Attention maps are saved as high-resolution PNG files with multiple subplots showing different attention heads.

## Output Structure

Generated artifacts are organized as follows:

```text
artifact_dir/
├── embeddings/
│   ├── train_embeddings.pt
│   ├── val_embeddings.pt
│   └── test_embeddings.pt
├── gradcam/
│   └── patient_123/
│       ├── class_0.png
│       ├── class_1.png
│       └── ...
└── attention_maps/
    └── patient_123_attention.png
```

## Advanced Usage

### Customizing Output Directory

```bash
rye run generate_artifacts experiment=embeddings/my_experiment \
    output_dir=path/to/custom/dir
```

### Generating Specific Artifact Types

```bash
# Only generate embeddings
rye run generate_artifacts experiment=embeddings/my_experiment \
    generate_embeddings=true generate_gradcam=false generate_attention=false

# Only generate GradCAM for specific classes
rye run generate_artifacts experiment=embeddings/my_experiment \
    generate_embeddings=false generate_gradcam=true \
    target_classes=[0,1]
```

### Memory Management

For large datasets or memory-intensive models:

```bash
# Process with smaller batch size
rye run generate_artifacts experiment=embeddings/my_experiment \
    data.batch_size=16

# Generate artifacts for splits separately
rye run generate_artifacts experiment=embeddings/my_experiment \
    splits=[train]
rye run generate_artifacts experiment=embeddings/my_experiment \
    splits=[val,test]
```