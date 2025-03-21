# Embedding Space Visualization

This guide explains how to visualize and analyze embedding spaces using UMAP projections. The tools provided help you understand how embeddings evolve during training and how different groups of data are distributed in the embedding space.

## Core Visualization Functions

### UMAP Projection

::: src.visualization.embedding_viz
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true
        members: [run_umap]
        show_docstring_description: true

### Global Visualization

::: src.visualization.embedding_viz
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true
        members: [plot_global_umap_grid]
        show_docstring_description: true

### Group Analysis Tools

::: src.visualization.embedding_viz
    options:
        show_root_heading: true
        heading_level: 3
        show_source: true
        members: [prepare_group_data, overlay_group_on_embedding]
        show_docstring_description: true

## Example Usage

### Basic Embedding Comparison

```python
# Prepare embeddings
pretrained_embeddings = ...  # shape: (N, D)
finetuned_embeddings = ...   # shape: (N, D)

# Create UMAP projections
umap_pretrained = run_umap(pretrained_embeddings)
umap_finetuned = run_umap(finetuned_embeddings)

# Visualize
umaps_dict = {
    'Pre-trained': umap_pretrained,
    'Fine-tuned': umap_finetuned
}
plot_global_umap_grid(umaps_dict, metadata_df)
```

### Group Analysis

```python
# Analyze a specific disease group
target_group = "Atrial Fibrillation"
df_group_full, df_group_sub = prepare_group_data(metadata_df, target_group)

# Visualize group distribution
fig, ax = overlay_group_on_embedding(umap_coords, metadata_df, df_group_sub)
plt.title(f"{target_group} Distribution")
plt.show()
```

## Best Practices

1. **Consistency**: Use the same UMAP parameters (metric, random_state) when comparing different embeddings.
2. **Sampling**: For large datasets, consider using `prepare_group_data` to sample a manageable subset.
3. **Visual Clarity**: 
      - Use appropriate alpha values for background points
      - Choose distinct colors for different groups
      - Add legends and titles for clear interpretation

## Advanced Customization

The visualization functions are designed to be flexible:

- Modify color schemes by adjusting the `highlight_color` and background colors
- Customize marker styles for different types of samples
- Adjust figure sizes and grid layouts for different numbers of embeddings
- Add additional metadata overlays or annotations
