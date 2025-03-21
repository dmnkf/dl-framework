# Fine-Tuning Pre-Trained ECG and CMR Encoders with a Modular Pipeline and Comprehensive Evaluation for Cardiovascular Diagnostics
> Authors: Dominik Filliger & Noah Leuenberger (2025)

This repository provides a modular deep-learning pipeline for ECG and CMR representation learning, building on the work of Turgut et al. (2025) and extending the MMCL-ECG-CMR approach. Designed with reproducibility, extensibility, and adaptability in mind, the pipeline enables systematic experimental configurations while integrating preprocessing, training, and postprocessing workflows. Its modular architecture supports easy customization, allowing seamless integration of new components without disrupting the core framework. Accompanied by comprehensive technical documentation, the pipeline facilitates efficient adoption and future extensions for diverse research applications in the field of cardiovascular diagnostics. In the creation of this framework and all related code of this repository, AI coding assistants were used.

## 🚀 Quick Start

1. **Install Rye**
   To install Rye, visit the [official website](https://rye.astral.sh/guide/installation/) and follow the instructions. For Unix-based systems, you can use the following command:

   ```bash
   curl -sSf https://rye.astral.sh/get | bash
   ```

2. **Setup Project**
   ```bash
   # Clone repository
   git clone https://github.com/IPOLE-BAT-CMR-ECG/dl-framework.git
   cd dl-framework

   # Install dependencies
   rye sync --no-lock

   # Pull project data (requires access request from team)
   dvc pull  # This will set up the exact data configuration used in our work
   ```

3. **Configure Environment**
   ```bash
   cp sample.env .env
   # Edit .env with your API keys for Weights & Biases / Neptune
   ```

## 📚 Documentation

Comprehensive documentation is available covering:
- Detailed installation and setup guides
- Data pipeline documentation
- Model architectures and training
- Analysis and visualization tools
- Development guidelines

The documentation can be viewed by running the following command:

```bash
rye run docs
```

Afterwards open the following link in your browser: [http://localhost:8000](http://localhost:8000)

## 📥 Fine-tuned Models

Download our fine-tuned model weights from [SWITCHdrive](https://drive.switch.ch/index.php/s/ipmDfMIio2K3E7z).

## 📖 Research

For more details about our research:

- [**Thesis**](docs/thesis.pdf)
- [**Poster**](docs/poster.png)

## 🛠️ Basic Usage

```bash
# Preprocess data
rye run preprocess data=ecg

# Train model
rye run train experiment=fine_tuning/ecg_arryhthmia

# Generate embeddings
rye run generate_embeddings experiment=embeddings/ecg_arryhthmia
```

For detailed usage instructions and advanced features, please refer to our [documentation](docs/index.md).

## 🙏 Acknowledgments

This work builds upon:
- [MMCL-ECG-CMR](https://github.com/oetu/MMCL-ECG-CMR) by Turgut et al. (2025)
- The research paper: ["Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI"](http://arxiv.org/abs/2308.05764)

If you use this code in your research, please cite the original paper:

```bibtex
@article{turgut2025unlocking,
  title={Unlocking the diagnostic potential of electrocardiograms through information transfer from cardiac magnetic resonance imaging},
  author={Turgut, {\"O}zg{\"u}n and M{\"u}ller, Philip and Hager, Paul and Shit, Suprosanna and Starck, Sophie and Menten, Martin J and Martens, Eimo and Rueckert, Daniel},
  journal={Medical Image Analysis},
  pages={103451},
  year={2025},
  publisher={Elsevier}
}
```
