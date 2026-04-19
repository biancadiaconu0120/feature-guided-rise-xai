Interpreting Deep Neural Networks: Explaining the Methods Behind AI-Generated Image Recognition


Table of Contents

1.Summary

2.Project Description

2.1 The "Black Box" Problem

2.2 Experimental Approach

3.Repository Guide

3.1 Repository Structure

3.2 File Descriptions

4.How to Reproduce the Experiment

4.1 System Requirements

4.2 Setup and Installation

4.3 Running the Notebook

5.Key Findings: The "Dual Strategy"

6.Acknowledgments

7.Citation

1. Summary

This project thesis, "Interpreting Deep Neural Networks," was created as the basis for my Master's Thesis. The goal of this project was to create a framework for interpreting a Convolutional Neural Network (CNN) based discriminator, trained to classify images as either real or synthetically generated. High-resolution human portrait images were chosen for this task. A custom discriminator was trained from scratch against a pre-trained, frozen StyleGAN generator using the FFHQ dataset. The output of the model is a probability score on the authenticity of the input image, and the core of this research is the application of Explainable AI (XAI) techniques to understand the model's decision-making process.

This README.md file provides a comprehensive overview of the project and the contents of this repository. While the thesis report (Master_thesis_report.pdf) describes why certain decisions were made and presents the final results, this README explains how the experiment was conducted and how to reproduce it using the provided code.

2. Project Description

This section provides a high-level overview of the project's motivation and the experimental design. For a complete theoretical background and in-depth analysis, please refer to the full thesis report.

2.1 The "Black Box" Problem

Generative Adversarial Networks (GANs), and particularly StyleGAN, have achieved state-of-the-art performance in synthesizing high-fidelity images. However, their internal mechanisms, especially those of the discriminator, operate as a "black box." This lack of transparency is a significant barrier to model validation, debugging, and trustworthiness. This research aims to open this black box.

2.2 Experimental Approach

To interpret the discriminator, a controlled experiment was designed to isolate its learning process.

Isolating the Discriminator: A pre-trained, high-performance StyleGAN generator was used as a source of synthetic images. Crucially, this generator's weights were frozen, meaning it produced a consistent and unchanging distribution of fake images.

Training from Scratch: A new discriminator network was designed and trained from scratch to distinguish between real images (from the FFHQ dataset) and the synthetic images from the frozen generator.

Interpretation: With a fully trained and high-performing discriminator, gradient-based XAI techniques (Grad-CAM and Saliency Maps) were applied to visualize and analyze its learned features.

This setup ensures that any patterns discovered are a result of the discriminator's learning process, not an artifact of a co-adapting generator.

3. Repository Guide

3.1 Repository Structure
.
├── Master_thesis_report.pdf
├── Interpreting Deep Neural Networks.pptx
├── CODE.ipynb
└── README.md

3.2 File Descriptions

Master_thesis_report.pdf: The complete, final version of the thesis report. This document contains the full theoretical background, literature review, in-depth methodology, results, and discussion.

Interpreting Deep Neural Networks.pptx: The final PowerPoint presentation used for the master's defense.

CODE.ipynb: A Jupyter Notebook containing the complete Python code for the experiments. This includes the discriminator architecture, the data loading and preprocessing pipeline, the training loop, and the implementation of the XAI visualization techniques.

README.md: This documentation file.

4. How to Reproduce the Experiment

This section provides a step-by-step guide to setting up the environment and running the code to reproduce the results presented in the thesis.

4.1 System Requirements

Python 3.9+

PyTorch 1.10+

TorchVision

CUDA-enabled GPU (highly recommended for performance)

Additional libraries: NumPy, Matplotlib, scikit-learn, OpenCV-Python, Pillow

4.2 Setup and Installation

1. Clone the Repository
git clone [https://gitlab.oth-regensburg.de/EI/Labore/ses-labor/iai/interpretationgan/-/tree/main/Master_thesis%20Interpreting%20deep%20neural%20networks:%20Explaining%20the%20methods%20behind%20AI-generated%20image%20recognition]
cd [Master_thesis Interpreting deep neural networks: Explaining the methods behind AI-generated image recognition]

2. Install Dependencies
It is highly recommended to use a virtual environment (e.g., venv or conda).

-Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate

-Install the required Python packages

pip install torch torchvision numpy matplotlib scikit-learn opencv-python Pillow

3. Download Required Data and Models
Due to their large size, the dataset and pre-trained generator model are not included in this repository. You will need to download them separately:

FFHQ Dataset: The Flickr-Faces-HQ dataset can be downloaded from this Dataset link:https://www.kaggle.com/datasets/tommykamaz/faces-dataset-small

Pre-trained StyleGAN Generator: The official pre-trained StyleGAN model weights (.pt file) used in this project can be found in this dataset link:https://www.kaggle.com/datasets/songseungwon/ffhq-1024x1024-pretrained

4.3 Running the Notebook

The CODE.ipynb Jupyter Notebook is the primary file for running the experiment.

Update File Paths: Open CODE.ipynb in a Jupyter environment. In the initial cells, you will find variables for file paths. Update these paths to point to the locations where you have saved the FFHQ dataset and the pre-trained StyleGAN generator weights.

Execute Cells Sequentially: Run the cells in the notebook in order. The notebook is structured to perform the following sequence of operations:

Import libraries and define the network architecture.

Load the FFHQ dataset and the pre-trained generator.

Execute the discriminator training loop for 100 epochs.

Generate and display the quantitative results (loss curves, confusion matrix, ROC curve).

Generate and display the qualitative XAI visualizations (Grad-CAM and Saliency Maps) on both individual and averaged images.

5. Key Findings: The "Dual Strategy"

The central discovery of this research is the "Dual Strategy" employed by the StyleGAN discriminator:

Semantic Location (The "Where"): The model first uses a high-level understanding of facial structure to identify semantically meaningful regions, primarily the forehead/hairline and the central facial triangle (eyes and nose).

Textural Scrutiny (The "What"): Within these identified regions, the model then performs a fine-grained, pixel-level analysis to detect subtle textural artifacts and inconsistencies that are characteristic of the GAN generation process.

This two-step process demonstrates a more complex reasoning strategy than previously understood, moving beyond simple feature detection to a hierarchical, context-aware analysis.

6. Acknowledgments

I would like to express my sincere gratitude to my supervisors, Prof. Dr.-Ing. Johannes Reschke and Prof. Dr.-Ing. Armin Sehr, for their invaluable guidance, support, and mentorship throughout this research project. I would also like to thank the Ostbayerische Technische Hochschule (OTH) Regensburg for providing the resources and academic environment necessary to complete this work.

7. Citation

If you use this work for research purposes, please cite it as follows:

Tandel, N. (2025). Interpreting deep neural networks: Explaining the methods behind AI-generated image recognition (Master's Thesis). Ostbayerische Technische Hochschule (OTH) Regensburg, Regensburg, Germany.