# ChestXpert: AI-Driven Chest X-ray Analysis with LLM Integration

## Overview

ChestXpert is an AI-driven medical imaging project designed to analyze chest X-ray images and generate comprehensive medical reports. It combines deep learning-based image classification with a large language model (LLM) to assist in automated diagnosis and report generation.

## Features

- **Image Classification:** Uses ResNet50 for chest X-ray classification.
- **Fine-Tuned Model:** Layers of ResNet50 were unfrozen for fine-tuning to enhance performance.
- **LLM-Powered Report Generation:** Integrates LLama 3 via Hugging Face to generate a medical report based on classification results.
- **User Prompt Interaction:** The model allows users to input custom prompts to refine the generated report.

## Dataset

The dataset is sourced from Kaggle's ChestXpert dataset, which contains labeled chest X-ray images for training and evaluation.

<img width="933" alt="Screenshot 2025-01-31 at 8 23 46 PM" src="https://github.com/user-attachments/assets/1c9915a3-74ac-4bc3-a93f-227e70e49474" />


## Model Architecture

- **ResNet50:** Pretrained on ImageNet and fine-tuned on the ChestXpert dataset.
- **LLM Integration:** LLama 3 is used for natural language report generation.
- **Hugging Face:** Provides access to the LLM for inference and report creation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChestXpert.git
   cd ChestXpert
   ```
2. Download the dataset from Kaggle https://www.kaggle.com/datasets/ashery/chexpert and place it in the `data/` directory.

## Usage

1. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
2. **Run the Notebook:**
   - Open the `.ipynb` file in Jupyter Notebook.
   - Execute each cell sequentially to train the model, perform inference, and generate medical reports.

## Results

The model provides:

- Accurate classification of chest X-ray images.
- AI-generated reports explaining findings based on the classification results.

## Future Improvements

- Enhance classification accuracy with additional fine-tuning.
- Expand LLM capabilities for more detailed and structured medical reports.
- Develop a web-based interface for easy user interaction.

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

## Acknowledgments

- Kaggle for providing the ChestXpert dataset.
- Hugging Face for the LLM integration.
- The deep learning community for open-source research and tools.

---

For any questions or support, feel free to open an issue or contact me!

