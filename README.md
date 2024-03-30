# GANs Architectures

This repository contains implementations of different Generative Adversarial Networks (GANs) architectures. The purpose of this repository is to provide a comprehensive collection of GANs architectures along with case studies and code examples to facilitate understanding and experimentation.

## About

GANs are a class of deep learning models that consist of two neural networks, a generator and a discriminator, which compete against each other to generate realistic data. This repository explores various GANs architectures and their applications in generating different types of data, such as images and sequences.

## Repository Structure

The repository is organized into the following directories:

1. **Vanilla GANs**: This directory contains the implementation of Vanilla GANs, the foundational architecture of GANs. It includes a Jupyter notebook with a case study on generating handwritten digits using Vanilla GANs. The directory also contains three Python scripts: `loss.py`, `models.py`, and `training_helper.py`, which provide the necessary components for training and evaluating Vanilla GANs.

2. **DCGANs**: This directory focuses on Deep Convolutional GANs (DCGANs), an extension of Vanilla GANs that incorporates convolutional layers for improved image generation. It includes a Jupyter notebook with a case study on generating faces using DCGANs. Similar to the Vanilla GANs directory, it also contains the `loss.py`, `models.py`, and `training_helper.py` scripts specific to DCGANs.

## Blog Posts

For a detailed explanation of the theory and implementation of these GANs architectures, please refer to the following blog posts:

1. [Understanding Vanilla GANs: Theory and Implementation](https://roberto-verdugo-blog.vercel.app/posts/001-vanilla-gans)
2. [Deep Convolutional GANs (DCGANs): Theory and Implementation](https://roberto-verdugo-blog.vercel.app/posts/002-dc-gans)

These blog posts provide a comprehensive overview of each GANs architecture, including the underlying concepts, network architectures, training procedures, and case studies.

## Getting Started

To get started with the code in this repository, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/GANs__Architectures.git`
2. Navigate to the desired GANs architecture directory: `cd Vanilla_GANs` or `cd DCGANs`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Jupyter notebook for the case study: `jupyter notebook`
5. Explore the code and experiment with different configurations and datasets.

## Dependencies

The code in this repository requires the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

Please refer to the `requirements.txt` file in each GANs architecture directory for the specific versions of the dependencies.

## Contributing

Contributions to this repository are welcome! If you have any improvements, bug fixes, or new GANs architectures to add, please submit a pull request. Make sure to follow the existing code style and provide appropriate documentation.

## License

This project is licensed under the [MIT License](link-to-license-file). Feel free to use the code for educational and research purposes.

## Acknowledgements

The code and case studies in this repository are inspired by various research papers and open-source implementations. We would like to acknowledge their contributions to the field of generative models and GANs.

## Contact

If you have any questions, suggestions, or feedback, please feel free to reach out:

- Email: verdugo.rds@gmail.com
- Blog: [Average AI Student blog](https://roberto-verdugo-blog.vercel.app/)

Happy exploring and generating with GANs!
