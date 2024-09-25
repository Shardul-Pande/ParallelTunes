# ParallelTunes: Music Genre Classification with Parallel Computing

## Overview
ParallelTunes is a project focused on classifying music genres using deep learning techniques. The project aims to enhance the efficiency of training models on large datasets by leveraging parallel computing methods. This repository explores the use of multiple GPUs, Distributed Data Parallelism (DDP), and Automatic Mixed Precision (AMP) to speed up the model training process for music genre classification.

## Features
- **Music Genre Classification**: Classifies songs into 10 different music genres.
- **Parallelization**: Utilizes multiple GPUs for faster training.
- **Distributed Data Parallelism (DDP)**: Synchronizes updates across devices to scale training.
- **Automatic Mixed Precision (AMP)**: Optimizes computational efficiency by mixing precision (16-bit and 32-bit) training.

## Project Architecture
This project uses a combination of Python scripts and Jupyter notebooks for data processing, model training, and analysis. The architecture includes:

- **Config.py**: Defines the path to the dataset.
- **Data.py**: Reads and processes audio files using the `librosa` library.
- **Set.py**: Splits the data into training, testing, and validation sets.
- **Model.py**: Defines the Convolutional Neural Network (CNN) model.
- **Train_ddp.py**: Implements training using Distributed Data Parallelism.
- **Train_amp.py**: Implements training using Automatic Mixed Precision.
- **Get_genre.py**: Predicts the genre of a given audio file.
- **EDA.ipynb**: Explores the dataset using Exploratory Data Analysis (EDA).
- **DataPreprocessing.ipynb**: Applies Principal Component Analysis (PCA) to preprocess the data.

## Dataset
The dataset used for this project is the [GTZAN dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification), a popular benchmark dataset for music genre classification. It contains 1000 audio files categorized into 10 genres:
- Metal
- Disco
- Classical
- Hip-hop
- Jazz
- Country
- Pop
- Blues
- Reggae
- Rock

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Shardul-Pande/ParallelTunes.git
   cd ParallelTunes
   ```

2. Load Anaconda environment:
   ```bash
   module load anaconda3/2022.05
   source activate cuda_env
   ```

3. Train the model:
   - To train with DDP:
     ```bash
     python3 train_ddp.py
     ```
   - To train with AMP:
     ```bash
     python3 train_amp.py
     ```

4. Predict the genre:
   ```bash
   python3 get_genre.py filepath
   ```

## Parallelization Techniques
The project employs various parallelization techniques to speed up the training process:
- **Multiple GPU Utilization**: Distributed training across 1, 2, or 4 GPUs.
- **DDP**: Divides data across devices and synchronizes updates to efficiently scale the model.
- **AMP**: Combines 16-bit and 32-bit precision to enhance training speed while maintaining accuracy.

## Results
- **Speedup**: Using multiple GPUs reduced training time significantly.
- **Batch Size Optimization**: Batch sizes of 64 and 128 were tested to balance memory usage and computational efficiency.
- **Memory Efficiency**: Further improvements in memory allocation are required to optimize performance fully.
  
## Hardware Specifications
- **GPU**: Nvidia Tesla P100-PCIE-12GB (up to 12 GPUs)
- **Memory**: 100 GB Reservation
- **CPUs**: 12 cores

## Results and Analysis
- The experiments showed that using more GPUs doesn't always boost performance, and there is a tradeoff between batch size, number of GPUs, and memory consumption.
- PCA was applied to reduce the dimensionality of the audio data, making the classification task more manageable.

## Conclusion
ParallelTunes demonstrates that parallel computing methods can significantly reduce the time required for deep learning tasks in audio analysis. By employing multiple GPUs and optimizing the training process with DDP and AMP, we were able to achieve faster convergence and improved memory efficiency. However, balancing batch size and GPU usage remains crucial for optimal performance.

## References
- [How to Apply Machine Learning and Deep Learning Methods to Audio Analysis](https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc)
- [Exploring Different Approaches for Music Genre Classification](https://www.sciencedirect.com/science/article/pii/S1110866512000151)
- [Music Genre Classification and Recommendation by Using Machine Learning Techniques](https://ieeexplore.ieee.org/document/8554016)

Feel free to explore the code, test the models, and experiment with different parallelization techniques to further improve the speed and accuracy of music genre classification!
