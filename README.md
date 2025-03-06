# TripRecs: A Landscape Classifier, Recommendation, and Search Engine

## Overview
TripRecs is a deep-learning-powered landscape classification and recommendation system that helps users identify and explore travel destinations. By leveraging computer vision, Apache Solr, and deep learning, it enables:
- **Image-based location identification** using CNN models
- **Personalized travel recommendations** based on similar landscapes
- **Real-time travel fare lookup** using a search engine interface

## Features
- **Landscape Image Classification**: Identifies locations from user-uploaded images using deep learning models (Custom CNN, ResNet-18, VGG-16, MobileNet-v3-Small).
- **Recommendation System**: Suggests visually similar landscapes for alternative travel options.
- **Search Engine**: Allows keyword-based location search with Apache Solr indexing over 1.1M travel records.
- **Travel Fare Lookup**: Fetches estimated trip costs to different destinations.
- **Web Interface**: Streamlit-powered UI for seamless interaction.

## Dataset & Preprocessing
- Uses **Google Landmarks v2 Dataset** (~4M images across 30K landmarks).
- **Data Cleaning**: Downsampled images to optimize storage and performance.
- **Feature Engineering**: Applied metadata analysis, data augmentation, and format standardization.

## Model Development
- **Custom CNN Model**: 5 convolutional layers, batch normalization, dropout, and ReLU activations.
- **Transfer Learning**: ResNet-18, VGG-16, MobileNet-v3-Small with frozen pretrained weights.
- **Training**: Optimized using PyTorch, trained with **Distributed Data Parallel (DDP)** on Falcon HPC Cluster.
- **Performance**:
  - Custom CNN: 51% accuracy
  - ResNet-18: 70% accuracy
  - VGG-16: 73% accuracy
  - MobileNet-v3-Small: 73% accuracy (lightweight & faster)

## Distributed Training Setup
- Implemented PyTorch **Distributed Data Parallel (DDP)** training.
- Ran training on **2 Nvidia A100 GPUs** via Slurm job scheduler.
- Observed a **50% reduction in training time** compared to single-node training.

## Deployment
- **Model Export**: Trained models saved in `.pt` format for fast inference.
- **Web App**: Built using **Streamlit** for intuitive UI/UX.
- **Search Engine**: Apache Solr for real-time query handling and ranking.

## Usage
1. **Image Upload**: Upload a landscape image for classification and recommendations.
2. **Search by Keywords**: Find locations using keyword-based search.
3. **Check Travel Fares**: Look up flight or travel cost estimates.

## Technologies Used
- **Deep Learning**: PyTorch, Transfer Learning, CNNs
- **Big Data & Processing**: Apache Spark, Solr, Distributed Training
- **Cloud & Infrastructure**: Falcon HPC, Nvidia A100 GPUs
- **Web & APIs**: Streamlit, FastAPI
- **Data Storage**: Google Landmarks Dataset, Solr Indexing

## Future Improvements
- Expand dataset for better classification coverage.
- Optimize search engine ranking for more relevant results.
- Deploy as a scalable cloud-native application.

## Contributors
- **Dilochan Karki**: Distributed Training, Backend Integration, Apache Solr Setup.
- **Yunik Tamrakar**: Data Collection, CNN Model, Web UI Development.

## License
This project is released under [MIT License].

