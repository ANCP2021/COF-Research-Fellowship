# Network Attack Detection using Machine Learning Models

## **Overview**
This repository contains research work focusing on the detection of network attacks using various machine learning models. The research evaluates the effectiveness of different classifiers in identifying Distributed Denial-of-Service (DDoS) attacks and explores preprocessing techniques applied to the CIC-DDoS2019 dataset.

## **Authors**
- Alexander J. Nemecek 
- Chad Mourning 

## **Abstract**
During the Covid-19 pandemic, many institutions made the switch to be fully online. This upwards trend in connectivity positively correlates with attacks on Internet of Things devices. A distributed denial-of-service (DDoS) attack is an attempt to interrupt or disable network traffic by overwhelming a system or network, preventing users from accessing resources. Machine learning (ML) is a promising approach for the detection of DDoS attacks. This study evaluates a range of classifiers, including Deep Neural Network (DNN), AdaBoost, K-Nearest Neighbor, XGBoost, Decision Tree, Random Forest, Support Vector Machine, Linear and Quadratic Discriminant Analysis, Logistic Regression, Stochastic Gradient Descent, and Naïve Bayes to assess accuracy and identify the best-performing classifiers. Preprocess techniques were utilized and applied to the CIC-DDoS2019 dataset including feature selection, manipulation, normalization, and reduction to split the data into training and testing datasets. The dataset includes packet captures of benign and common DDoS attacks. Eleven of the ML models achieved an accuracy of 90% or higher, with DNN and AdaBoost exhibiting the highest accuracies. The results hold promise of reducing parameters and size comparable to the original dataset while resulting in similar accuracy measures. The findings have implications for the potential of ML improving the detection and prevention of DDoS attacks. However, the study has limitations such as, the data do not mimic all types of DDoS attacks which occur in real-world environments. Future research could explore alternative ML algorithms, feature selection approaches, and different evaluation metrics to enhance the detection of network attacks in the cybersecurity community.

## **Directory Structure**
- /models
    - Contains files related to running the specified classifiers.
- /preprocessing
    - Contains files used for data preprocessing.
- DDoS 2019 | Datasets | Research | Canadian Institute for Cybersecurity | UNB.pdf
    - Information relating to the dataset structure and usage.

## **Limitations & Future Work**
We acknowledge limitations, particularly regarding the dataset's representation of all real-world DDoS attacks. Suggestions for future research include exploring alternative ML algorithms, different feature selection approaches, and diverse evaluation metrics to enhance network attack detection in cybersecurity.

## **Acknowledgments**
This work was conducted as part of the Choose Ohio First Scholar Fellowship by The Ohio Department of Higher Education and Ohio University.
