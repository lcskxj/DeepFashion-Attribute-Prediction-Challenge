
# Project Report for Assignment 1 for *Deep Learning for Data Science (CE7454 Course)*


## Abstract
Recently, deep learning has been applied in a wide range of fields. Convolutional Neural Networks (CNN) are commonly used to analyze visual content, like images and videos. Considering its powerful performance on vision tasks, CNN models also have been applied in the fashion industry. DeepFashion dataset,  a large-scale fashion image database, has been treated as the benchmark for fashion recognition tasks. In this work, I solved the attribute identification task on a subset of the DeepFashion Category and Attribute Prediction Benchmark. I designed two neural network structures based on ResNet50 to solve this task. To further improve the performance, I adopted several model optimization and data regularization techniques in the training process. Experimental results show that the combination of the designed model structure and the optimization methods is efficient for the attribute identification task and the final classification accuracy reaches above 80\%.



## Model structures

### Multi-head 

![Multi-head model stracture](https://github.com/lcskxj/DeepFashion-Attribute-Prediction-Challenge/blob/main/figs/1.png)

### Multi-label

![Multi-label model stracture](https://github.com/lcskxj/DeepFashion-Attribute-Prediction-Challenge/blob/main/figs/2.png)




## Main experimental results

![Results for multi-label model when using different network size](https://github.com/lcskxj/DeepFashion-Attribute-Prediction-Challenge/blob/main/figs/256_1.png)
![Results for multi-head model when using different network size](https://github.com/lcskxj/DeepFashion-Attribute-Prediction-Challenge/blob/main/figs/256_2.png)
![Performance with and without data augmentation](https://github.com/lcskxj/DeepFashion-Attribute-Prediction-Challenge/blob/main/figs/256_1_2.png)
![Training progresses of different training methods](https://github.com/lcskxj/DeepFashion-Attribute-Prediction-Challenge/blob/main/figs/256_1_3.png)

## Conclusion
This work solves the DeepFashion Attribute Prediction Challenge by proposing two neural network structures based on ResNet50. These two neural networks are built through regarding the attribute prediction task as six multi-class and a multi-label classification task, separately. To improve the performance, I applied several model optimization and data regularization techniques in the training progress. Experimental results show that proposed models can predict fashion attributes with high accuracy. 


## How to run our code?
- Create a virtual python environment
- Install the requirement packages in the `requirements.txt`
- Specify the GPU cards that can be used in your machine at the head of the `config.py` file you are going to run (e.g., `multi_label_train.py`).
- Run the target `.py` file and observe the results.
