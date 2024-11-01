# To-Do List
* This file documents the features plan to be added to main py file. I will provide update for each bullet points once implemented.
## Feature Engineering
* How to properly deal with timestamp (seansonality?)
* How to include category features like state that has ~40 distinct values. 
## Data Splitting
* Add a validation set.
## Optimizer
* Considering other optimizers, ex.Adam
## Overfitting Management Strategies
* Considering early stoppage.
* Monitor tendency of overfitting from Validationset, visualize loss of Validation & Train.
* Overfit the model first, then address overfitting.
* Prevent Overfitting
    * Weight Decay in optimizer, ex. optim.SGD(model.parameters(), lr = 0.0004, weight_decay = 1e-4)
    * Add a dropout layer after activation function, ex. nn.Dropout(p=0.3)
## Model Initialization and Normalization Techniques
* Kaiming Initialization
    * Set the initial weights of layers in deep neural networks. Help prevent vanishing or exploding gradients.
* Batch Normalization:
    * Batch normalization is a technique that normalizes the output of a layer for each mini-batch, which helps stabilize the learning process. By reducing internal covariate shifts, batch normalization allows for faster training and can enable the use of higher learning rates. It also introduces a slight regularization effect, potentially reducing the need for other regularization techniques.


