# To-Do List
* This file documents the features plan to be added to main py file. I will provide update for each bullet points once implemented.
## Feature Engineering
- [ ] How to properly deal with timestamp (seansonality?)
- [ ] How to include category features like state that has ~40 distinct values. 
## Data Splitting
- [ ] **Add a validation set.**
## Optimizer
- [ ] Considering other optimizers, ex.Adam
## Overfitting Management Strategies
Monitor tendency of overfitting from Validationset, visualize loss of Validation & Train.

Overfit the model first, then address overfitting.

Prevent Overfitting
- [ ] Considering early stoppage.
- [ ] Weight Decay in optimizer, ex. optim.SGD(model.parameters(), lr = 0.0004, weight_decay = 1e-4)
- [ ] Add a dropout layer after activation function, ex. nn.Dropout(p=0.3)
## Model Initialization and Normalization Techniques
- [x] Kaiming Initialization (2024-10-31: Completed)
* Set the initial weights of layers in deep neural networks. Help prevent vanishing or exploding gradients.
- [ ] Batch Normalization:
* In deep networks, as the parameters of the network change during training, the distributions of the inputs to each layer also change. This phenomenon is known as internal covariate shift. Batch normalization reduces this shift by normalizing the inputs to a layer, which helps maintain a consistent mean and variance, thus stabling the learning process.


