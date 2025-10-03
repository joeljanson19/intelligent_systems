# %% [markdown]
# ## Intelligent Systems - Assignment 2
# #### Joel Janson (ist1116925)
# Corresponding GitHub: [https://github.com/joeljanson19/intelligent_systems.git](https://github.com/joeljanson19/intelligent_systems.git)

# %% [markdown]
# ## Neural Network

# %%
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas

# %%
# CHOOSE DATASET

# 1. Regression
# diabetes = datasets.load_diabetes(as_frame=True)
# X = diabetes.data.values
# y = diabetes.target.values

# 2. Classification
diabetes = datasets.fetch_openml(name="diabetes", version=1, as_frame=True)
X = diabetes.data.values
y = diabetes.target.astype(str).map({'tested_positive': 1, 'tested_negative': 0}).values

# %%
#train test spliting
test_size=0.2
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

# %%
# Standardize features
scaler=StandardScaler()
Xtr= scaler.fit_transform(Xtr)
Xte= scaler.transform(Xte)

# %% [markdown]
# ### NN Architecture
# The architecture can be tuned by changing the number of layers, layer size and regularization (dropout). Dropout prevents overfitting by randomly "dropping out" (setting to zero) a fraction of the neurons during training, which forces the network to learn more robust and generalized features. Regularization is determined later together with other hyperparameters.  
#   
# It can be very hard to determine an optimal architecture but I ended up using the following:  
# **Regression**: 3 hidden layers with 64 neurons each.  
# **Classification**: 4 hidden layers with 64 neurons in the first three and 32 neurons in the last.

# %%
class MLP(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_prob=0.5):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, output_size)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        x = self.out(x)
        return x

# %% [markdown]
# ### Hyperparameters
# - **`num_epochs`** – Number of training passes over the entire dataset. Don't want too many epochs to avoid overfitting to noise.  
# - **`lr`** – Learning rate. Step size for updating weights during training. Controls how fast the model learns.  
# - **`dropout`** – Fraction of neurons randomly dropped during training to reduce overfitting. In this task I'm using 10% but for toy datasets, higher fraction could be used.  
# - **`batch_size`** – Number of samples processed before updating the model. If too much RAM is being used, this value could for example be dropped to 32.  
#   
# After tuning the model, the following values were chosen for the different datasets:  
# **Regression**: `num_epochs`=60, `lr`=0.001, `dropout`=0.1, `batch_size`=64  
# **Classification**: `num_epochs`=75, `lr`=0.001, `dropout`=0.1, `batch_size`=64  
#   
# Important to note is that these values are not necessarily fully optimized. Training the model over and over with the same hyperparameters can give very different performance.

# %%
num_epochs=75 
lr=0.001
dropout=0.1
batch_size=64 

# %%
Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
yte = torch.tensor(yte, dtype=torch.float32)

# Wrap Xtr and ytr into a dataset
train_dataset = TensorDataset(Xtr, ytr)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %%
# Model, Loss, Optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# Ignoring this line since I'm not using cuda

model = MLP(input_size=Xtr.shape[1], dropout_prob=dropout)#.to(device)
criterion = nn.BCEWithLogitsLoss()  # for binary classification
# criterion = nn.MSELoss() # for regression
optimizer = optim.Adam(model.parameters(), lr=lr) #can use different optimizer such as AdamW but not necessary

# %%
# Training loop
for epoch in range(num_epochs):
    model.train() #train or evolve
    epoch_loss = 0.0

    for batch_x, batch_y in train_dataloader:
        batch_x = batch_x#.to(device)
        batch_y = batch_y#.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y.view(-1, 1))

        optimizer.zero_grad()
        loss.backward() #directly related to the forward function defined above
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# %% [markdown]
# We print `mean_squared_error` and `accuracy_score` as indications of the performance of the model for regression and classification respectively.

# %%
y_pred=model(Xte)
# Performance metric for regression
# print(f'MSE:{mean_squared_error(yte.detach().numpy(),y_pred.detach().numpy())}') 

# Performance metric for classification
print(f'ACC:{accuracy_score(yte.detach().numpy(),y_pred.detach().numpy()>0.5)}') 


# %% [markdown]
# ## Discussion  
# 
# For ``Dataset 1 (regression)``, the following MSE scores were obtained using the different models:  
# - **LS**:    2545.29  
# - **ANFIS**: 2491.41  
# - **NN**:    2818.01  
# 
# We can see that the Hybrid ANFIS model achieved the lowest MSE, slightly outperforming the least squares model from assignment 1. This is because ANFIS combines least squares with gradient descent to adjust both the antecedent (fuzzy membership) and consequent parameters. The Neural Network model performed worst, likely due to insufficient tuning, leading to underfitting or poor generalization. For this dataset, a simpler model with fuzzy structure may be more effective.
# 
# For ``Dataset 2 (classification)``, the following accuracy scores were obtained using the different models:  
# - **LS**:    0.7532  
# - **ANFIS**: 0.7857  
# - **NN**:    0.7792  
# 
# All models performed reasonably well, with ANFIS achieving the highest accuracy, suggesting that the fuzzy clustering and rule-based structure captured class boundaries effectively. The neural network performed slightly worse than ANFIS but better than LS. In order to outperform ANFIS, it would probably require more careful tuning of architecture and hyperparameters. 


