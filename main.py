from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
from transformers import AdamW
from transformers import BertModel
from io import BytesIO
# from transformers import BertTokenizer, BertModel
from transformers import LongformerModel, LongformerTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



#data preprocessing
df = pd.read_parquet('./problem1.parquet')
df.to_csv('problem1.csv')
df = pd.read_csv('problem1.csv')


# Load Longformer model and tokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')


# dialogue embedding into 786-d feature
def process_dialogue(dialogue):
    inputs = tokenizer(dialogue, return_tensors='pt', padding='max_length', truncation=True, max_length=4096) # tokenize and convert text to input tensors
    with torch.no_grad(): # generate BERT embeddings for the entire dialogue
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy() # extract the embedding for the [CLS] token
    return cls_embedding

embeddings_file = 'bert_embeddings_long.npy'
if os.path.exists(embeddings_file):
    embeddings = np.load(embeddings_file) 
    print('embedding loaded.')
else:
    embeddings_list = df['Text'].apply(process_dialogue).tolist()
    embeddings = torch.cat([torch.tensor(embedding) for embedding in embeddings_list], dim=0)
    np.save(embeddings_file, embeddings)
    print('embedding saved.')


# training/validation sets
train_embeddings, val_embeddings = train_test_split(embeddings, test_size=0.3, shuffle=False) 
train_embeddings = torch.tensor(train_embeddings).detach().requires_grad_(False)
val_embeddings = torch.tensor(val_embeddings).detach().requires_grad_(False)

supports = torch.tensor(df['Yt'].values).detach().requires_grad_(False)  # Y_t
targets = torch.tensor(df['Yt+1'].values).detach().requires_grad_(False)  # Y_t+1
train_targets, val_targets = train_test_split(targets, test_size=0.3, shuffle=False)
train_supports, val_supports = train_test_split(supports, test_size=0.3, shuffle=False)


# build model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(768, 64)  # BERT embedding dimension to hidden layer 1
        self.fc2 = nn.Linear(64, 16)  # hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(16, 2)  # hidden layer 2 to output layer
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the model, loss function, optimizer, and hyper-parameters
model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5


# Train the model
train_losses = []
val_losses = []
val_losses2 = []

for epoch in range(epochs):
    #training
    optimizer.zero_grad()
    para = model(train_embeddings) #shape (n,2)
    a, b = para[:,0].mean(), para[:,1].mean() #we take mean of from training data as hat{a} and hat{b}
    outputs = a + b*train_supports
    loss = criterion(outputs, train_targets)
    print(f'epoch:{epoch+1}/{epochs}: train loss:{loss}')
    train_losses.append(loss.item())  
    loss.backward()
    optimizer.step()

    #validation
    with torch.no_grad():
        model.eval()
        
        #use model-outputed a and b
        para = model(val_embeddings)
        a_, b_ = para[:,0].mean(), para[:,1].mean()
        outputs = a_ + b_*val_supports
        loss = criterion(outputs, val_targets)
        val_losses.append(loss.item()) 
        print(f'epoch:{epoch+1}/{epochs}: val loss (with new generated a and b):{loss}')
        
        #use a and b generated during the training session
        outputs = a + b*val_supports
        loss = criterion(outputs, val_targets)
        val_losses2.append(loss.item()) 
        print(f'epoch:{epoch+1}/{epochs}: val (with a and b from training session) loss:{loss}')


# Plotting and saving the loss
plt.plot(train_losses, label='Train Loss (with a_train & b_train)')
plt.plot(val_losses2, label='Validation Loss (with a_train & b_train)')
plt.plot(val_losses, label='Validation Loss (with a_val & b_val)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.xticks(range(epochs))  # Set the ticks to 1, 2, 3, 4, 5
plt.legend()
plt.savefig('loss_plot.png')  # Save the figure as a PNG file



