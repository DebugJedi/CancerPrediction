import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

def get_clean_data():
    data = pd.read_csv("Datasets/data.csv")
    # print(data.columns)
    data = data.drop(columns=['Unnamed: 32', 'id'], axis= 1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data


def columns_max_min():
    data = get_clean_data()
    columns = {}
    cols = list(data)
    for col in cols:
        if col != 'diagnosis':
            max_value = data[col].max()
            min_value = data[col].mean()
            columns[col] = (max_value, min_value)
    return columns

def preprocess(data):

    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']
    
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2, 
                                                        random_state = 32)
    #Scale the data
    scaler = StandardScaler()
    X_train_trans = scaler.fit_transform(X_train)
    
    # convert data to PyTorch tensors
    X_train_trans = torch.tensor(X_train_trans, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)

    return X_train_trans, X_test, y_train, y_test, scaler

    # print("The shape is: ************")
    # print(X_train_trans.shape)

#Neural Network Architecture

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size ):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)    
        return out 
        
    # Model building

def create_model(X_train, y_train, input_size, hidden_size, output_size, learning_rate, num_epoch):

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.BCELoss()
    optimzer = optim.Adam(model.parameters(), lr= learning_rate)

    for epoch in range(num_epoch):
        model.train()
        optimzer.zero_grad()
        outputs = model(X_train)
        loss  = criterion(outputs, y_train.view(-1,1))
        loss.backward()
        optimzer.step()

        # Calculate accuracy
        with torch.no_grad():
            predicted = outputs.round()
            correct = (predicted==y_train.view(-1,1)).float().sum()
            accuracy = correct/y_train.size(0)
        if (epoch+1) % 10 ==0:
            print(f"Epoch [{epoch+1}/{num_epoch}], Loss {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}% ")
    return model 

def model_train_evaluation(model, X_train, y_train):
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
        predicted = outputs.round()
        correct = (predicted==y_train.view(-1,1)).float().sum()
        accuracy = correct/y_train.size(0)
        print(f"Accuracy with training data: {accuracy.item() *100:.2f}%")
 
def model_test_evaluation(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = outputs.round()
        correct = (predicted==y_test.view(-1,1)).float().sum()
        accuracy = correct/y_test.size(0)
        print(f"Accuracy with test data: {accuracy.item() *100:.2f}%")
        print(f"Sklearn Accuracy with test data: {accuracy_score(y_test.view(-1,1), predicted)*100:.2f}")
        print(f"Sklearn Classification report: \n  {classification_report(y_test.view(-1,1), predicted)}")

def transform_test(scaler,X_test):
    X_test_trans = scaler.transform(X_test)
    X_test_trans = torch.tensor(X_test_trans, dtype=torch.float32).to(device)
    # y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    return X_test_trans

def save_model(model, scaler):

    with open('resources/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('resources/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def main():
    
    data = get_clean_data()
     
    X_train_trans, X_test, y_train, y_test, scaler = preprocess(data)
    input_size = X_train_trans.shape[1]
    hidden_size = 54
    output_size = 1
    learning_rate = 0.001
    num_epoch = 100
    model = create_model(X_train_trans, y_train, input_size, 
                 hidden_size, output_size, learning_rate, num_epoch)
    
    X_test_trans = transform_test(scaler, X_test)

    model_train_evaluation(model, X_train_trans, y_train)

    model_test_evaluation(model, X_test_trans, y_test)
    # save_model(model, scaler)
    
        


if __name__ == '__main__':
    main()

    