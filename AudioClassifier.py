import gc
import torch
import numpy as np
from ClassifierNet import ClassifierNet
from AudioData import AudioData
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch import nn


class AudioClassifier:
    __model = ClassifierNet()
    __is_trained = False

    def __init__(self, use_pretrained=False):
        if use_pretrained:
            self.__model = torch.load("./pretrained_model/model")
            self.__is_trained = True

    def train(self, data_df, data_folder,
              device=torch.device("cpu"),
              num_epochs=75,
              batch_size=50,
              learning_rate=0.001,
              shuffle=True,
              criterion=nn.CrossEntropyLoss(),
              optimizer=None,
              save_model_to=None,
              keep_data_in_ram=False):
        train_data = AudioData(data_folder, data_df, 'wav_path')
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
        self.__model.to(device)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.__model.parameters(),
                                         lr=learning_rate)
        acc_list = []
        for epoch in trange(num_epochs, desc=f"Epochs:"):
            for i, (images, labels) in enumerate(
                    tqdm(train_loader,
                         leave=False,
                         desc=f"Epoch {epoch + 1} progress: ")):
                # Run the forward pass
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.__model(images)
                loss = criterion(outputs, labels)

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)
        self.__is_trained = True
        print("Model trained successfully")
        if save_model_to is not None:
            torch.save(self.__model, save_model_to)
            print(f"Model saved to {save_model_to}")
        if not keep_data_in_ram:
            del train_data
            del train_loader
        return acc_list

    def predict(self, data_df, data_folder,
                device=torch.device("cpu"),
                n_splits=2,
                batch_size=50,
                shuffle=True):
        predictions = []
        for i in trange(n_splits, desc="Splits: "):
            valid_data = AudioData(data_folder, data_df, 'wav_path')
            valid_loader = DataLoader(valid_data,
                                      batch_size=batch_size,
                                      shuffle=shuffle)
            self.__model.eval()
            with torch.no_grad():
                for images, labels in valid_loader:
                    images = images.to(device)
                    outputs = self.__model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.append(predicted.cpu().numpy())
            del valid_data
            del valid_loader
            gc.collect()
        predictions = np.hstack(predictions)
        return predictions

