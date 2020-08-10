import torch
import torch.nn as nn
from torch.backends import cudnn
import numpy as np

class NetworkManager():
    """Manage training, validation and testing of a network"""

    def __init__(self, device, net, criterion, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader):
        self.device = device

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        self.logs = {}
    
    def train(self, num_epochs):
        # By default, everything is loaded to cpu
        self.net = self.net.to(self.device)  # This will bring the network to GPU if DEVICE is cuda

        cudnn.benchmark  # Calling this optimizes runtime

        self.logs['train_loss'] = [] # Mean training loss of each epoch (len = NUM_EPOCHS)
        self.logs['train_accuracy'] = [] # Training accuracy of each epoch
        self.logs['val_loss'] = [] # Mean loss on validation set after each epoch
        self.logs['val_accuracy'] = [] # Accuracy on validation set after each epoch

        self.best_net = self.net # Best performing network on the validation set
        self.best_loss = None # Accuracy of best_net on the validation set

        current_step = 0

        for epoch in range(num_epochs): # Iterate over epochs
            print(f"Starting epoch {epoch+1}/{num_epochs}, LR = {self.scheduler.get_lr()}")

            epoch_loss_total = 0
            
            epoch_running_corrects = 0
            epoch_total = 0

            for images, labels in self.train_dataloader: # Iterate over batches
                # Bring data over the device of choice
                images = images.to(self.device)
                labels = labels.to(self.device)

                epoch_total += labels.size(0)

                self.net.train()  # Sets module in training mode

                # PyTorch, by default, accumulates gradients after each backward pass
                # We need to manually set the gradients to zero before starting a new iteration
                self.optimizer.zero_grad()  # Zero-ing the gradients

                # Forward pass to the network
                outputs = self.net(images)

                # Get predictions
                _, preds = torch.max(outputs.data, 1)

                # Update corrects
                epoch_running_corrects += torch.sum(preds == labels.data).data.item()

                # Compute loss based on output and ground truth
                loss = self.criterion(outputs, labels)

                # Log loss
                epoch_loss_total += loss.item()

                # Compute gradients for each layer and update weights
                loss.backward()  # backward pass: computes gradients
                self.optimizer.step()  # update weights based on accumulated gradients

            # Log average loss over epoch
            self.logs['train_loss'].append(epoch_loss_total / len(self.train_dataloader))

            # Log accuracy over epoch
            self.logs['train_accuracy'].append(epoch_running_corrects / float(epoch_total))

            print(f"Loss: {self.logs['train_loss'][-1]}")

            if np.isnan(self.logs['train_loss'][-1]):
                print("Exiting training due to nan loss\n")
                break

            # Validation: after each training epoch, evaluate the model on the validation set
            val_loss, val_accuracy = self.validate()

            self.logs['val_loss'].append(val_loss)
            self.logs['val_accuracy'].append(val_accuracy)

            print(f"Validation loss: {val_loss}, validation accuracy: {val_accuracy}")

            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss < self.best_loss:
                self.best_net = self.net
                self.best_loss = val_loss
                print("Best model updated")

            print("")

            # Step the scheduler
            self.scheduler.step()

    def validate(self):
        self.net.train(False)

        loss_total = 0

        running_corrects = 0
        total = 0

        for images, labels in self.val_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            total += labels.size(0)

            # Forward Pass
            outputs = self.net(images)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

            # Compute loss based on output and ground truth
            loss = self.criterion(outputs, labels)

            # Log loss
            loss_total += loss.item()
        
        # Calculate loss
        loss = loss_total / len(self.val_dataloader)

        # Calculate Accuracy
        accuracy = running_corrects / float(total)

        return loss, accuracy

    def test(self):
        """Test the best performing network (validated on self.val_dataloader)"""

        self.best_net = self.best_net.to(self.device) # this will bring the network to GPU if DEVICE is cuda
        self.best_net.train(False) # Set Network to evaluation mode

        running_corrects = 0
        total = 0

        for images, labels in self.test_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            total += labels.size(0)

            # Forward Pass
            outputs = self.best_net(images)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / float(total)

        print(f"Test Accuracy: {accuracy}")

        return accuracy

    def get_logs(self):
        return self.logs