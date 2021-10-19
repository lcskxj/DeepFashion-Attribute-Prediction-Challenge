# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from read_data import DeepFashionDataset
from network import MultiLabelsNetwork
from config import *
import numpy as np


data_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(CROP_SIZE),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transform_test = transforms.Compose([
    transforms.Resize(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# train_loader = torch.utils.data.DataLoader(
#     DeepFashionDataset(data_type="train", transform=data_transform_train),
#     batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
# )

val_loader = torch.utils.data.DataLoader(
    DeepFashionDataset(data_type="val", transform=data_transform_test),
    batch_size=TEST_BATCH_SIZE, num_workers=1, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    DeepFashionDataset(data_type="test", transform=data_transform_test), shuffle=False,
    batch_size=TEST_BATCH_SIZE, num_workers=1, pin_memory=True
)


# build a model
# model = MultiLabelsNetwork().cuda(GPU_ID)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.BCELoss()


def train(model, learning_rate, batch_size):

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(
    DeepFashionDataset(data_type="train", transform=data_transform_train),
    batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data = Variable(data.cuda(GPU_ID))

        optimizer.zero_grad()

        output = model(data)

        # rearrange label
        target = torch.stack(target).T
        target_shape = list(target.size())
        target_attribute = np.zeros((target_shape[0], sum(CATEGORIES)))
        for batch_pos, one_target in enumerate(target):
            # print(one_target)
            for attr_idx, attr in enumerate(one_target):
                attr_label = attr.item()
                # print("attr label:", attr_label)

                attr_pos = 0
                for i in range(attr_idx):
                    attr_pos += CATEGORIES[i]
                attr_pos = attr_pos + attr_label
                # print("attr pos:", attr_pos)

                target_attribute[batch_pos][attr_pos] = 1
                # print(target_attribute)

        target_attribute = torch.Tensor(target_attribute).cuda(GPU_ID)
        # print(target_attribute)
        loss = criterion(output, target_attribute)
        loss.backward()
        optimizer.step()


def evaluate():
    model.eval()

    correct_number = 0
    for batch_idx, (data, target_attrs) in enumerate(val_loader):
        data = Variable(data.cuda(GPU_ID))
        output = model(data)

        # print("==="*20)
        attr_list = []
        label_start_pos = 0
        output = output[0].cpu().detach().numpy()
        for i, num_classes in enumerate(CATEGORIES):
            label_end_pos = label_start_pos + CATEGORIES[i]

            one_hot_pred = output[label_start_pos:label_end_pos]
            pred = np.argmax(one_hot_pred)
            target = target_attrs[i]
            attr_list.append(pred)
            # print("idx", i, ", pred", pred, ", target", target)
            if pred == target:
                correct_number += 1

            label_start_pos = label_end_pos

    acc = correct_number / (len(val_loader) * 6)

    model.train()

    return acc


def test(model, epoch):
    model.eval()
    print("Testing the images...")
    text_file = open("multi-labels_models/multi_label_test_attr_{}.txt".format(epoch), "w")

    # well_trained_model = MultiLabelsNetwork()
    # well_trained_model.load_state_dict(torch.load("multi-labels_models/multi-label.h5"))
    # well_trained_model.eval().cuda(GPU_ID)

    # test on val dataset
    correct_number = 0
    for batch_idx, (data, target_attrs) in enumerate(val_loader):
        data = Variable(data.cuda(GPU_ID))
        output = model(data)

        # print("==="*20)
        attr_list = []
        label_start_pos = 0
        output = output[0].cpu().detach().numpy()
        for i, num_classes in enumerate(CATEGORIES):
            label_end_pos = label_start_pos + CATEGORIES[i]

            one_hot_pred = output[label_start_pos:label_end_pos]
            pred = np.argmax(one_hot_pred)
            target = target_attrs[i]
            attr_list.append(pred)
            # print("idx", i, ", pred", pred, ", target", target)
            if pred == target:
                correct_number += 1

            label_start_pos = label_end_pos

    acc = correct_number / (len(val_loader) * 6)
    print("Evaluate on val dataset, acc: ", acc)


    for data in test_loader:
        data = Variable(data.cuda(GPU_ID))
        output = model(data)

        # print("==="*20)
        attr_list = []
        label_start_pos = 0
        output = output[0].cpu().detach().numpy()
        for i, num_classes in enumerate(CATEGORIES):
            label_end_pos = label_start_pos + CATEGORIES[i]

            one_hot_pred = output[label_start_pos:label_end_pos]
            pred = np.argmax(one_hot_pred)

            attr_list.append(pred)
            # print("idx", i, ", pred", pred, ", target", target)

            label_start_pos = label_end_pos

        # print(attr_list)
        for attr in attr_list:
            text_file.write(str(attr) + " ")
        text_file.write("\n")

    text_file.close()
    print("Saved results for test set.")
    model.train()
    return acc


if __name__ == "__main__":
    # print(model)

    # for learning_rate in [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]:
    for learning_rate in [1e-4]:
        for batch_size in [64]:
        # for batch_size in [32, 64, 128]:
            model = MultiLabelsNetwork().cuda(GPU_ID)

            # main training process
            best_acc = 0
            accuracy_list = []
            for epoch in range(1, EPOCH + 1):
                print("******" * 20)
                print("Training Epoch:", epoch)
                train(model, learning_rate, batch_size)

                if epoch % 1 == 0:
                    accuracy = test(model, epoch)
                    print("Epoch:", epoch, ", Validation acc", accuracy)
                    accuracy_list.append(accuracy)

                    if accuracy > best_acc:
                        best_acc = accuracy
                    # if accuracy > 0.81:
                    #     break

            torch.save(model.state_dict(), "multi-labels_models/multi-label.h5")
            print(learning_rate, batch_size, best_acc)
            print(accuracy_list)
            # test()
