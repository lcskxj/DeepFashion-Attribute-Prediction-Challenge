# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from config import *
from read_data import DeepFashionDataset
from network import MultiHeadsNetwork


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

# test on test set
test_loader = torch.utils.data.DataLoader(
    DeepFashionDataset(data_type="test", transform=data_transform_test), shuffle=False,
    batch_size=TEST_BATCH_SIZE, num_workers=1, pin_memory=True
)


# build a model
# model = MultiHeadsNetwork().cuda(GPU_ID)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss()


def train(model, learning_rate, batch_size):

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        DeepFashionDataset(data_type="train", transform=data_transform_train),
        batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data = Variable(data.cuda(GPU_ID))

        optimizer.zero_grad()

        outputs = model(data)

        loss = 0
        for idx in range(len(CATEGORIES)):

            output = outputs[idx]

            target_attribute = target[idx].cuda(GPU_ID)

            category_loss = criterion(output, target_attribute)
            loss += category_loss

        loss.backward()
        optimizer.step()


def evaluate():
    model.eval()

    # well_trained_model = MultiHeadsNetwork()
    # well_trained_model.load_state_dict(torch.load("multi-heads_models/multi-head.h5"))
    # well_trained_model.eval().cuda(GPU_ID)

    # text_file = open("multi-heads_val_attr.txt", "w")
    correct_number = 0
    for batch_idx, (data, target_attrs) in enumerate(val_loader):
        data = Variable(data.cuda(GPU_ID))
        outputs = model(data)

        # print("==="*20)
        attr_list = []
        for idx in range(len(CATEGORIES)):
            # print(outputs[idx].data)
            pred = outputs[idx].data.max(1, keepdim=True)[1].item()
            target = target_attrs[idx].item()
            attr_list.append(pred)
            # print("idx", idx, ", pred", pred, ", target", target)
            if pred == target:
                correct_number += 1

    #     for attr in attr_list:
    #         text_file.write(str(attr) + " ")
    #     text_file.write("\n")
    #
    # text_file.close()
    acc = correct_number / (len(val_loader) * 6)

    model.train()

    return acc


def test(epoch):
    # print("Load well-trained model...")
    model.eval()
    # well_trained_model = MultiHeadsNetwork()
    # well_trained_model.load_state_dict(torch.load("multi-heads_models/multi-head.h5"))
    # well_trained_model.eval().cuda(GPU_ID)

    correct_number = 0
    for batch_idx, (data, target_attrs) in enumerate(val_loader):
        data = Variable(data.cuda(GPU_ID))
        outputs = model(data)

        # print("==="*20)
        attr_list = []
        for idx in range(len(CATEGORIES)):
            # print(outputs[idx].data)
            pred = outputs[idx].data.max(1, keepdim=True)[1].item()
            target = target_attrs[idx].item()
            attr_list.append(pred)
            # print("idx", idx, ", pred", pred, ", target", target)
            if pred == target:
                correct_number += 1

    #     for attr in attr_list:
    #         text_file.write(str(attr) + " ")
    #     text_file.write("\n")
    #
    # text_file.close()
    acc = correct_number / (len(val_loader) * 6)
    print("Evaluate on val dataset, acc: ", acc)


    print("Testing the images...")
    text_file = open("multi-heads_models/multi_head_test_attr_{}.txt".format(epoch), "w")

    for data in test_loader:
        data = Variable(data.cuda(GPU_ID))
        outputs = model(data)
        attr_list = []
        for idx in range(len(CATEGORIES)):
            pred = outputs[idx].data.max(1, keepdim=True)[1].item()
            attr_list.append(pred)

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
            model = MultiHeadsNetwork().cuda(GPU_ID)
            best_acc = 0
            accuracy_list = []
            for epoch in range(1, EPOCH + 1):
                print("******" * 20)
                print("Training Epoch:", epoch)
                train(model, learning_rate, batch_size)

                if epoch % 1 == 0:
                    accuracy = test(epoch)
                    print("Epoch:", epoch, ", Validation acc", accuracy)
                    accuracy_list.append(accuracy)

                    if accuracy > best_acc:
                        best_acc = accuracy
            # if accuracy > 0.81:
            #     break
    
            torch.save(model.state_dict(), "multi-heads_models/multi-head.h5")
            print(learning_rate, batch_size, best_acc)
            print(accuracy_list)

    # test()

    # print(evaluate())
