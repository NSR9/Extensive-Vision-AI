import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_accuracy_curves(final_train_loss_values, final_test_loss_values, final_train_accuracy_values, final_test_accuracy_values):
    sns.set(style='whitegrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (40, 10)

    # Plot the learning curve.
    fig, (plt1, plt2, plt3, plt4) = plt.subplots(1, 4)
    plt1.plot(np.array(final_train_loss_values[0]), 'r', label="Group Norm Training Loss")
    plt1.plot(np.array(final_train_loss_values[1]), 'b', label="Layer Norm Training Loss")
    plt1.plot(np.array(final_train_loss_values[2]), 'g', label="Batch Norm + L1 Training Loss")
    plt1.plot(np.array(final_train_loss_values[3]), 'c', label="Group Norm + L1 Training Loss")
    plt1.plot(np.array(final_train_loss_values[4]), 'y', label="Layer Norm + L1 Training Loss")
    plt1.plot(np.array(final_train_loss_values[5]), 'm', label="Batch Norm + L1 + L2 Training Loss")

    plt2.plot(np.array(final_test_loss_values[0]), 'r', label="Group Norm Test Loss")
    plt2.plot(np.array(final_test_loss_values[1]), 'b', label="Layer Norm Test Loss")
    plt2.plot(np.array(final_test_loss_values[2]), 'g', label="Batch Norm + L1 Test Loss")
    plt2.plot(np.array(final_test_loss_values[3]), 'c', label="Group Norm + L1 Test Loss")
    plt2.plot(np.array(final_test_loss_values[4]), 'y', label="Layer Norm + L1 Test Loss")
    plt2.plot(np.array(final_test_loss_values[5]), 'm', label="Batch Norm + L1 + L2 Test Loss")

    # plt1.plot(np.array(test_loss_values), 'b', label="Validation Loss")

    plt3.plot(np.array(final_train_accuracy_values[0]), 'r', label="Group Norm Train Accuracy")
    plt3.plot(np.array(final_train_accuracy_values[1]), 'b', label="Layer Norm Train Accuracy")
    plt3.plot(np.array(final_train_accuracy_values[2]), 'g', label="Batch Norm + L1 Train Accuracy")
    plt3.plot(np.array(final_train_accuracy_values[3]), 'c', label="Group Norm + L1 Train Accuracy")
    plt3.plot(np.array(final_train_accuracy_values[4]), 'y', label="Layer Norm + L1 Train Accuracy")
    plt3.plot(np.array(final_train_accuracy_values[5]), 'm', label="Batch Norm + L1 + L2 Train Accuracy")

    plt4.plot(np.array(final_test_accuracy_values[0]), 'r', label="Group Norm Test Accuracy")
    plt4.plot(np.array(final_test_accuracy_values[1]), 'b', label="Layer Norm Test Accuracy")
    plt4.plot(np.array(final_test_accuracy_values[2]), 'g', label="Batch Norm + L1 Test Accuracy")
    plt4.plot(np.array(final_test_accuracy_values[3]), 'c', label="Group Norm + L1 Test Accuracy")
    plt4.plot(np.array(final_test_accuracy_values[4]), 'y', label="Layer Norm + L1 Test Accuracy")
    plt4.plot(np.array(final_test_accuracy_values[5]), 'm', label="Batch Norm + L1 + L2 Test Accuracy")

    plt4.set_title("Test Accuracy Curve")
    plt4.set_xlabel("Epoch")
    plt4.set_ylabel("Accuracy")
    plt4.legend()
    plt3.set_title("Training Accuracy Curve")
    plt3.set_xlabel("Epoch")
    plt3.set_ylabel("Accuracy")
    plt3.legend()
    plt2.set_title("Validation Loss Curve")
    plt2.set_xlabel("Epoch")
    plt2.set_ylabel("Loss")
    plt2.legend()
    plt1.set_title("Training Loss Curve")
    plt1.set_xlabel("Epoch")
    plt1.set_ylabel("Loss")
    plt1.legend()

    plt.show()


def wrong_predictions(test_loader, model, device):
    import torch
    wrong_images = []
    wrong_label = []
    correct_label = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])

            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

        fig = plt.figure(figsize=(8, 10))
        fig.tight_layout()
        for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
            img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
            ax = fig.add_subplot(5, 2, i + 1)
            ax.axis('off')
            ax.set_title(f'\nactual {target.item()}\npredicted {pred.item()}', fontsize=10)
            ax.imshow(img.squeeze())

        plt.show()