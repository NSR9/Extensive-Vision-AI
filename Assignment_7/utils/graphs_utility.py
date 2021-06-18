import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_graphs(train_losses, test_losses, train_accuracy, test_accuracy):
    sns.set(style='whitegrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (20, 10)

    # Plot the learning curve.
    fig, (plt1, plt2) = plt.subplots(1, 2)
    plt1.plot(np.array([x / 1000 for x in train_losses]), 'r', label="Training Loss")
    plt1.plot(np.array(test_losses), 'b', label="Validation Loss")
    plt2.plot(np.array(train_accuracy), 'r', label="Training Accuracy")
    plt2.plot(np.array(test_accuracy), 'b', label="Validation Accuracy")

    plt2.set_title("Training-Validation Accuracy Curve")
    plt2.set_xlabel("Epoch")
    plt2.set_ylabel("Accuracy")
    plt2.legend()
    plt1.set_title("Training-Validation Loss Curve")
    plt1.set_xlabel("Epoch")
    plt1.set_ylabel("Loss")
    plt1.legend()

    plt.show()
