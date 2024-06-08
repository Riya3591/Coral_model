
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Path to the log file for no adaptation
    path_no_adapt = './logs/log_no_adaptation.pth'

    # Load logs for no adaptation
    log_no_adapt = torch.load(path_no_adapt)

    # Compute mean for each epoch for no adaptation
    no_adapt = {
        'classification_loss': torch.FloatTensor(log_no_adapt['classification_loss']).mean(dim=1).numpy(),
        'coral_loss': torch.FloatTensor(log_no_adapt['CORAL_loss']).mean(dim=1).numpy(),
        'source_accuracy': torch.FloatTensor(log_no_adapt['source_accuracy']).mean(dim=1).numpy(),
        'target_accuracy': torch.FloatTensor(log_no_adapt['target_accuracy']).mean(dim=1).numpy()
    }

    # Add the first 0 value for target_accuracy and source_accuracy
    no_adapt['target_accuracy'] = np.insert(no_adapt['target_accuracy'], 0, 0)
    no_adapt['source_accuracy'] = np.insert(no_adapt['source_accuracy'], 0, 0)

    # Plot the accuracies and losses for no adaptation
    plt.gca().set_color_cycle(['blue', 'green', 'red', 'm'])

    axes = plt.gca()
    axes.set_ylim([0, 1.1])

    l2, = plt.plot(no_adapt['target_accuracy'], label="test acc. w/o coral loss", marker='.')
    l4, = plt.plot(no_adapt['source_accuracy'], label="training acc. w/o coral loss", marker='+')

    plt.legend(handles=[l2, l4], loc=4)

    fig_acc = plt.gcf()
    plt.show()
    plt.figure()

    fig_acc.savefig('accuracies_no_adapt.pdf', dpi=1000)

    # Plot classification loss and CORAL loss for training w/o CORAL loss
    plt.gca().set_color_cycle(['red', 'blue'])

    axes = plt.gca()
    axes.set_ylim([0, 0.5])

    l9, = plt.plot(no_adapt['classification_loss'], label="classification loss w/o coral loss", marker='*')
    l7, = plt.plot(no_adapt['coral_loss'], label="distance w/o coral loss", marker='.')

    plt.legend(handles=[l9, l7], loc=1)

    fig_acc = plt.gcf()
    plt.show()
    plt.figure()
    fig_acc.savefig('losses_no_adapt.pdf', dpi=1000)

if __name__ == '__main__':
    main()