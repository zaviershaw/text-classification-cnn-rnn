
import matplotlib.pyplot as plt
# vis=visdom.Visdom(env='show')

epochs = []
steps = []
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

with open('./tensorboard/textcnn/log-conv_lstm.csv') as f:
    for line in f.readlines():
        l = line.split(' ')
        epoch = float(l[0])
        epochs.append(epoch)

        step = float(l[1])
        steps.append(step)

        train_loss = float(l[2])
        train_losses.append(train_loss)

        train_accuracy = float(l[3])
        train_accuracies.append(train_accuracy)

        val_loss = float(l[4])
        val_losses.append(val_loss)

        val_accuracy = float(l[5])
        val_accuracies.append(val_accuracy)

        plt.figure()
        plt.plot(steps,train_losses,label = 'train_loss')
        plt.plot(steps,val_losses,label = 'val_loss')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Loss')
        plt.legend()
        plt.savefig('./images/conv_lstm_Loss',dpi = 200)
        plt.show()

        plt.figure()
        plt.plot(steps,train_accuracies,label = 'train_accuracy')
        plt.plot(steps,val_accuracies,label = 'val_accuracy')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig('./images/conv_lstm_Accuracy',dpi = 200)
        plt.show()


