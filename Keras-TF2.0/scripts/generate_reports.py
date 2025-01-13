from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def gen_report(testY, predIdxs, lb, H, EPOCHS, report_dir):
    

    # show a nicely formatted classification report
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))

    # serialize the model to disk
   
    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(report_dir + 'plot.png')

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def report_N_matrix(save_results_to, y_pred, y_test, categories, Extension):

    print("[INFO]... Computing the Classification Report and Confusion Matrix....")

    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    np.set_printoptions(precision=2)

    class_rep = classification_report(y_test, y_pred.round(), target_names=categories)
    print(class_rep)

    f = open(save_results_to + 'report.txt', 'w')
    f.write('Title\n\nClassification Report\n\n{}\n\n\nTitle\n\nConfusion Matrix\n\n{}\n\n'.format(class_rep, cnf_matrix))
    f.close()

    # Compute confusion matrix

    # Plot non-normalized confusion matrix

    print("[INFO].... Saving the Results...")
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=categories,
                          title='Confusion matrix, without normalization')
    plt.savefig(save_results_to + 'Confusion_matrix_without_Normalization'+ Extension, dpi = 300)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=categories, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(save_results_to + 'Confusion_matrix_with_Normalization' + Extension, dpi = 300)
    plt.show()