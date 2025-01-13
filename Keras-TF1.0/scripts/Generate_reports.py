from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools



def acc_loss_analysis(path_toSave_report, history, Extension):

    print("[INFO]..... Analysing the Model Performance..")
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path_toSave_report + 'Acc_Progress'+ Extension, dpi = 300)
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path_toSave_report +"Loss_Analysis" + Extension, dpi = 300)
    plt.show()
    

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
    plt.colorbar()
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
