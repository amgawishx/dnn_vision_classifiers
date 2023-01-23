from functools import reduce
from matplotlib.pyplot import subplots
from sklearn import metrics as m
from numpy import argmax
    
def get_performance_graphs(model_histories):
    def add(a,b, key): return a.history[key]+b.history[key]
    figs, axes = [], []
    for model_history in model_histories:
        loss_values = reduce(model_history[0], 
                            model_history[1], 'loss')
        val_loss_values = reduce(model_history[0], 
                            model_history[1], 'val_loss')
        accuracy = reduce(model_history[0], 
                            model_history[1], 'accuracy')
        val_accuracy = reduce(model_history[0], 
                            model_history[1], 'val_accuracy')
        epochs = range(1, len(loss_values) + 1)
        fig, ax = subplots(1, 2, figsize=(14, 6))
        # Plot the model accuracy vs Epochs
        ax[0].plot(epochs, accuracy, 'r', label='Training accuracy')
        ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        ax[0].set_title('Training & Validation Accuracy', fontsize=16)
        ax[0].set_xlabel('Epochs', fontsize=16)
        ax[0].set_ylabel('Accuracy', fontsize=16)
        ax[0].legend()
        ax[0].grid("on")
        # Plot the loss vs Epochs
        ax[1].plot(epochs, loss_values, 'r', label='Training loss')
        ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
        ax[1].set_title('Training & Validation Loss', fontsize=16)
        ax[1].set_xlabel('Epochs', fontsize=16)
        ax[1].set_ylabel('Loss', fontsize=16)
        ax[1].legend()
        ax[1].grid("on")
        axes.append(ax)
        figs.append(fig)
    return figs, axes

def get_confusion_matrices(models, test_data):
    displays = []
    for model in models:
        y_pred = [argmax(img) for img in model.predict(test_data)]
        confusion_matrix = m.confusion_matrix(test_data.labels,y_pred)
        cm_display = m.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
        displays.append(cm_display)
    return displays
