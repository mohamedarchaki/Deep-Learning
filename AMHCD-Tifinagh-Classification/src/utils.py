import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path='reports/figures/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names is not None else "auto",
                yticklabels=class_names if class_names is not None else "auto")
    plt.title('Matrice de confusion (Test set)')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Matrice de confusion sauvegardée sous : {save_path}")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path='reports/figures/loss_accuracy_plot.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Courbe de perte')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Perte')
    ax1.legend()

    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Courbe de précision')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Précision')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"Courbes métriques sauvegardées sous : {save_path}")

def display_classification_report(y_true, y_pred, target_names):
    print("\nRapport de classification (Test set) :")
    print(classification_report(y_true, y_pred, target_names=target_names))