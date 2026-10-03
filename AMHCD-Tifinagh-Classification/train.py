import os
from src.data_loader import prepare_dataset
from src.model import MultiClassNeuralNetwork
from src.utils import plot_confusion_matrix, plot_metrics, display_classification_report

def main():
    # 1. Chemin vers les donnees
    data_dir = os.path.join(os.getcwd(), 'data', 'raw')
    
    # Verification si le sous-dossier specifique existe
    alt_dir = os.path.join(data_dir, 'amhcd-data-64', 'tifinagh-images')
    if os.path.exists(alt_dir):
        data_dir = alt_dir

    print(f"Chargement des donnees depuis : {data_dir}")
    data = prepare_dataset(data_dir)

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    label_encoder = data['label_encoder']
    num_classes = data['num_classes']

    print(f"Train samples : {X_train.shape[0]} | Val samples : {X_val.shape[0]} | Test samples : {X_test.shape[0]}")
    print(f"Nombre de classes detectees : {num_classes}")

    # 2. Configuration de l'architecture (32x32=1024 -> 64 -> 32 -> 33)
    layer_sizes = [X_train.shape[1], 64, 32, num_classes]
    learning_rate = 0.01
    epochs = 100
    batch_size = 32

    print("\nInitialisation et debut de l'entrainement...")
    model = MultiClassNeuralNetwork(layer_sizes=layer_sizes, learning_rate=learning_rate, lambda_reg=0.0)

    # 3. Entrainement
    train_losses, val_losses, train_accuracies, val_accuracies = model.train(
        X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size
    )

    # 4. Evaluation sur le Test set
    print("\nEvaluation sur l'ensemble de test...")
    y_pred = model.predict(X_test)

    # 5. Visualisations et Rapports
    display_classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    plot_confusion_matrix(y_test, y_pred, class_names=label_encoder.classes_)
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

if __name__ == '__main__':
    main()