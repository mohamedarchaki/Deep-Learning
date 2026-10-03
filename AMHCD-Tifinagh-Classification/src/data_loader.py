import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Lit une image, la convertit en niveaux de gris, la redimensionne en 32x32,
    la normalise dans [0, 1] et l'aplatit en vecteur de taille 1024.
    """
    assert os.path.exists(image_path), f"Image introuvable : {image_path}"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Impossible de lire : {image_path}"
    
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten()

def prepare_dataset(data_dir):
    """
    Parcourt recursivement data_dir pour recuperer toutes les images
    et leurs classes correspondantes.
    """
    print(f"Recherche des images dans : {data_dir}")
    
    image_paths = []
    labels = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

    # Parcours récursif pour trouver toutes les images
    for root, dirs, files in os.walk(data_dir):
        # Exclure les dossiers cachés s'il y en a
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(root, file)
                # Le nom du dossier parent sert de label (nom du caractere ou id)
                label = os.path.basename(root)
                image_paths.append(full_path)
                labels.append(label)

    labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    
    if labels_df.empty:
        raise FileNotFoundError(
            f"Aucune image n'a ete detectee dans '{data_dir}'. "
            "Assurez-vous d'avoir dezippe l'archive AMHCD directement dans le dossier 'data/raw/'."
        )

    print(f"Images trouvees : {len(labels_df)}")
    print(f"Classes distinctes trouvees : {labels_df['label'].nunique()}")

    # Encodage des labels
    label_encoder = LabelEncoder()
    labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
    num_classes = len(label_encoder.classes_)

    # Chargement et vectorisation
    print("Pretraitement des images en cours...")
    X = np.array([load_and_preprocess_image(p) for p in labels_df['image_path']])
    y = labels_df['label_encoded'].values

    # Decoupage stratifie : 60% Train, 20% Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    # Conversion One-Hot
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = np.array(one_hot_encoder.fit_transform(y_train.reshape(-1, 1)))
    y_val_oh = np.array(one_hot_encoder.transform(y_val.reshape(-1, 1)))
    y_test_oh = np.array(one_hot_encoder.transform(y_test.reshape(-1, 1)))

    return {
        'X_train': X_train, 'y_train': y_train_oh,
        'X_val': X_val, 'y_val': y_val_oh,
        'X_test': X_test, 'y_test': y_test,
        'label_encoder': label_encoder,
        'num_classes': num_classes
    }