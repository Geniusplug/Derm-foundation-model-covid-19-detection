import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
import tensorflow as tf
import time
from tqdm import trange, tqdm

BATCH_SIZE = 64
EPOCHS = 2
IMG_SIZE = (224, 224)
SEED = 42
DATASET_DIR = 'Dataset'
RESULT_DIR = 'Result'
CUSTOM_MODEL_DIR = 'CustomClassifier'
MODELS = ['GoogleDermModel', 'VGG19', 'Resnet50']

def load_dataset(dataset_dir):
    data = []
    labels = []
    for label in ['Covid', 'Normal']:
        class_dir = os.path.join(dataset_dir, label)
        for fname in os.listdir(class_dir):
            if fname.endswith('.png'):
                data.append(os.path.join(class_dir, fname))
                labels.append(label)
    return np.array(data), np.array(labels)

def augment_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    aug_iter = datagen.flow(x, batch_size=1)
    aug_img = next(aug_iter)[0].astype('uint8')
    return aug_img

def preprocess_images(filepaths, augment=False):
    images = []
    for path in filepaths:
        if augment:
            img = augment_image(path)
        else:
            img = load_img(path, target_size=IMG_SIZE, color_mode='rgb')
            img = img_to_array(img)
        images.append(img)
    return np.array(images)

def get_feature_extractor(model_name):
    if model_name == 'VGG19':
        base = VGG19(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        preprocess = vgg_preprocess
    elif model_name == 'Resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        preprocess = resnet_preprocess
    elif model_name == 'GoogleDermModel':
     
        base = VGG19(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        preprocess = vgg_preprocess
    else:
        raise ValueError('Unknown model')
    model = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))
    return model, preprocess

def extract_features(model, preprocess, images):
    images = preprocess(images)
    features = model.predict(images, batch_size=BATCH_SIZE, verbose=1)
    return features

def build_custom_classifier(input_dim, n_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)
    log_file = os.path.join(RESULT_DIR, 'visual_results.log')
    with open(log_file, 'w') as log:
        X, y = load_dataset(DATASET_DIR)
        class_names = np.unique(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        X_train_img = preprocess_images(X_train, augment=True)
        X_val_img = preprocess_images(X_val, augment=False)
        y_train_enc = to_categorical([np.where(class_names == label)[0][0] for label in y_train], num_classes=len(class_names))
        y_val_enc = to_categorical([np.where(class_names == label)[0][0] for label in y_val], num_classes=len(class_names))
        results = []
        for model_name in MODELS:
            print(f'\n--- Evaluating {model_name} ---')
            log.write(f'\n--- Evaluating {model_name} ---\n')
            feature_extractor, preprocess = get_feature_extractor(model_name)
            X_train_feat = extract_features(feature_extractor, preprocess, X_train_img)
            X_val_feat = extract_features(feature_extractor, preprocess, X_val_img)
            clf = build_custom_classifier(X_train_feat.shape[1], len(class_names))
            callbacks = [
                ModelCheckpoint(os.path.join(CUSTOM_MODEL_DIR, f'{model_name}_best.h5'), save_best_only=True, monitor='val_accuracy', mode='max'),
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            ]
            epoch_times = []
            history_acc = []
            history_val_acc = []
            history_loss = []
            history_val_loss = []
            print(f"Training {model_name}...")
            log.write(f"Training {model_name}...\n")
            for epoch in trange(EPOCHS, desc=f"Training {model_name}", unit="epoch"):
                start_time = time.time()
                hist = clf.fit(X_train_feat, y_train_enc, validation_data=(X_val_feat, y_val_enc),
                               epochs=1, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)
                end_time = time.time()
                epoch_time = end_time - start_time
                epoch_times.append(epoch_time)
                history_acc.append(hist.history['accuracy'][0])
                history_val_acc.append(hist.history['val_accuracy'][0])
                history_loss.append(hist.history['loss'][0])
                history_val_loss.append(hist.history['val_loss'][0])
                tqdm.write(
                    f"Epoch {epoch+1}/{EPOCHS} - Time: {epoch_time:.2f}s - "
                    f"Train Acc: {history_acc[-1]:.4f} - Val Acc: {history_val_acc[-1]:.4f} - "
                    f"Train Loss: {history_loss[-1]:.4f} - Val Loss: {history_val_loss[-1]:.4f}"
                )
                log.write(
                    f"Epoch {epoch+1}/{EPOCHS} - Time: {epoch_time:.2f}s - "
                    f"Train Acc: {history_acc[-1]:.4f} - Val Acc: {history_val_acc[-1]:.4f} - "
                    f"Train Loss: {history_loss[-1]:.4f} - Val Loss: {history_val_loss[-1]:.4f}\n"
                )
            total_time = sum(epoch_times)
            print(f"Total training time for {model_name}: {total_time:.2f}s")
            log.write(f"Total training time for {model_name}: {total_time:.2f}s\n")
            clf.save(os.path.join(CUSTOM_MODEL_DIR, f'{model_name}_final.h5'))
            y_pred = np.argmax(clf.predict(X_val_feat), axis=1)
            y_true = np.argmax(y_val_enc, axis=1)
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            TP = np.diag(cm)
            FP = cm.sum(axis=0) - TP
            FN = cm.sum(axis=1) - TP
            TN = cm.sum() - (FP + FN + TP)
            for idx, cls in enumerate(list(class_names)):
                cls_report = report[cls] if cls in report else {}
                results.append({
                    'Model': model_name,
                    'Class': cls,
                    'Recall': float(cls_report['recall']) if 'recall' in cls_report else 0.0,  # type: ignore
                    'F1-Score': float(cls_report['f1-score']) if 'f1-score' in cls_report else 0.0,  # type: ignore
                    'Support': int(cls_report['support']) if 'support' in cls_report else 0,  # type: ignore
                    'Precision': float(cls_report['precision']) if 'precision' in cls_report else 0.0,  # type: ignore
                    'TP': int(TP[idx]),
                    'TN': int(TN[idx]),
                    'FP': int(FP[idx]),
                    'FN': int(FN[idx]),
                    'Total Images': int(len(X_val))
                })
            results.append({
                'Model': model_name,
                'Class': 'Overall',
                'Recall': recall_score(y_true, y_pred, average='macro'),
                'F1-Score': f1_score(y_true, y_pred, average='macro'),
                'Support': len(y_true),
                'Precision': precision_score(y_true, y_pred, average='macro'),
                'TP': TP.sum(),
                'TN': TN.sum(),
                'FP': FP.sum(),
                'FN': FN.sum(),
                'Total Images': len(X_val),
                'Accuracy': accuracy_score(y_true, y_pred)
            })
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names.tolist(), yticklabels=class_names.tolist())
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, f'confusion_matrix_{model_name}.png'))
            plt.close()
            plt.figure()
            plt.plot(range(1, EPOCHS+1), history_acc, label='Train Acc')
            plt.plot(range(1, EPOCHS+1), history_val_acc, label='Val Acc')
            plt.title(f'Training/Validation Accuracy - {model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, f'training_validation_accuracy_{model_name}.png'))
            plt.close()
            np.save(os.path.join(RESULT_DIR, f'epoch_times_{model_name}.npy'), np.array(epoch_times))
    df = pd.DataFrame(results)
    df.to_excel(os.path.join(RESULT_DIR, 'classification_results.xlsx'), index=False)
    print(tabulate(df.values.tolist(), headers=df.columns.tolist(), tablefmt='psql'))
    
if __name__ == '__main__':
    main()