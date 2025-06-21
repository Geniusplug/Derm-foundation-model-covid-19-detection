import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


BATCH_SIZE = 64
EPOCHS = 10  
IMG_SIZE = (224, 224)
SEED = 42
DATASET_DIR = 'Dataset'
RESULT_DIR = 'Result'
MODEL_DIR = 'CustomClassifier'
MODELS = [
    'GoogleDermModel_final.h5',
    'GoogleDermModel_finetuned.h5',
    'VGG19_final.h5',
    'VGG19_finetuned.h5',
    'Resnet50_final.h5',
    'Resnet50_finetuned.h5',
]

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

def preprocess_images(filepaths):
    images = []
    for path in filepaths:
        img = load_img(path, target_size=IMG_SIZE, color_mode='rgb')
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

def get_feature_extractor_for_model(model_file):
    if model_file.startswith('VGG19'):
        base = VGG19(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        preprocess = vgg_preprocess
    elif model_file.startswith('Resnet50'):
        base = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        preprocess = resnet_preprocess
    elif model_file.startswith('GoogleDermModel'):
        base = VGG19(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        preprocess = vgg_preprocess
    else:
        raise ValueError('Unknown model file')
    feature_extractor = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))
    return feature_extractor, preprocess

def is_finetuned_model(model_file):
    return 'finetuned' in model_file

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    cv_dir = os.path.join(os.getcwd(), 'Cross-validation folds')
    os.makedirs(cv_dir, exist_ok=True)
    X, y = load_dataset(DATASET_DIR)
    class_names = np.unique(y)
    y_encoded = np.array([np.where(class_names == label)[0][0] for label in y])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_model_results = []
    all_folds_df = []
    all_eval_df = []
    for model_file in MODELS:
        print(f'\n--- Cross-validation for {model_file} ---')
        feature_extractor, preprocess = get_feature_extractor_for_model(model_file)
        fold_results = []
        accs, precs, recs, f1s = [], [], [], []
        tps, tns, fps, fns = [], [], [], []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
            X_val, y_val = X[val_idx], y_encoded[val_idx]
            X_val_img = preprocess_images(X_val)
            if is_finetuned_model(model_file):
                X_val_input = preprocess(X_val_img)
            else:
                X_val_img = preprocess(X_val_img)
                X_val_input = feature_extractor.predict(X_val_img, batch_size=BATCH_SIZE, verbose=0)
            y_val_enc = to_categorical(y_val, num_classes=len(class_names))
            model = load_model(os.path.join(MODEL_DIR, model_file))
            y_pred = np.argmax(model.predict(X_val_input, batch_size=BATCH_SIZE, verbose=0), axis=1)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average=None, zero_division=0)
            rec = recall_score(y_val, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
            cm = confusion_matrix(y_val, y_pred)
            if not isinstance(prec, (np.ndarray, list)):
                prec = np.array([prec] * len(class_names))
            if not isinstance(rec, (np.ndarray, list)):
                rec = np.array([rec] * len(class_names))
            if not isinstance(f1, (np.ndarray, list)):
                f1 = np.array([f1] * len(class_names))
            for idx, cls in enumerate(class_names):
                if cm.shape == (2,2):
                    tp = cm[idx, idx]
                    fp = cm[:, idx].sum() - tp
                    fn = cm[idx, :].sum() - tp
                    tn = cm.sum() - (tp + fp + fn)
                else:
                    tp = tn = fp = fn = 0
                fold_results.append({
                    'Model': model_file,
                    'Class': cls,
                    'Accuracy': acc,
                    'Precision': float(prec[idx]),
                    'Recall': float(rec[idx]),
                    'F1-Score': float(f1[idx]),
                    'TP': tp,
                    'TN': tn,
                    'FP': fp,
                    'FN': fn,
                    'Total Images': len(y_val)
                })
        df = pd.DataFrame(fold_results)
        df.loc['mean'] = df.mean(numeric_only=True)
        df.loc['std'] = df.std(numeric_only=True)
        df.to_excel(os.path.join(cv_dir, f'cv_results_{model_file.replace(".h5","")}.xlsx'))
        print(tabulate(df.values.tolist(), headers=df.columns.tolist(), tablefmt='psql'))
        all_folds_df.append((df, model_file))
        accs = df[df['Class'] == class_names[0]]['Accuracy'].values[:-2]  # exclude mean/std rows
        precs = df[df['Class'] == class_names[0]]['Precision'].values[:-2]
        recs = df[df['Class'] == class_names[0]]['Recall'].values[:-2]
        f1s = df[df['Class'] == class_names[0]]['F1-Score'].values[:-2]
        all_model_results.append({
            'Model': model_file,
            'Accuracy': np.mean(accs),
            'Precision': np.mean(precs),
            'Recall': np.mean(recs),
            'F1-Score': np.mean(f1s)
        })
       
    with pd.ExcelWriter(os.path.join(cv_dir, 'cross_validation_all_models.xlsx')) as writer:
        for df, model_file in all_folds_df:
            sheet_name = model_file.replace('.h5', '')[:31]  
            df.to_excel(writer, sheet_name=sheet_name)
    for model_file in MODELS:
        per_model_path = os.path.join(cv_dir, f'cv_results_{model_file.replace(".h5","")}.xlsx')
        if os.path.exists(per_model_path):
            os.remove(per_model_path)
    for model_file in MODELS:
        feature_extractor, preprocess = get_feature_extractor_for_model(model_file)
        X_img = preprocess_images(X)
        if is_finetuned_model(model_file):
            X_input = preprocess(X_img)
        else:
            X_img = preprocess(X_img)
            X_input = feature_extractor.predict(X_img, batch_size=BATCH_SIZE, verbose=0)
        evaluate_on_full_set(model_file, class_names, X_input, y_encoded, lambda x: x, MODEL_DIR, BATCH_SIZE, cv_dir)

  
    model_names = [result['Model'].replace('.h5','') for result in all_model_results]
    accuracies = [result['Accuracy'] for result in all_model_results]
    plt.figure(figsize=(10,6))
    plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.ylabel('Mean Accuracy (5-fold CV)')
    plt.title('Model Comparison (Cross-Validation)')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, 'cv_model_comparison.png'))
    plt.close()

    from sklearn.preprocessing import label_binarize
    n_classes = len(class_names)
    plt.figure(figsize=(8, 6))
    for result, model_file in zip(all_model_results, MODELS):
        feature_extractor, preprocess = get_feature_extractor_for_model(model_file)
        X_img = preprocess_images(X)
        if is_finetuned_model(model_file):
            X_input = preprocess(X_img)
            model = load_model(os.path.join(MODEL_DIR, model_file))
            y_score = model.predict(X_input, batch_size=BATCH_SIZE, verbose=0)
        else:
            X_img = preprocess(X_img)
            X_input = feature_extractor.predict(X_img, batch_size=BATCH_SIZE, verbose=0)
            model = load_model(os.path.join(MODEL_DIR, model_file))
            y_score = model.predict(X_input, batch_size=BATCH_SIZE, verbose=0)
        y_bin = label_binarize(y_encoded, classes=range(n_classes))
        for idx, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_file.replace(".h5","")} - {cls} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models and Classes')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, 'roc_all_models.png'))
    plt.close()

    plt.figure(figsize=(16, 6))
    plt.text(0.05, 0.7, '1. Image Dataset\n(Chest X-ray Images)', fontsize=14, bbox=dict(facecolor='lightblue', alpha=0.7))
    plt.arrow(0.18, 0.7, 0.08, 0, head_width=0.04, head_length=0.02, fc='k', ec='k')
    plt.text(0.28, 0.7, '2. Data Preprocessing\n- Resize\n- Normalize', fontsize=14, bbox=dict(facecolor='lavender', alpha=0.7))
    plt.arrow(0.41, 0.7, 0.08, 0, head_width=0.04, head_length=0.02, fc='k', ec='k')
    plt.text(0.51, 0.7, '3. Feature Extraction\n(GoogleDermModel/VGG19/ResNet50)', fontsize=14, bbox=dict(facecolor='lightgreen', alpha=0.7))
    plt.arrow(0.64, 0.7, 0.08, 0, head_width=0.04, head_length=0.02, fc='k', ec='k')
    plt.text(0.74, 0.7, '4. Custom Classifier or Fine-tuned Model', fontsize=14, bbox=dict(facecolor='orange', alpha=0.7))
    plt.arrow(0.87, 0.7, 0.06, 0, head_width=0.04, head_length=0.02, fc='k', ec='k')
    plt.text(0.95, 0.7, '5. Cross-Validation & Evaluation\n(Accuracy, ROC, Confusion, Excel, Plots)', fontsize=14, bbox=dict(facecolor='pink', alpha=0.7))
    plt.axis('off')
    plt.title('Unified Workflow: Cross-Validation & Evaluation for All Models')
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, 'workflow_all_models.png'))
    plt.close()
def evaluate_on_full_set(model_file, class_names, X_feat, y_encoded, preprocess_images, MODEL_DIR, BATCH_SIZE, cv_dir):
    print(f'\n--- Evaluation on Full Dataset for {model_file} ---')
    model = load_model(os.path.join(MODEL_DIR, model_file))
    y_pred = np.argmax(model.predict(X_feat, batch_size=BATCH_SIZE, verbose=0), axis=1)
    acc = accuracy_score(y_encoded, y_pred)
    prec = precision_score(y_encoded, y_pred, average=None, zero_division=0)
    rec = recall_score(y_encoded, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_encoded, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_encoded, y_pred)
    eval_results = []
    if not isinstance(prec, (np.ndarray, list)):
        prec = np.array([prec] * len(class_names))
    if not isinstance(rec, (np.ndarray, list)):
        rec = np.array([rec] * len(class_names))
    if not isinstance(f1, (np.ndarray, list)):
        f1 = np.array([f1] * len(class_names))
    for idx, cls in enumerate(class_names):
        if cm.shape == (2,2):
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
        else:
            tp = tn = fp = fn = 0
        eval_results.append({
            'Model': model_file,
            'Class': cls,
            'Accuracy': acc,
            'Precision': float(prec[idx]),
            'Recall': float(rec[idx]),
            'F1-Score': float(f1[idx]),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Total Images': len(y_encoded)
        })
    df_eval = pd.DataFrame(eval_results)
    df_eval.to_excel(os.path.join(cv_dir, f'eval_fullset_{model_file.replace(".h5","")}.xlsx'))
    print(tabulate(df_eval.values.tolist(), headers=df_eval.columns.tolist(), tablefmt='psql'))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_file} (Full Set)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f'confusion_matrix_fullset_{model_file.replace(".h5","")}.png'))
    plt.close()
    for idx, cls in enumerate(class_names):
        y_true_bin = (y_encoded == idx).astype(int)
        if is_finetuned_model(model_file):
            y_score = model.predict(X_feat, batch_size=BATCH_SIZE, verbose=0)
        else:
            y_score = model.predict(X_feat, batch_size=BATCH_SIZE, verbose=0)
        if y_score.ndim == 1 or y_score.shape[1] == 1:
            y_score_cls = y_score.ravel()
        else:
            y_score_cls = y_score[:, idx]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score_cls)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_file} - {cls} (Full Set)')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(cv_dir, f'roc_{model_file.replace(".h5","")}_{cls}_fullset.png'))
        plt.close()




if __name__ == '__main__':
    main()
