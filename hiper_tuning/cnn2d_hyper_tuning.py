#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
# Camadas específicas para CNN 2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam # Usando legacy Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory # Para carregar imagens
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import keras_tuner as kt # Importar Keras Tuner
import os
import time

# --- Configurações Básicas ---
plt.style.use('fivethirtyeight')
sns.set_theme(style="whitegrid")

# --- Parâmetros para Carregamento e Modelo ---
IMAGE_DIR = 'data/VALE3.SA/imagens/'
TRAIN_DIR = os.path.join(IMAGE_DIR, 'treino')
TEST_DIR = os.path.join(IMAGE_DIR, 'teste')
# ATUALIZADO: Usar a resolução real das imagens
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 333
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
BATCH_SIZE = 32
COLOR_MODE = 'grayscale' # Mantendo grayscale, assumindo que são tons de cinza
CHANNELS = 1 if COLOR_MODE == 'grayscale' else 3
SEED = 42

# --- Configuração do Dispositivo ---
print("Configurando dispositivo TensorFlow...")
try:
    # ... (código de configuração do dispositivo mantido igual) ...
    if tf.config.list_physical_devices('GPU'):
        print("GPU disponível. Usando TensorFlow com GPU.")
    elif hasattr(tf.config, 'list_physical_devices') and tf.config.list_physical_devices('MPS'):
         print("Apple MPS disponível. Usando TensorFlow com MPS.")
    else:
        print("Nenhuma GPU ou MPS detectada. Usando TensorFlow em CPU.")
except Exception as e:
    print(f"Erro ao configurar dispositivos: {e}. Usando configuração padrão.")


# --- 1. Carregamento e Pré-processamento dos Dados (Imagens) ---
print(f"\nCarregando imagens de treino de: {TRAIN_DIR}")
try:
    # Carrega datasets usando image_dataset_from_directory
    train_dataset = image_dataset_from_directory(
        TRAIN_DIR, labels='inferred', label_mode='binary',
        image_size=IMAGE_SIZE, interpolation='nearest', batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE, shuffle=True, seed=SEED
    )
    print(f"Carregando imagens de teste de: {TEST_DIR}")
    test_dataset = image_dataset_from_directory(
        TEST_DIR, labels='inferred', label_mode='binary',
        image_size=IMAGE_SIZE, interpolation='nearest', batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE, shuffle=False, seed=SEED # Não embaralhar teste
    )

    class_names = train_dataset.class_names
    print(f"Classes encontradas: {class_names}")

    # Normalização e Otimização dos Datasets
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    print("Datasets pré-processados e otimizados.")

except FileNotFoundError as e: exit(f"Erro Crítico: Diretório não encontrado: {e}")
except Exception as e: exit(f"Erro Crítico ao carregar/processar imagens: {e}")


# --- 2. Definição da Função para Construir o Modelo CNN 2D (Keras Tuner) ---
def build_cnn2d_model(hp):
    """
    Constrói o modelo CNN 2D com hiperparâmetros definidos pelo Keras Tuner.
    """
    model = Sequential(name="CNN_2D_Tuned")
    model.add(keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))) # Entrada explícita

    # --- Hiperparâmetros a Otimizar ---
    # Número de blocos convolucionais
    hp_num_conv_blocks = hp.Int('num_conv_blocks', min_value=1, max_value=3, step=1)
    # Filtros (começando com um valor e talvez aumentando)
    hp_filters_start = hp.Choice('filters_start', values=[32, 64])
    # Kernel size
    hp_kernel_size = hp.Choice('kernel_size', values=[3, 5])
    # Usar Batch Normalization?
    hp_use_batchnorm = hp.Boolean('use_batchnorm', default=False)
    # Dropout rate
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    # Unidades na camada densa final (antes da saída)
    hp_dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
    # Taxa de aprendizado
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # 0.0001 foi bom no teste manual

    # --- Construindo a Arquitetura Dinamicamente ---
    filters = hp_filters_start
    for i in range(hp_num_conv_blocks):
        block_num = i + 1
        model.add(Conv2D(filters=filters,
                         kernel_size=(hp_kernel_size, hp_kernel_size), # Kernel quadrado
                         activation='relu',
                         padding='same',
                         name=f'conv{block_num}'))
        # Opcional: adicionar segunda conv por bloco
        # model.add(Conv2D(filters=filters, kernel_size=(hp_kernel_size, hp_kernel_size), activation='relu', padding='same', name=f'conv{block_num}b'))
        model.add(MaxPooling2D(pool_size=(2, 2), name=f'pool{block_num}'))
        if hp_use_batchnorm:
            model.add(BatchNormalization(name=f'bn{block_num}'))
        # Aumentar filtros para o próximo bloco (opcional)
        filters *= 2

    # Achatar a saída
    model.add(Flatten(name='flatten'))

    # Camada Densa Intermediária
    model.add(Dense(units=hp_dense_units, activation='relu', name='dense_intermediate'))

    # Dropout
    model.add(Dropout(hp_dropout_rate, name='dropout_final'))

    # Camada de Saída
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Compilação (usando LegacyAdam)
    model.compile(
        optimizer=LegacyAdam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# --- 3. Configuração e Execução do Keras Tuner ---
print("\nConfigurando o Keras Tuner (RandomSearch) para CNN 2D...")

tuner_cnn2d = kt.RandomSearch(
    hypermodel=build_cnn2d_model,
    objective=kt.Objective('val_auc', direction='max'),
    max_trials=20,  # Número de tentativas (ajuste conforme necessário)
    executions_per_trial=1,
    overwrite=True,
    directory='keras_tuner_cnn2d_dir', # Diretório específico
    project_name='cnn2d_stock_tuning'
)

tuner_cnn2d.search_space_summary()

stop_early_cnn2d = EarlyStopping(monitor='val_auc', patience=5, mode='max') # Early stopping para a busca

print("\nIniciando a busca de hiperparâmetros para CNN 2D...")
start_time_cnn2d = time.time()
tuner_cnn2d.search(train_ds, # Usa o dataset diretamente
                   epochs=30, # Épocas por trial
                   validation_data=test_ds,
                   callbacks=[stop_early_cnn2d],
                   # class_weight não é passado diretamente para fit com datasets
                   verbose=1
                   )
end_time_cnn2d = time.time()
print(f"\nBusca CNN 2D concluída em {(end_time_cnn2d - start_time_cnn2d)/60:.2f} minutos.")

# --- 4. Obter e Avaliar o Melhor Modelo CNN 2D ---
print("\nObtendo os melhores hiperparâmetros e o melhor modelo CNN 2D...")

best_hps_cnn2d = tuner_cnn2d.get_best_hyperparameters(num_trials=1)[0]

print("\nMelhores Hiperparâmetros CNN 2D Encontrados:")
for hp_name, hp_value in best_hps_cnn2d.values.items():
    print(f"- {hp_name}: {hp_value}")

# Obter o melhor modelo treinado pelo tuner
best_model_cnn2d = tuner_cnn2d.get_best_models(num_models=1)[0]
best_model_cnn2d.summary()

# Avaliar o melhor modelo CNN 2D
print("\nAvaliando o MELHOR modelo CNN 2D no conjunto de teste...")
results_cnn2d = best_model_cnn2d.evaluate(test_ds, verbose=0, batch_size=BATCH_SIZE)
loss_cnn2d = results_cnn2d[0]; accuracy_cnn2d = results_cnn2d[1]; auc_metric_cnn2d = results_cnn2d[2]

print(f"\nResultados da Avaliação (Melhor CNN 2D):")
print(f"Perda (Loss): {loss_cnn2d:.4f}")
print(f"Acurácia: {accuracy_cnn2d:.4f} ({accuracy_cnn2d*100:.2f}%)")
print(f"AUC (da evaluate): {auc_metric_cnn2d:.4f}")

# Gerar previsões com o melhor modelo CNN 2D
print("\nGerando previsões com o MELHOR modelo CNN 2D...")
y_pred_proba_list = []
y_true_list = []
for images, labels in test_ds: # Iterar sobre o dataset de teste
    preds = best_model_cnn2d.predict(images, verbose=0)
    y_pred_proba_list.extend(preds.flatten())
    y_true_list.extend(labels.numpy().flatten())

y_pred_proba_cnn2d = np.array(y_pred_proba_list)
y_true_cnn2d = np.array(y_true_list).astype(int) # Garantir que true labels são inteiros

threshold = 0.5
y_pred_cnn2d = (y_pred_proba_cnn2d >= threshold).astype(int)
print(f"Usando limiar padrão: {threshold:.2f} para classificação")

# Métricas de avaliação CNN 2D
cm_cnn2d = confusion_matrix(y_true_cnn2d, y_pred_cnn2d)
print("\nMatriz de Confusão (Melhor CNN 2D):"); print(cm_cnn2d)
report_cnn2d = classification_report(y_true_cnn2d, y_pred_cnn2d, target_names=class_names, zero_division=0)
print("\nRelatório de Classificação (Melhor CNN 2D):"); print(report_cnn2d)
try:
    roc_auc_score_cnn2d = roc_auc_score(y_true_cnn2d, y_pred_proba_cnn2d)
    print(f"AUC Score (calculado): {roc_auc_score_cnn2d:.4f}")
except ValueError as e: print(f"Não foi possível calcular AUC score: {e}")
except Exception as e: print(f"Erro ao calcular AUC Score: {e}")

# --- 5. Visualização (para o Melhor Modelo CNN 2D) ---
# (Gráficos de ROC e Distribuição de Probabilidades)
print("\nGerando gráficos para o MELHOR modelo CNN 2D...")
# Plotar curva ROC
if len(np.unique(y_true_cnn2d)) > 1:
    plt.figure(figsize=(8, 6))
    try:
        fpr_cnn2d, tpr_cnn2d, _ = roc_curve(y_true_cnn2d, y_pred_proba_cnn2d)
        roc_auc_cnn2d = auc(fpr_cnn2d, tpr_cnn2d)
        plt.plot(fpr_cnn2d, tpr_cnn2d, color='darkred', lw=2, label=f'Curva ROC (AUC = {roc_auc_cnn2d:.2f})') # Cor diferente
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos'); plt.ylabel('Taxa de Verdadeiros Positivos'); plt.title('Curva ROC (Melhor CNN 2D)')
        plt.legend(loc="lower right"); plt.grid(True); plt.savefig('tuned_cnn2d_roc_curve.png'); plt.show()
    except Exception as e: print(f"Erro ao gerar curva ROC CNN 2D: {e}")
else: print("Curva ROC não gerada.")

# Plotar distribuição das probabilidades
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba_cnn2d, bins=50, alpha=0.7, label='Probabilidades Previstas (CNN 2D)', color='indigo') # Cor diferente
plt.axvline(x=threshold, color='r', linestyle='--', label=f'Limiar Padrão: {threshold:.2f}')
plt.title('Distribuição das Probabilidades Previstas (Melhor CNN 2D)'); plt.xlabel('Probabilidade'); plt.ylabel('Frequência')
plt.legend(); plt.savefig('tuned_cnn2d_prediction_distribution.png'); plt.show()

# --- 6. Salvar o Melhor Modelo CNN 2D ---
print("\nSalvando o MELHOR modelo CNN 2D treinado...")
try:
    os.makedirs('models', exist_ok=True)
    best_model_cnn2d.save('models/tuned_cnn2d_model.keras')
    print("Melhor modelo CNN 2D salvo em 'models/tuned_cnn2d_model.keras'")
    with open('models/best_cnn2d_hyperparameters.txt', 'w') as f: # Salvar HPs da CNN 2D
        for hp_name, hp_value in best_hps_cnn2d.values.items():
            f.write(f"{hp_name}: {hp_value}\n")
    print("Melhores hiperparâmetros CNN 2D salvos em 'models/best_cnn2d_hyperparameters.txt'")
except Exception as e:
    print(f"Erro ao salvar o melhor modelo CNN 2D: {e}")

print("\nScript de Tuning CNN 2D finalizado.")
