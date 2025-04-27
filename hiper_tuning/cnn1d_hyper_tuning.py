#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
# Importa camadas CNN 1D
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam # Usar Adam padrão para tuning
# from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam # Ou legacy se preferir
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.utils import class_weight
import keras_tuner as kt # Importar Keras Tuner
import os
import time # Para medir o tempo

# --- Configurações e Setup Inicial ---
plt.style.use('fivethirtyeight')
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)

# Configuração do dispositivo
try:
    if tf.config.list_physical_devices('GPU'):
        print("GPU disponível. Usando TensorFlow com GPU.")
    elif hasattr(tf.config, 'list_physical_devices') and tf.config.list_physical_devices('MPS'):
         print("Apple MPS disponível. Usando TensorFlow com MPS.")
    else:
        print("Usando TensorFlow em CPU com otimizações.")
except Exception as e:
    print(f"Erro ao configurar dispositivos: {e}")
    print("Usando configuração padrão.")

# --- 1. Carregamento dos Dados (Idêntico ao script anterior) ---
data_dir = 'data/VALE3.SA/'
train_path = os.path.join(data_dir, 'treino.csv')
test_path = os.path.join(data_dir, 'teste.csv')
try:
    df_train_raw = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    df_test_raw = pd.read_csv(test_path, index_col='Date', parse_dates=True)
    print("Dados carregados com sucesso!")
    if 'Unnamed: 0' in df_train_raw.columns: df_train_raw = df_train_raw.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in df_test_raw.columns: df_test_raw = df_test_raw.drop(columns=['Unnamed: 0'])
except FileNotFoundError as e: exit(f"Erro ao carregar os arquivos: {e}")
except Exception as e: exit(f"Ocorreu um erro inesperado ao carregar os dados: {e}")

# --- 2. Engenharia de Features e Definição do Alvo (Idêntico ao script anterior) ---
def calculate_features_and_target_adapted(df):
    # ... (Código da função idêntico ao anterior) ...
    df_copy = df.copy()
    price_col = 'Close'
    if price_col not in df_copy.columns: raise ValueError(f"Coluna '{price_col}' não encontrada.")
    if 'Label' not in df_copy.columns: raise ValueError("Coluna 'Label' (alvo) não encontrada.")
    df_copy['daily_return'] = df_copy[price_col].pct_change() * 100
    for window in [5, 10, 20, 50]: df_copy[f'sma_{window}'] = df_copy[price_col].rolling(window=window).mean()
    df_copy['ema_12'] = df_copy[price_col].ewm(span=12, adjust=False).mean()
    df_copy['ema_26'] = df_copy[price_col].ewm(span=26, adjust=False).mean()
    df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
    df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
    delta = df_copy[price_col].diff(); gain = delta.where(delta > 0, 0).fillna(0); loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean(); avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
    rs = avg_gain / avg_loss; df_copy['rsi'] = 100 - (100 / (1 + rs)); df_copy['rsi'] = df_copy['rsi'].fillna(50)
    for window in [5, 10, 20]: df_copy[f'roc_{window}'] = df_copy[price_col].pct_change(periods=window) * 100
    past_cols = [col for col in df_copy.columns if col.startswith('Past_') and col.endswith('_Close')]
    for col in past_cols:
        try: df_copy[f'return_{col}'] = (df_copy[price_col] / df_copy[col] - 1) * 100; df_copy[f'return_{col}'] = df_copy[f'return_{col}'].replace([np.inf, -np.inf], np.nan)
        except Exception as e: pass
    if df_copy['Label'].isin([1, -1]).all(): df_copy['signal'] = df_copy['Label'].map({1: 1, -1: 0})
    elif df_copy['Label'].isin([1, 0]).all(): df_copy['signal'] = df_copy['Label']
    elif df_copy['Label'].isin(['buy', 'sell']).all(): df_copy['signal'] = df_copy['Label'].map({'buy': 1, 'sell': 0})
    else: raise ValueError(f"Valores inesperados na coluna 'Label': {df_copy['Label'].unique()}.")
    if 'Label' in df_copy.columns: df_copy = df_copy.drop(columns=['Label'])
    return df_copy

print("\nProcessando dados...")
df_train_processed = calculate_features_and_target_adapted(df_train_raw)
df_test_processed = calculate_features_and_target_adapted(df_test_raw)

# --- 3. Tratamento de NaNs (Idêntico ao script anterior) ---
print("\nTratando NaNs...")
return_past_cols = [col for col in df_train_processed.columns if col.startswith('return_Past_')]
df_train_processed[return_past_cols] = df_train_processed[return_past_cols].fillna(0)
df_test_processed[return_past_cols] = df_test_processed[return_past_cols].fillna(0)
df_train_processed.dropna(inplace=True); df_test_processed.dropna(inplace=True)
print(f"Shape treino final: {df_train_processed.shape}, Shape teste final: {df_test_processed.shape}")
if df_train_processed.empty or df_test_processed.empty: exit("Erro: DataFrames vazios.")
if 'signal' in df_train_processed.columns:
    df_train_processed['signal'] = df_train_processed['signal'].astype(int)
    df_test_processed['signal'] = df_test_processed['signal'].astype(int)
else: exit("Erro: Coluna 'signal' não encontrada.")

# --- 4. Seleção de Features (Idêntico ao script anterior) ---
feature_cols = [
    'daily_return', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'macd', 'macd_signal', 'macd_hist', 'rsi', 'roc_5', 'roc_10', 'roc_20',
    'return_Past_1_Days_Close', 'return_Past_3_Days_Close', 'return_Past_5_Days_Close',
    'return_Past_10_Days_Close', 'return_Past_15_Days_Close', 'Close'
]
target_col = 'signal'
feature_cols = sorted(list(set([col for col in feature_cols if col in df_train_processed.columns and col in df_test_processed.columns])))
print(f"\nFeatures selecionadas ({len(feature_cols)}): {feature_cols}")
missing_train = [col for col in feature_cols if col not in df_train_processed.columns]; missing_test = [col for col in feature_cols if col not in df_test_processed.columns]
if missing_train: exit(f"ERRO: Features faltando no treino: {missing_train}")
if missing_test: exit(f"ERRO: Features faltando no teste: {missing_test}")
X_train = df_train_processed[feature_cols]; y_train = df_train_processed[target_col]
X_test = df_test_processed[feature_cols]; y_test = df_test_processed[target_col]

# --- 5. Escalonamento (Idêntico ao script anterior) ---
print("\nEscalonando dados...")
scaler = MinMaxScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

# --- 6. Criação de Sequências (Idêntico ao script anterior) ---
def create_sequences(X, y, n_steps):
    # ... (Código da função idêntico ao anterior) ...
    if len(X) < n_steps: return np.array([]), np.array([]), np.array([])
    Xs, ys, y_indices = [], [], []
    for i in range(len(X) - n_steps + 1):
        end_ix = i + n_steps; Xs.append(X[i:end_ix]); ys.append(y.iloc[end_ix - 1]); y_indices.append(y.index[end_ix - 1])
    if not Xs: return np.array([]), np.array([]), np.array([])
    return np.array(Xs), np.array(ys), np.array(y_indices)

n_steps = 15
print(f"\nCriando sequências com n_steps = {n_steps}...")
X_train_seq, y_train_seq, _ = create_sequences(X_train_scaled, y_train, n_steps)
X_test_seq, y_test_seq, y_test_indices = create_sequences(X_test_scaled, y_test, n_steps)
if X_train_seq.size == 0 or X_test_seq.size == 0: exit("Erro: Não foi possível criar as sequências.")
print(f"X_train_seq shape: {X_train_seq.shape}, X_test_seq shape: {X_test_seq.shape}")

# --- 7. Balanceamento de Classes (Idêntico ao script anterior) ---
print("\nCalculando pesos das classes...")
if len(y_train_seq) > 0 and len(np.unique(y_train_seq)) > 1:
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Pesos das classes: {class_weight_dict}")
else:
    print("Aviso: Não foi possível calcular pesos."); class_weight_dict = None

# --- 8. Definição da Função para Construir o Modelo CNN 1D (Keras Tuner) ---
def build_cnn_model(hp):
    """
    Constrói o modelo CNN 1D com hiperparâmetros definidos pelo Keras Tuner.
    Args:
        hp: Objeto HyperParameters do Keras Tuner.
    Returns:
        Modelo Keras compilado.
    """
    n_features = X_train_seq.shape[2]
    model = Sequential(name="CNN_1D_Tuned")

    # --- Definindo Hiperparâmetros para Otimizar ---
    # 1. Número de filtros na primeira camada Conv1D
    hp_filters_1 = hp.Int('filters_1', min_value=32, max_value=128, step=32) # Ex: 32, 64, 96, 128

    # 2. Tamanho do Kernel
    hp_kernel_size = hp.Choice('kernel_size', values=[3, 5]) # Ex: 3 ou 5

    # 3. (Opcional) Adicionar segunda camada Conv/Pool?
    hp_add_conv2 = hp.Boolean('add_conv2', default=False)
    hp_filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32, parent_name='add_conv2', parent_values=True)

    # 4. Tipo de Pooling após a última camada convolucional
    hp_pooling_type = hp.Choice('pooling_type', values=['max', 'global_max']) # MaxPooling1D ou GlobalMaxPooling1D

    # 5. Taxa de Dropout após Pooling/Flatten
    hp_dropout_cnn = hp.Float('dropout_cnn', min_value=0.1, max_value=0.5, step=0.1)

    # 6. (Opcional) Unidades na camada Dense intermediária
    hp_add_dense = hp.Boolean('add_dense', default=False)
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32, parent_name='add_dense', parent_values=True)


    # 7. Taxa de Aprendizado (Learning Rate)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # --- Construindo a Arquitetura ---
    model.add(Conv1D(filters=hp_filters_1,
                     kernel_size=hp_kernel_size,
                     activation='relu',
                     padding='same', # 'same' pode ajudar a manter o tamanho da sequência
                     input_shape=(n_steps, n_features),
                     name='conv1d_layer_1'))

    if hp_add_conv2:
        model.add(Conv1D(filters=hp_filters_2,
                         kernel_size=hp_kernel_size, # Pode usar o mesmo ou outro hp
                         activation='relu',
                         padding='same',
                         name='conv1d_layer_2'))

    # Adiciona a camada de Pooling escolhida
    if hp_pooling_type == 'max':
        # MaxPooling reduz a dimensão, pode precisar ajustar pool_size
        model.add(MaxPooling1D(pool_size=2, name='maxpooling1d_layer'))
        model.add(Flatten(name='flatten_layer')) # Flatten é necessário após MaxPooling
    elif hp_pooling_type == 'global_max':
        model.add(GlobalMaxPooling1D(name='globalmaxpooling1d_layer'))
        # Flatten não é necessário após GlobalMaxPooling

    # Adiciona camada Dense intermediária opcional
    if hp_add_dense:
        model.add(Dense(units=hp_dense_units, activation='relu', name='dense_intermediate'))

    # Dropout
    model.add(Dropout(hp_dropout_cnn, name='dropout_cnn_tuned'))

    # Camada de Saída
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Compilação
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate), # Usar Adam padrão
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# --- 9. Configuração e Execução do Keras Tuner ---
print("\nConfigurando o Keras Tuner (RandomSearch) para CNN 1D...")

tuner_cnn = kt.RandomSearch(
    hypermodel=build_cnn_model, # Função que constrói o modelo CNN 1D
    objective=kt.Objective('val_auc', direction='max'),
    max_trials=15,  # Aumentar um pouco os trials para CNN?
    executions_per_trial=1,
    overwrite=True,
    directory='keras_tuner_cnn_dir', # Diretório diferente do LSTM
    project_name='cnn1d_stock_tuning'
)

tuner_cnn.search_space_summary()

stop_early_cnn = EarlyStopping(monitor='val_auc', patience=5, mode='max')

print("\nIniciando a busca de hiperparâmetros para CNN 1D...")
start_time_cnn = time.time()
tuner_cnn.search(X_train_seq, y_train_seq,
                 epochs=30, # Épocas para cada trial
                 validation_data=(X_test_seq, y_test_seq),
                 callbacks=[stop_early_cnn],
                 class_weight=class_weight_dict,
                 batch_size=64, # Manter fixo ou otimizar
                 verbose=1
                 )
end_time_cnn = time.time()
print(f"\nBusca CNN 1D concluída em {(end_time_cnn - start_time_cnn)/60:.2f} minutos.")

# --- 10. Obter e Avaliar o Melhor Modelo CNN 1D ---
print("\nObtendo os melhores hiperparâmetros e o melhor modelo CNN 1D...")

best_hps_cnn = tuner_cnn.get_best_hyperparameters(num_trials=1)[0]

print("\nMelhores Hiperparâmetros CNN 1D Encontrados:")
# Iterar para mostrar todos os HPs encontrados
for hp_name, hp_value in best_hps_cnn.values.items():
    print(f"- {hp_name}: {hp_value}")

# Construir/obter o melhor modelo
best_model_cnn = tuner_cnn.get_best_models(num_models=1)[0]
best_model_cnn.summary()

# Avaliar o melhor modelo CNN 1D
print("\nAvaliando o MELHOR modelo CNN 1D no conjunto de teste...")
results_cnn = best_model_cnn.evaluate(X_test_seq, y_test_seq, verbose=0, batch_size=128)
loss_cnn = results_cnn[0]; accuracy_cnn = results_cnn[1]; auc_metric_cnn = results_cnn[2]

print(f"\nResultados da Avaliação (Melhor CNN 1D):")
print(f"Perda (Loss): {loss_cnn:.4f}")
print(f"Acurácia: {accuracy_cnn:.4f} ({accuracy_cnn*100:.2f}%)")
print(f"AUC (da evaluate): {auc_metric_cnn:.4f}")

# Gerar previsões com o melhor modelo CNN 1D
print("\nGerando previsões com o MELHOR modelo CNN 1D...")
y_pred_proba_cnn = best_model_cnn.predict(X_test_seq, batch_size=128, verbose=0).flatten()
threshold = 0.5
y_pred_cnn = (y_pred_proba_cnn >= threshold).astype(int)
print(f"Usando limiar padrão: {threshold:.2f} para classificação")

# Criar DataFrame com resultados CNN 1D
results_df_cnn = pd.DataFrame({'Actual': y_test_seq, 'Predicted': y_pred_cnn, 'Probability': y_pred_proba_cnn}, index=y_test_indices)
plot_df_cnn = df_test_processed.loc[results_df_cnn.index].copy().join(results_df_cnn)

# Métricas de avaliação CNN 1D
cm_cnn = confusion_matrix(y_test_seq, y_pred_cnn)
print("\nMatriz de Confusão (Melhor CNN 1D):"); print(cm_cnn)
report_cnn = classification_report(y_test_seq, y_pred_cnn, target_names=['Venda (0)', 'Compra (1)'], zero_division=0)
print("\nRelatório de Classificação (Melhor CNN 1D):"); print(report_cnn)
try:
    roc_auc_score_cnn = roc_auc_score(y_test_seq, y_pred_proba_cnn)
    print(f"AUC Score (calculado): {roc_auc_score_cnn:.4f}")
except ValueError as e: print(f"Não foi possível calcular AUC score: {e}")
except Exception as e: print(f"Erro ao calcular AUC Score: {e}")

# --- 11. Visualização (para o Melhor Modelo CNN 1D) ---
print("\nGerando gráficos para o MELHOR modelo CNN 1D...")

# Plotar curva ROC
if len(np.unique(y_test_seq)) > 1:
    plt.figure(figsize=(8, 6))
    try:
        fpr_cnn, tpr_cnn, _ = roc_curve(y_test_seq, y_pred_proba_cnn)
        roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
        plt.plot(fpr_cnn, tpr_cnn, color='purple', lw=2, label=f'Curva ROC (AUC = {roc_auc_cnn:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos'); plt.ylabel('Taxa de Verdadeiros Positivos'); plt.title('Curva ROC (Melhor CNN 1D)')
        plt.legend(loc="lower right"); plt.grid(True); plt.savefig('tuned_cnn1d_roc_curve.png'); plt.show()
    except Exception as e: print(f"Erro ao gerar curva ROC CNN 1D: {e}")
else: print("Curva ROC não gerada.")

# Plotar distribuição das probabilidades
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba_cnn, bins=50, alpha=0.7, label='Probabilidades Previstas (CNN 1D)', color='green')
plt.axvline(x=threshold, color='r', linestyle='--', label=f'Limiar Padrão: {threshold:.2f}')
plt.title('Distribuição das Probabilidades Previstas (Melhor CNN 1D)'); plt.xlabel('Probabilidade'); plt.ylabel('Frequência')
plt.legend(); plt.savefig('tuned_cnn1d_prediction_distribution.png'); plt.show()

# Plotar Preço Real vs. Sinais Previstos
print("\nGerando gráfico de Preço vs. Sinais Previstos (Melhor CNN 1D)...")
plt.figure(figsize=(18, 8))
plt.plot(plot_df_cnn.index, plot_df_cnn['Close'], label='Preço Fechamento (Real)', color='skyblue', alpha=0.8, zorder=1)
buy_signals_cnn = plot_df_cnn[plot_df_cnn['Predicted'] == 1]; sell_signals_cnn = plot_df_cnn[plot_df_cnn['Predicted'] == 0]
plt.scatter(buy_signals_cnn.index, buy_signals_cnn['Close'], label='Compra Prevista (1)', marker='^', color='lime', s=100, alpha=0.9, zorder=2, edgecolor='black')
plt.scatter(sell_signals_cnn.index, sell_signals_cnn['Close'], label='Venda Prevista (0)', marker='v', color='red', s=100, alpha=0.9, zorder=2, edgecolor='black')
plt.title('Preço Real vs. Sinais Previstos (Melhor CNN 1D Pós-Tuning)'); plt.xlabel('Data'); plt.ylabel('Preço Fechamento')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate(); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig('tuned_cnn1d_price_vs_predictions.png'); plt.show()

# --- 12. Salvar o Melhor Modelo CNN 1D ---
print("\nSalvando o MELHOR modelo CNN 1D treinado...")
try:
    os.makedirs('models', exist_ok=True)
    best_model_cnn.save('models/tuned_cnn1d_model.keras')
    print("Melhor modelo CNN 1D salvo em 'models/tuned_cnn1d_model.keras'")
    with open('models/best_cnn1d_hyperparameters.txt', 'w') as f: # Salvar HPs da CNN
        for hp_name, hp_value in best_hps_cnn.values.items():
            f.write(f"{hp_name}: {hp_value}\n")
    print("Melhores hiperparâmetros CNN 1D salvos em 'models/best_cnn1d_hyperparameters.txt'")
except Exception as e:
    print(f"Erro ao salvar o melhor modelo CNN 1D: {e}")

print("\nScript de Tuning CNN 1D finalizado.")
