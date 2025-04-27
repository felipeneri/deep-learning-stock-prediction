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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam # Importar Adam explicitamente
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.utils import class_weight
import keras_tuner as kt # Importar Keras Tuner
import os
import time # Para medir o tempo

# --- Configurações e Setup Inicial (igual ao script anterior) ---
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

# --- 1. Carregamento dos Dados (igual ao script anterior) ---
data_dir = 'data/VALE3.SA/'
train_path = os.path.join(data_dir, 'treino.csv')
test_path = os.path.join(data_dir, 'teste.csv')

try:
    df_train_raw = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    df_test_raw = pd.read_csv(test_path, index_col='Date', parse_dates=True)
    print("Dados carregados com sucesso!")
    if 'Unnamed: 0' in df_train_raw.columns:
        df_train_raw = df_train_raw.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in df_test_raw.columns:
        df_test_raw = df_test_raw.drop(columns=['Unnamed: 0'])
except FileNotFoundError as e:
    print(f"Erro ao carregar os arquivos: {e}")
    exit(1)
except Exception as e:
    print(f"Ocorreu um erro inesperado ao carregar os dados: {e}")
    exit(1)

# --- 2. Engenharia de Features e Definição do Alvo (igual ao script anterior) ---
def calculate_features_and_target_adapted(df):
    df_copy = df.copy()
    price_col = 'Close'
    if price_col not in df_copy.columns: raise ValueError(f"Coluna '{price_col}' não encontrada.")
    if 'Label' not in df_copy.columns: raise ValueError("Coluna 'Label' (alvo) não encontrada.")

    # Cálculos de features (SMA, EMA, MACD, RSI, ROC, return_Past_*)
    # ... (código da função mantido igual ao script anterior) ...
    df_copy['daily_return'] = df_copy[price_col].pct_change() * 100
    for window in [5, 10, 20, 50]:
         df_copy[f'sma_{window}'] = df_copy[price_col].rolling(window=window).mean()
    df_copy['ema_12'] = df_copy[price_col].ewm(span=12, adjust=False).mean()
    df_copy['ema_26'] = df_copy[price_col].ewm(span=26, adjust=False).mean()
    df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
    df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
    delta = df_copy[price_col].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df_copy['rsi'] = 100 - (100 / (1 + rs))
    df_copy['rsi'] = df_copy['rsi'].fillna(50)
    for window in [5, 10, 20]:
        df_copy[f'roc_{window}'] = df_copy[price_col].pct_change(periods=window) * 100
    past_cols = [col for col in df_copy.columns if col.startswith('Past_') and col.endswith('_Close')]
    for col in past_cols:
        try:
            df_copy[f'return_{col}'] = (df_copy[price_col] / df_copy[col] - 1) * 100
            df_copy[f'return_{col}'] = df_copy[f'return_{col}'].replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            pass # Ignora erros silenciosamente aqui
    # Mapeamento do Label para signal
    if df_copy['Label'].isin([1, -1]).all():
        df_copy['signal'] = df_copy['Label'].map({1: 1, -1: 0})
    else:
        unique_labels = df_copy['Label'].unique()
        if df_copy['Label'].isin([1, 0]).all():
             df_copy['signal'] = df_copy['Label']
        elif df_copy['Label'].isin(['buy', 'sell']).all():
             df_copy['signal'] = df_copy['Label'].map({'buy': 1, 'sell': 0})
        else:
             raise ValueError(f"Valores inesperados na coluna 'Label': {unique_labels}.")
    if 'Label' in df_copy.columns:
        df_copy = df_copy.drop(columns=['Label'])
    return df_copy

print("\nProcessando dados...")
df_train_processed = calculate_features_and_target_adapted(df_train_raw)
df_test_processed = calculate_features_and_target_adapted(df_test_raw)

# --- 3. Tratamento de NaNs (igual ao script anterior) ---
print("\nTratando NaNs...")
return_past_cols = [col for col in df_train_processed.columns if col.startswith('return_Past_')]
df_train_processed[return_past_cols] = df_train_processed[return_past_cols].fillna(0)
df_test_processed[return_past_cols] = df_test_processed[return_past_cols].fillna(0)
df_train_processed.dropna(inplace=True)
df_test_processed.dropna(inplace=True)
print(f"Shape treino final: {df_train_processed.shape}")
print(f"Shape teste final: {df_test_processed.shape}")
if df_train_processed.empty or df_test_processed.empty: exit("Erro: DataFrames vazios.")
if 'signal' in df_train_processed.columns:
    df_train_processed['signal'] = df_train_processed['signal'].astype(int)
    df_test_processed['signal'] = df_test_processed['signal'].astype(int)
else: exit("Erro: Coluna 'signal' não encontrada.")

# --- 4. Seleção de Features (igual ao script anterior) ---
feature_cols = [
    'daily_return', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'macd', 'macd_signal', 'macd_hist', 'rsi', 'roc_5', 'roc_10', 'roc_20',
    'return_Past_1_Days_Close', 'return_Past_3_Days_Close', 'return_Past_5_Days_Close',
    'return_Past_10_Days_Close', 'return_Past_15_Days_Close', 'Close'
]
target_col = 'signal'
feature_cols = sorted(list(set([col for col in feature_cols if col in df_train_processed.columns and col in df_test_processed.columns])))
print(f"\nFeatures selecionadas ({len(feature_cols)}): {feature_cols}")
missing_train = [col for col in feature_cols if col not in df_train_processed.columns]
missing_test = [col for col in feature_cols if col not in df_test_processed.columns]
if missing_train: exit(f"ERRO: Features faltando no treino: {missing_train}")
if missing_test: exit(f"ERRO: Features faltando no teste: {missing_test}")

X_train = df_train_processed[feature_cols]
y_train = df_train_processed[target_col]
X_test = df_test_processed[feature_cols]
y_test = df_test_processed[target_col]

# --- 5. Escalonamento (igual ao script anterior) ---
print("\nEscalonando dados...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Criação de Sequências (igual ao script anterior) ---
def create_sequences(X, y, n_steps):
    if len(X) < n_steps: return np.array([]), np.array([]), np.array([])
    Xs, ys, y_indices = [], [], []
    for i in range(len(X) - n_steps + 1):
        end_ix = i + n_steps
        Xs.append(X[i:end_ix])
        ys.append(y.iloc[end_ix - 1])
        y_indices.append(y.index[end_ix - 1])
    if not Xs: return np.array([]), np.array([]), np.array([])
    return np.array(Xs), np.array(ys), np.array(y_indices)

n_steps = 15
print(f"\nCriando sequências com n_steps = {n_steps}...")
X_train_seq, y_train_seq, _ = create_sequences(X_train_scaled, y_train, n_steps)
X_test_seq, y_test_seq, y_test_indices = create_sequences(X_test_scaled, y_test, n_steps)
if X_train_seq.size == 0 or X_test_seq.size == 0: exit("Erro: Não foi possível criar as sequências.")
print(f"X_train_seq shape: {X_train_seq.shape}, X_test_seq shape: {X_test_seq.shape}")

# --- 7. Balanceamento de Classes (igual ao script anterior) ---
print("\nCalculando pesos das classes...")
if len(y_train_seq) > 0 and len(np.unique(y_train_seq)) > 1:
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Pesos das classes: {class_weight_dict}")
else:
    print("Aviso: Não foi possível calcular pesos.")
    class_weight_dict = None

# --- 8. Definição da Função para Construir o Modelo (Keras Tuner) ---
def build_model(hp):
    """
    Constrói o modelo LSTM com hiperparâmetros definidos pelo Keras Tuner.
    Args:
        hp: Objeto HyperParameters do Keras Tuner.
    Returns:
        Modelo Keras compilado.
    """
    n_features = X_train_seq.shape[2] # Pega o número de features dos dados globais
    # CORREÇÃO: Remover hp.get('trial_id') do nome
    model = Sequential(name="LSTM_Tuned") # Nome estático

    # --- Definindo Hiperparâmetros para Otimizar ---
    # 1. Número de unidades na camada LSTM
    hp_units = hp.Int('units', min_value=32, max_value=128, step=16) # Ex: 32, 48, 64, ..., 128

    # 2. Taxa de Dropout após LSTM
    hp_dropout_lstm = hp.Float('dropout_lstm', min_value=0.1, max_value=0.5, step=0.1) # Ex: 0.1, 0.2, ..., 0.5

    # 3. (Opcional) Adicionar uma segunda camada LSTM?
    # hp_add_lstm2 = hp.Boolean('add_lstm2', default=False)

    # 4. (Opcional) Adicionar camada Dense intermediária?
    # hp_add_dense = hp.Boolean('add_dense', default=False)
    # hp_dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16, parent_name='add_dense', parent_values=True)
    # hp_dropout_dense = hp.Float('dropout_dense', min_value=0.1, max_value=0.5, step=0.1, parent_name='add_dense', parent_values=True)

    # 5. Taxa de Aprendizado (Learning Rate) do Otimizador Adam
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # Ex: 0.01, 0.001, 0.0001

    # --- Construindo a Arquitetura com os Hiperparâmetros ---
    model.add(LSTM(units=hp_units,
                   input_shape=(n_steps, n_features),
                   recurrent_dropout=0.1, # Dropout recorrente pode ser fixo ou otimizado também
                   return_sequences=False, # Ajustar se adicionar mais camadas LSTM
                   name='lstm_layer_tuned'))
    model.add(Dropout(hp_dropout_lstm, name='dropout_lstm_tuned'))

    # Exemplo de como adicionar camadas opcionais (descomentar se quiser testar)
    # if hp_add_dense:
    #     model.add(Dense(units=hp_dense_units, activation='relu', name='dense_layer_tuned'))
    #     model.add(Dropout(hp_dropout_dense, name='dropout_dense_tuned'))

    # Camada de saída
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Compilação do Modelo
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate), # Usar a taxa de aprendizado otimizada
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] # Monitorar AUC
    )
    return model

# --- 9. Configuração e Execução do Keras Tuner ---
print("\nConfigurando o Keras Tuner (RandomSearch)...")

# Instanciar o Tuner
tuner = kt.RandomSearch(
    hypermodel=build_model, # Função que constrói o modelo
    objective=kt.Objective('val_auc', direction='max'), # Objetivo é maximizar AUC na validação
    max_trials=10,  # Número de combinações de hiperparâmetros a testar (ajuste conforme tempo/recursos)
    executions_per_trial=1, # Quantas vezes treinar cada combinação (1 é comum para começar)
    overwrite=True, # Sobrescrever resultados anteriores se o diretório existir
    directory='keras_tuner_dir', # Diretório para salvar os logs e checkpoints
    project_name='lstm_stock_tuning' # Nome do projeto
)

# Resumo do espaço de busca
tuner.search_space_summary()

# Callback EarlyStopping para a busca do tuner (evita que cada trial demore demais)
stop_early = EarlyStopping(monitor='val_auc', patience=5, mode='max') # Paciência menor durante a busca

# Executar a busca de hiperparâmetros
print("\nIniciando a busca de hiperparâmetros...")
start_time = time.time()
tuner.search(X_train_seq, y_train_seq,
             epochs=30, # Número menor de épocas para cada trial da busca (EarlyStopping vai parar antes se convergir)
             validation_data=(X_test_seq, y_test_seq),
             callbacks=[stop_early],
             class_weight=class_weight_dict,
             batch_size=64, # Pode otimizar o batch_size também, mas fixamos aqui
             verbose=1 # Mostrar progresso
             )
end_time = time.time()
print(f"\nBusca de hiperparâmetros concluída em {(end_time - start_time)/60:.2f} minutos.")

# --- 10. Obter e Avaliar o Melhor Modelo ---
print("\nObtendo os melhores hiperparâmetros e o melhor modelo...")

# Pegar os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Melhores Hiperparâmetros Encontrados:
- Unidades LSTM: {best_hps.get('units')}
- Dropout LSTM: {best_hps.get('dropout_lstm'):.2f}
- Taxa de Aprendizado: {best_hps.get('learning_rate')}
""")
# Imprimir outros HPs se foram adicionados à busca (ex: hp_add_dense, etc.)

# Construir o modelo final com os melhores hiperparâmetros
best_model = tuner.get_best_models(num_models=1)[0]
# Alternativamente, reconstruir e retreinar do zero com mais épocas:
# best_model = build_model(best_hps)
# print("\nRetreinando o melhor modelo com mais épocas...")
# history = best_model.fit(X_train_seq, y_train_seq, epochs=50, validation_data=(X_test_seq, y_test_seq), ...)

# Sumário do melhor modelo
best_model.summary()

# Avaliar o melhor modelo no conjunto de teste
print("\nAvaliando o MELHOR modelo no conjunto de teste...")
results = best_model.evaluate(X_test_seq, y_test_seq, verbose=0, batch_size=128) # Usar batch maior para avaliação
loss = results[0]
accuracy = results[1]
auc_metric = results[2]

print(f"\nResultados da Avaliação (Melhor Modelo):")
print(f"Perda (Loss): {loss:.4f}")
print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"AUC (da evaluate): {auc_metric:.4f}")

# Gerar previsões com o melhor modelo
print("\nGerando previsões com o MELHOR modelo...")
y_pred_proba = best_model.predict(X_test_seq, batch_size=128, verbose=0).flatten()

# Converter probabilidades em classes (0 ou 1) usando limiar 0.5
threshold = 0.5
y_pred = (y_pred_proba >= threshold).astype(int)
print(f"Usando limiar padrão: {threshold:.2f} para classificação")

# Criar DataFrame com resultados para facilitar análise e plotagem
results_df = pd.DataFrame({
    'Actual': y_test_seq,
    'Predicted': y_pred,
    'Probability': y_pred_proba
}, index=y_test_indices) # Usar os índices corretos

# Juntar com os dados originais de teste para pegar o preço 'Close'
plot_df = df_test_processed.loc[results_df.index].copy()
plot_df = plot_df.join(results_df)

# Matriz de confusão
cm = confusion_matrix(y_test_seq, y_pred)
print("\nMatriz de Confusão (Melhor Modelo):")
print(cm)

# Relatório de Classificação
print("\nRelatório de Classificação (Melhor Modelo):")
report = classification_report(y_test_seq, y_pred, target_names=['Venda (0)', 'Compra (1)'], zero_division=0)
print(report)

# AUC Score (calculado a partir das probabilidades)
try:
    roc_auc_score_val = roc_auc_score(y_test_seq, y_pred_proba)
    print(f"AUC Score (calculado): {roc_auc_score_val:.4f}")
except ValueError as e:
     print(f"Não foi possível calcular AUC score: {e}")
except Exception as e:
    print(f"Erro ao calcular AUC Score: {e}")

# --- 11. Visualização (para o Melhor Modelo) ---
print("\nGerando gráficos para o MELHOR modelo...")

# Plotar curva ROC
if len(np.unique(y_test_seq)) > 1:
    plt.figure(figsize=(8, 6))
    try:
        fpr, tpr, thresholds = roc_curve(y_test_seq, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC (Melhor Modelo)')
        plt.legend(loc="lower right")
        plt.savefig('tuned_lstm_roc_curve.png')
        plt.show()
    except Exception as e:
        print(f"Erro ao gerar curva ROC: {e}")
else:
    print("Curva ROC não gerada.")

# Plotar distribuição das probabilidades preditas
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=50, alpha=0.7, label='Probabilidades Preditas')
plt.axvline(x=threshold, color='r', linestyle='--', label=f'Limiar Padrão: {threshold:.2f}')
plt.title('Distribuição das Probabilidades Preditas (Melhor Modelo)')
plt.xlabel('Probabilidade (Saída Sigmoid)')
plt.ylabel('Frequência')
plt.legend()
plt.savefig('tuned_prediction_distribution.png')
plt.show()

# Plotar Preço Real vs. Sinais Previstos (Melhor Modelo)
print("\nGerando gráfico de Preço vs. Sinais Previstos (Melhor Modelo)...")
plt.figure(figsize=(18, 8))
plt.plot(plot_df.index, plot_df['Close'], label='Preço de Fechamento (Real)', color='skyblue', alpha=0.8, zorder=1)
buy_signals = plot_df[plot_df['Predicted'] == 1]
sell_signals = plot_df[plot_df['Predicted'] == 0]
plt.scatter(buy_signals.index, buy_signals['Close'], label='Compra Prevista (1)', marker='^', color='green', s=100, alpha=0.9, zorder=2)
plt.scatter(sell_signals.index, sell_signals['Close'], label='Venda Prevista (0)', marker='v', color='red', s=100, alpha=0.9, zorder=2)
plt.title('Preço Real vs. Sinais Previstos (Melhor Modelo LSTM Pós-Tuning)')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tuned_lstm_price_vs_predictions.png')
plt.show()

# --- 12. Salvar o Melhor Modelo ---
print("\nSalvando o MELHOR modelo treinado...")
try:
    os.makedirs('models', exist_ok=True)
    # Salvar o melhor modelo obtido pelo tuner
    best_model.save('models/tuned_lstm_model.keras')
    print("Melhor modelo salvo em 'models/tuned_lstm_model.keras'")
    # Salvar também os melhores HPs para referência
    with open('models/best_hyperparameters.txt', 'w') as f:
        for hp_name, hp_value in best_hps.values.items(): # Iterar sobre os itens do dicionário de HPs
            f.write(f"{hp_name}: {hp_value}\n")
    print("Melhores hiperparâmetros salvos em 'models/best_hyperparameters.txt'")

except Exception as e:
    print(f"Erro ao salvar o melhor modelo: {e}")

print("\nScript de Tuning finalizado.")
