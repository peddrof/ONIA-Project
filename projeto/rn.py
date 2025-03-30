from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_data = pd.read_csv('treino.csv')
test_data = pd.read_csv('teste.csv')

X_train = train_data.drop(['id', 'target'], axis=1)
y_train = train_data['target']

X_test = test_data.drop('id', axis=1)
test_ids = test_data['id']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(200), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

predictions_df = pd.DataFrame({'id': test_ids, 'target': y_pred})

predictions_df.to_csv('resposta.csv', index=False)
print("Resposta salva em: 'resposta.csv'")