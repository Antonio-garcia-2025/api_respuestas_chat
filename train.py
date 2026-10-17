import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Iniciando entrenamiento del modelo para Mercado Shop...")

# 1. Cargar los datos del archivo JSON
with open('intents.json', 'r', encoding='utf-8') as archivo:
    datos = json.load(archivo)

textos = []
etiquetas = []

# Extraer los patrones (preguntas) y sus etiquetas (categorías)
for intent in datos['intents']:
    for pattern in intent['patterns']:
        textos.append(pattern.lower())  # Convertir a minúsculas para estandarizar
        etiquetas.append(intent['tag'])

# 2. Convertir texto a números aplicando el filtro de Stop Words
palabras_vacias = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'a', 'ante', 'con', 'en', 'para', 'por', 'y', 'o', 'mi', 'tu', 'su', 'que', 'es', 'son']

vectorizador = TfidfVectorizer(stop_words=palabras_vacias)
X = vectorizador.fit_transform(textos)

# 3. Entrenar el modelo de Regresión Logística
modelo = LogisticRegression(random_state=42)
modelo.fit(X, etiquetas)

# 4. Guardar el cerebro y el vectorizador en archivos .pkl
with open('vectorizador.pkl', 'wb') as f:
    pickle.dump(vectorizador, f)

with open('modelo_nlp.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("✅ ¡Entrenamiento completado con éxito!")
print("Se generaron los archivos 'modelo_nlp.pkl' y 'vectorizador.pkl' con el filtro de palabras activo.")