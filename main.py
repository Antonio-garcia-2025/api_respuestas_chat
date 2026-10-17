from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import random

app = FastAPI()

# Cargar el modelo y vectorizador recién entrenados
with open("modelo_nlp.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("vectorizador.pkl", "rb") as f:
    vectorizador = pickle.load(f)

# Cargar el cerebro de respuestas
with open("intents.json", "r", encoding="utf-8") as f:
    datos = json.load(f)

class Mensaje(BaseModel):
    texto: str

@app.post("/predecir")
def predecir(mensaje: Mensaje):
    try:
        # Vectorizar el texto del usuario
        X_nuevo = vectorizador.transform([mensaje.texto.lower()])
        
        # Predecir etiqueta y probabilidad
        prediccion = modelo.predict(X_nuevo)[0]
        probabilidades = modelo.predict_proba(X_nuevo)[0]
        max_prob = max(probabilidades)

        # Umbral de seguridad (30%)
        if max_prob < 0.30:
            return {"respuesta": "Lo siento, no entendí tu pregunta. ¿Podrías ser más específico?"}

        # Buscar la respuesta correspondiente a la etiqueta predicha
        for intent in datos["intents"]:
            if intent["tag"] == prediccion:
                respuesta = random.choice(intent["responses"])
                return {"respuesta": respuesta, "intencion": prediccion}

    except Exception as e:
        return {"respuesta": f"Error interno en la IA: {str(e)}"}