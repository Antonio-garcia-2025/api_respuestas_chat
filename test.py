import pickle

# 1. Cargamos el "cerebro" que acabas de crear
# Usamos 'rb' que significa "read binary" (leer archivo binario)
with open('modelo_nlp.pkl', 'rb') as file:
    modelo = pickle.load(file)

print("Chatbot de prueba iniciado (escribe 'salir' para terminar)")
print("-" * 50)

# 2. Creamos un bucle para poder enviarle varios mensajes
while True:
    mensaje = input("Tú: ")
    
    if mensaje.lower() == 'salir':
        print("Cerrando prueba...")
        break
        
    # 3. Le pedimos al modelo que adivine la intención
    # Le pasamos el texto en una lista [mensaje] porque el modelo espera conjuntos de datos
    # El [0] extrae el resultado de esa lista
    intencion_detectada = modelo.predict([mensaje])[0]
    
    print(f"IA: Detecté la intención -> [{intencion_detectada}]")