import streamlit as st
import pandas as pd
import tensorflow as tf
import dask.dataframe as dd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Herramienta para Reemplazar Jugadores", layout="wide")

# 2. CARGA DE DATOS
@st.cache_data
def load_data_big_data():
    df_dask = dd.read_csv('cleaned_data_final.csv')
    
    columnas_necesarias = [
        'Jugador', 'Temporada', 'Posicion_Principal', 'Edad', 
        'Puntos_PP', 'Asistencias_PP', 'Rebotes_Totales_PP',
        'Partidos_Jugados', 'Minutos_Jugados_PP', 'Rol'
    ]
    
    df_dask = df_dask[columnas_necesarias]
    
    return df_dask.compute().fillna(0)

df = load_data_big_data()

# 3. INTERFAZ: TÍTULO Y SELECTORES
st.title("🏀 Reemplazamiento de Jugadores de la NBA")
st.markdown("Esta app usa **Deep Learning (Autoencoders)** para buscar sustitutos.")

col1, col2 = st.columns(2)
with col1:
    nombre_jugador = st.selectbox("Selecciona al jugador lesionado:", sorted(df['Jugador'].unique()))

with col2:
    temporadas_disponibles = sorted(df[df['Jugador'] == nombre_jugador]['Temporada'].unique().astype(int), reverse=True)
    temporada_seleccionada = st.selectbox("Año de la sustitución:", temporadas_disponibles)

# 4. FUNCIÓN DE DEEP LEARNING
def obtener_recomendacion_dl(nombre, temporada):
    # Filtro por posición y año
    perfil_lesionado = df[(df['Jugador'] == nombre) & (df['Temporada'] == temporada)].iloc[0]
    posicion = perfil_lesionado['Posicion_Principal']
    df_pool = df[(df['Posicion_Principal'] == posicion) & (df['Temporada'] == temporada)].copy().reset_index(drop=True)
    
    # Atributos para la Red Neuronal
    features = ['Edad', 'Partidos_Jugados', 'Minutos_Jugados_PP', 'Puntos_PP', 'Asistencias_PP', 'Rebotes_Totales_PP']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_pool[features])
    
    input_dim = len(features)
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation='relu')(input_layer)
    encoded = Dense(4, activation='relu')(encoded) # Bottleneck
    decoded = Dense(8, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, output_layer)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Entrenamiento
    autoencoder.fit(X, X, epochs=50, batch_size=16, verbose=0)
    
    huellas_profundas = encoder.predict(X)
    
    # Calcular similitud
    idx_jugador = df_pool[df_pool['Jugador'] == nombre].index[0]
    similitudes = cosine_similarity([huellas_profundas[idx_jugador]], huellas_profundas)[0]
    
    df_pool['Similitud'] = similitudes
    return df_pool.sort_values(by='Similitud', ascending=False).iloc[1:6]

# 5. EL BOTÓN QUE DISPARA LA TABLA
if st.button("🚀 Buscar Sustituto Ideal"):
    with st.spinner('La Red Neuronal está analizando a los jugadores...'):
        # Llamada a la función
        resultados = obtener_recomendacion_dl(nombre_jugador, temporada_seleccionada)
        
        # Mostrar la tabla
        st.subheader(f"Top 5 Sustitutos para {nombre_jugador} en {temporada_seleccionada}")
        
        # Definir qué columnas quieres ver en la tabla
        columnas_visibles = ['Jugador', 'Posicion_Principal', 'Puntos_PP', 'Asistencias_PP', 'Rebotes_Totales_PP', 'Rol']
        
        st.dataframe(resultados[columnas_visibles], use_container_width=True)
        
        st.success("✅ Análisis de Deep Learning completado con éxito.")

#uv venv --python 3.12
#.\.venv\Scripts\activate
#uv sync
#uv run streamlit run main.py