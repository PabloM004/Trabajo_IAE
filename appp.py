import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Configuración de la página
st.set_page_config(page_title="NBA Injury Replacement Tool", layout="wide")

# 1. CARGA DE DATOS (Importación)
@st.cache_data # Para que la app sea rápida
def load_data():
    df = pd.read_csv('cleaned_data_final.csv') # Asegúrate de que el nombre coincida
    # Limpieza básica inicial
    df['Temporada'] = df['Temporada'].fillna(0).astype(int) # Convertimos años a int limpio
    return df

df = load_data()

# 2. TÍTULO Y DESCRIPCIÓN (Definición del problema)
st.title("🏀 NBA Injury Replacement Tool")
st.markdown("""
Esta aplicación utiliza **Machine Learning (KNN)** para encontrar el sustituto ideal de un jugador lesionado, 
asegurando que el reemplazo juegue en la **misma posición** y esté activo en la **misma temporada**.
""")

# 3. INTERFAZ DE USUARIO (Comunicación)
col1, col2 = st.columns(2)

with col1:
    # Buscador de jugador
    nombre_jugador = st.selectbox("Selecciona al jugador lesionado:", sorted(df['Jugador'].unique()))

with col2:
    # Filtramos las temporadas disponibles para ESE jugador específicamente
    temporadas_disponibles = sorted(df[df['Jugador'] == nombre_jugador]['Temporada'].unique(), reverse=True)
    temporada_seleccionada = st.selectbox("Año de la sustitución:", temporadas_disponibles)

# 4. LÓGICA DEL MODELO (Modelización)
def obtener_recomendacion(nombre, temporada):
    # Filtro por posición y año
    perfil_lesionado = df[(df['Jugador'] == nombre) & (df['Temporada'] == temporada)].iloc[0]
    posicion = perfil_lesionado['Posicion_Principal']
    
    df_pool = df[(df['Posicion_Principal'] == posicion) & (df['Temporada'] == temporada)].copy()
    
    # Atributos y Pesos
    pesos = {
        'Edad': 0.5, 'Partidos_Jugados': 0.7, 'Minutos_Jugados_PP': 1.2,
        'Porcentaje_Tiros_Efectivo': 1.0, 'Puntos_PP': 1.5, 'Asistencias_PP': 1.5,
        'Rebotes_Totales_PP': 1.2, 'Tapones_PP': 1.0, 'Robos_PP': 1.0
    }
    features = list(pesos.keys())
    
    # Preprocesamiento (Escalado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pool[features])
    
    # Aplicar Ponderación
    for i, col in enumerate(features):
        X_scaled[:, i] *= pesos[col]
        
    # KNN
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(X_scaled)
    
    # Buscar vecinos
    idx_jugador = df_pool[df_pool['Jugador'] == nombre].index[0]
    pos_en_matriz = df_pool.index.get_loc(idx_jugador)
    distancias, indices = model.kneighbors([X_scaled[pos_en_matriz]])
    
    return df_pool.iloc[indices[0][1:]] # Retornamos los 5 más parecidos

# 5. BOTÓN Y RESULTADOS
if st.button("Buscar Sustituto Ideal"):
    resultados = obtener_recomendacion(nombre_jugador, temporada_seleccionada)
    
    st.subheader(f"Sustitutos recomendados para {nombre_jugador} ({temporada_seleccionada})")
    
    # Mostramos los resultados en una tabla bonita
    st.dataframe(resultados[['Jugador', 'Posicion_Principal', 'Puntos_PP', 'Asistencias_PP', 'Rebotes_Totales_PP', 'Rol']], use_container_width=True)
    
    # Visualización (Opcional: Gráfico comparativo)
    st.info("💡 Estos jugadores tienen el perfil estadístico más similar en su posición para esa temporada.")