import plotly.graph_objects as go
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
        'Partidos_Jugados', 'Minutos_Jugados_PP', 'Rol',
        'Porcentaje_Tiros_Efectivo'
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
    encoded = Dense(4, activation='relu')(encoded)
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

    # 1. Datos del lesionado
    perfil_lesionado = df[(df['Jugador'] == nombre) & (df['Temporada'] == temporada)].iloc[0] # Asegúrate de tener esta línea
    minutos_lesionado = perfil_lesionado['Minutos_Jugados_PP']
    eficiencia_lesionado = perfil_lesionado['Porcentaje_Tiros_Efectivo']
    
    # 2. Predicción de Puntos (con eFG%)
    factor_punteria = df_pool['Porcentaje_Tiros_Efectivo'] / eficiencia_lesionado
    df_pool['PTS_Predichos'] = (df_pool['Puntos_PP'] / df_pool['Minutos_Jugados_PP']) * minutos_lesionado * factor_punteria
    
    # 3. Predicción de Asistencias y Rebotes (por Minutos)
    df_pool['AST_Predichas'] = (df_pool['Asistencias_PP'] / df_pool['Minutos_Jugados_PP']) * minutos_lesionado
    df_pool['REB_Predichos'] = (df_pool['Rebotes_Totales_PP'] / df_pool['Minutos_Jugados_PP']) * minutos_lesionado
    
    # Redondeo
    columnas_pred = ['PTS_Predichos', 'AST_Predichas', 'REB_Predichos']
    df_pool[columnas_pred] = df_pool[columnas_pred].round(1)

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




    #Gráficos
    st.markdown("---")
    mejor_sustituto = resultados.iloc[0]
    nombre_sustituto = mejor_sustituto['Jugador']

    # 2. AQUÍ pones el título con el emoji
    st.markdown(f"### 📊 Comparativa Visual: {nombre_jugador} vs {nombre_sustituto}")
    # 1. Definimos los "Topes Reales" para que el gráfico sea proporcional
    limites_personalizados = {
        'Puntos_PP': 32.0,
        'Asistencias_PP': 15.0,
        'Rebotes_Totales_PP': 20.0,
        'Minutos_Jugados_PP': 48.0,
        'Edad': 40.0
    }
    categories = list(limites_personalizados.keys())

    # 2. Extraer datos
    perfil_lesionado = df[(df['Jugador'] == nombre_jugador) & (df['Temporada'] == temporada_seleccionada)].iloc[0]
    mejor_sustituto = resultados.iloc[0]

    # 3. Escalamos los valores para que "encajen" en el dibujo (de 0 a 1)
    # Esto hace que 10 rebotes se vean igual de importantes que 24 minutos (ambos al 50% del eje)
    val_original_radar = [perfil_lesionado[cat] / limites_personalizados[cat] for cat in categories]
    val_sustituto_radar = [mejor_sustituto[cat] / limites_personalizados[cat] for cat in categories]

    # 4. Creamos el gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=val_original_radar,
        theta=categories,
        fill='toself',
        name=f"Lesionado: {nombre_jugador}",
        hoverinfo="text",
        # El texto del ratón SIEMPRE muestra el valor real original
        text=[f"{perfil_lesionado[cat]} {cat}" for cat in categories],
        line_color='#1f77b4'
    ))

    fig.add_trace(go.Scatterpolar(
        r=val_sustituto_radar,
        theta=categories,
        fill='toself',
        name=f"Sustituto: {mejor_sustituto['Jugador']}",
        hoverinfo="text",
        text=[f"{mejor_sustituto[cat]} {cat}" for cat in categories],
        line_color='#ef553b'
    ))

    # 5. Configuración estética
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1], # El rango del dibujo es de 0 a 1 (0% a 100% del límite)
                tickvals=[0.25, 0.5, 0.75, 1],
                ticktext=['25%', '50%', '75%', 'Límite Máx.'],
                gridcolor="gray",
            )),
        showlegend=True,
        height=600,
        title=dict(text="Comparación de Rendimiento (Escala Proporcional)", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)


    #Predicciones
    st.markdown("---")
    # --- TABLA COMPARATIVA DE PREDICCIONES (TOP 5) ---
    perfil_lesionado = df[(df['Jugador'] == nombre_jugador) & (df['Temporada'] == temporada_seleccionada)].iloc[0]
    minutos_lesionado = perfil_lesionado['Minutos_Jugados_PP']
    st.markdown(f"### 📈 Proyección de Estadísticas (Rol de {minutos_lesionado} min)")
        
    tabla_comparativa = resultados[['Jugador', 'PTS_Predichos', 'AST_Predichas', 'REB_Predichos']].copy()
    tabla_comparativa.columns = ['Candidato', 'Puntos (Pred)', 'Asistencias (Pred)', 'Rebotes (Pred)']
        
    # Definimos qué columnas queremos resaltar (las de números)
    cols_a_resaltar = ['Puntos (Pred)', 'Asistencias (Pred)', 'Rebotes (Pred)']

    st.dataframe(
        tabla_comparativa.style.highlight_max(
            axis=0, 
            color='#2e7d32', 
            subset=cols_a_resaltar
        ), 
        use_container_width=True,
        hide_index=True
    )
    st.caption("Nota: Los puntos proyectados incluyen el ajuste por Porcentaje de Tiros Efectivo.")    

#uv venv --python 3.12
#.\.venv\Scripts\activate
#uv sync
#uv run streamlit run appdl.py