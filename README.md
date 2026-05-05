El objetivo del proyecto es que, dado un jugador y un año concretos, se pueda proporcionar otro jugador con las misma habilidades en dicha temporada. 

Durante cada temporada de la NBA cada equipo disputa 82 partidos fijos, con posibilidad de sumar hasta 20 partidos extras si compite en la post-temporada. Dicha carga de partidos puede suponer:
- Lesiones desafortunadas de jugadores, tanto estrellas como jugadores de rol.
- Fallos de confección de plantilla, las cuales pueden hacer que el equipo no rinda como se esperaba de él.

Por ello, hemos pensado en realizar una aplicación en la cual, seleccionando un jugador cualquiera (sea lesionado o por decisión de la directiva del equipo), se pueda encontrar el reemplazo que mejor se ajuste a su rol en el equipo.

cómo crear el entorno;?
ENTORNO :
uv venv --python 3.12
uv pip install --upgrade pip
uv pip install tensorflow --no-cache-dir
uv pip install pandas scikit-learn streamlit plotly
streamlit run app.py

cómo instalar las dependencias; ?
cómo ejecutar la aplicación;?
cualquier dato adicional necesario para reproducir el proyecto?

Una vez reproducido el proyecto, se mostrarán en la pantalla dos desplegables. 
En el primero aparece la lista de los jugadores existentes en la base de datos. También permite buscar al jugador que se desee escribiendo su nombre. Si este no se encuentra en la base de datos se mostrará el mensaje : "El jugador '{nombre_jugador}' no existe en la base de datos."
Una vez seleccioando el jugador, en el segundo desplegable aparece una lista de las temporadas en las que jugó dicho profesional. 
Cuando estén ambos campos especificados podremos pinchar en el botón "Buscar Sustituto Ideal", y nos aparecerá una tabla con los mejores 5 sustitutos para los datos que hemos indicado.
La tabla mostrará, para cada jugador, la posición principal, los puntos, asistencias y rebotes anotados por partido, y el rol.
