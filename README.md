¡¡¡EL ARCHIVO 'appp.py' ES INNECESARIO!!!

OBJETIVO:
El objetivo del proyecto es que, dado un jugador y un año concretos, se pueda proporcionar otro jugador con las misma habilidades en dicha temporada. 

Durante cada temporada de la NBA cada equipo disputa 82 partidos fijos, con posibilidad de sumar hasta 20 partidos extras si compite en la post-temporada. Dicha carga de partidos puede suponer:
- Lesiones desafortunadas de jugadores, tanto estrellas como jugadores de rol.
- Fallos de confección de plantilla, las cuales pueden hacer que el equipo no rinda como se esperaba de él.

Por ello, hemos pensado en realizar una aplicación en la cual, seleccionando un jugador cualquiera (sea lesionado o por decisión de la directiva del equipo), se pueda encontrar el reemplazo que mejor se ajuste a su rol en el equipo.

LIMPIEZA DE DATOS Y ANÁLISIS DESCRIPTIVO DE LAS VARIABLES:
En el archivo 'datos copia.ipynb' hemos realizado un análisis de cada variable, una limpieza del dataset original 'cleaned_data.csv', en el que hemos renombrado variables para una mayor facilidad a la hora de nombrarlas y eliminado variables que nos parecían redundantes.
Una vez realizado la limpieza de datos, hemos guardado este dataset en un archivo csv llamdo 'cleaned_data_final.csv'.
A continuación, se ha procedido a un análisis descriptivos de las variables restantes, en el que se pueden observar:
- Medias, Medianas y Diagramas de caja y bigote de cada variable.
- Matrices de Correlación de las variables.

ENTORNO:
uv venv --python 3.12
.\.venv\Scripts\activate
uv sync

DEPENDENCIAS:
Las dependencias necesarias vienen en el archivo "requirements.txt", y se instalan ejecutando en la terminal el siguiente código:
pip install 'dependencias'

LANZAR APLICACIÓN:
Una vez creado el entorno, sincronizarlo e instaladas las dependencias, basta con ejecutar la siguiente linea de código en la terminal:
uv run streamlit run appdl.py

También podremos acceder a la aplicación desde el siguiente enlace:
https://trabajoiae-b98rijigaqyzcjgqchrww5.streamlit.app/

INFORMACIÓN DE LA APLICACIÓN
Una vez reproducido el proyecto, se mostrarán en la pantalla dos desplegables. 
En el primero aparece la lista de los jugadores existentes en la base de datos. También permite buscar al jugador que se desee escribiendo su nombre. Si este no se encuentra en la base de datos se mostrará el mensaje : "El jugador '{nombre_jugador}' no existe en la base de datos."
Una vez seleccionado el jugador, en el segundo desplegable aparece una lista de las temporadas en las que jugó dicho profesional. 
Cuando estén ambos campos especificados podremos pinchar en el botón "Buscar Sustituto Ideal", y nos aparecerá lo siguiente:

- Una tabla la cual adjunta a los 5 jugadores con perfil más similar al jugador a reemplazar, con su posición, puntos, asistencias, rebotes y rol (hay que tener en cuenta que la aplicación posee un filtro para que solo aparezcan los jugadores que, como mínimo, han anotado un 70% de puntos del jugador a reemplazar.
- Un gráfico de radar en el que se compara la mejor alternativa con el jugador a reemplazar, comparando puntos, asistencias, rebotes, minutos y edad.
- Una tabla con la proyección de las estadísticas que haría cada alternativa con los minutos del jugador a reemplazar, teniendo en cuenta el eFG% de cada jugador.
