README – OPTIMIZACIÓN DEL PROCESO DE BIODIESEL (10.000 TON/AÑO)

Este proyecto contiene los códigos, modelos y herramientas utilizados para la optimización del proceso de producción de biodiesel, incluyendo modelos en Python, GAMS y archivos auxiliares como HINT para análisis Pinch y redes de intercambio de calor.

El proyecto incluye modelos de:

Optimización del reactor CSTR.

Modelado termodinámico.

Optimización multiobjetivo del proceso.

Diseño de la Red de Intercambio de Calor (HEN).

Herramientas de análisis y generación de gráficas.

1. ESTRUCTURA DEL REPOSITORIO
Proyecto/
│
├── 114479_Informe_Tecnico.pdf
├── README.txt
│
├── Problema 1/
│ ├── Problem_1.py
│
├── Problema 2/
│ ├── Problem_2.gms
│ ├── Problem_2.hint
│ ├── Problem_2.lst
│ ├── HINT_SETUP.zip (instalador del programa HINT)
│
├── Problema 3/
│ ├── Problem_3.py
│ ├── Problem_3-1.py
│ ├── Diagrama de Gantt de Producción.pdf
│
└── Problema 4/
├── Problem_4.py
├── Diagrama de Flujo de Proceso.pdf


2. LIBRERÍAS NECESARIAS (PYTHON)

Numericas y Científicas:

   numpy

   scipy

   pandas

   math

   copy

   random

   warnings

   isfinite

Optimización:

   pulp

  scipy.optimize

Gráficas:

  matplotlib

   mpl_toolkits.mplot3d

  matplotlib.gridspec

  matplotlib.patches

El proyecto también usa:

   sys

   deepcopy


3. REQUISITOS DEL SISTEMA

Requisitos mínimos recomendados:

  Hardware:
    - Procesador 2 núcleos (Intel i5 o equivalente)
    - 8 GB de RAM (recomendado 16 GB para GAMS)
    - 2 GB de espacio libre en disco
    - Resolución mínima de pantalla 1366x768

  Software:
    Python 3.8 o superior
    Pip actualizado
    GAMS 39 o superior
     GAMS debe tener activados los siguientes solvers:
      - CPLEX (LP y MIP)
      - CONOPT (NLP)
      - DICOPT (MINLP)
     El archivo .gms utiliza las siguientes opciones:
     - option LP = CPLEX;
     - option MIP = CPLEX;
     - option NLP = CONOPT;
     - option MINLP= DICOPT;
     - option OPTCR= 0;
     Para abrir el archivo HINT, es obligatorio instalar el programa HINT que se encuentra en la carpeta “Problema 2”.


4. INSTALACIÓN DE DEPENDENCIAS

Actualizar pip:
  pip install --upgrade pip

Instalar dependencias desde requirements.txt:
  pip install -r requirements.txt

Instalación manual:
  pip install numpy scipy pandas matplotlib Pulp


5. INSTRUCCIONES DE EJECUCIÓN

EJECUTAR CÓDIGOS PYTHON:

 - Abrir terminal o CMD.

 - Ubicarse en la carpeta donde está el archivo .py:
   cd Codigo_Python

  Ejecutar:
  - python nombre_del_archivo.py

EJECUTAR MODELO GAMS:

  - Abrir GAMS.

  - Cargar el archivo problema2.gms.

  - Ejecutar presionando la tecla F9.

ABRIR ARCHIVO HINT:

  - Abrir la carpeta “Problema 2”.

  - Instalar el programa HINT incluido allí.

  - Abrir el archivo .hint con dicha aplicación.

NOTAS IMPORTANTES

- Algunos scripts generan gráficas; se requiere un entorno con soporte gráfico (Jupyter, VSCode, PyCharm, Spyder, SublimeText, etc.).

- Si usas WSL en Windows, debes habilitar X11 para mostrar gráficas.

- Si GAMS muestra errores de licencia, verificar solvers habilitados.

SOPORTE

Para dudas o mejoras relacionadas con el proyecto, escribe al correro: elianam.mayorgach@ecci.edu.co
