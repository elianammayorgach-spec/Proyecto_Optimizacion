#***************************************************#
# Problem 4: Process Optimization Project           #
# Optimización Global - Planta de Biodiesel
# Professor: Francisco Javier Vasquez Vasquez       #
# EMMCH                                            #
#***************************************************#

from matplotlib.gridspec import GridSpec
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from math import isfinite
import matplotlib.gridspec as gridspec
import sys

# Configuración para caracteres especiales y encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Establecer semillas para reproducibilidad
random.seed(42)
np.random.seed(42)

# =============================================
# PARÁMETROS DEL PROCESO Y DATOS ECONÓMICOS
# =============================================

# Producción objetivo anual
produccion_anual_objetivo = 10000.0  # ton/año de biodiesel

# Precios de materias primas (USD/ton)
precio_aceite = 800.0
precio_metanol = 400.0
precio_naoh = 600.0

# Precios de biodiesel por trimestre (USD/ton)
precio_biodiesel_trimestral = {
    'Ene-Mar': 1250.0,
    'Abr-Jun': 1300.0,
    'Jul-Sep': 1350.0,
    'Oct-Dic': 1300.0
}

# Demanda mensual por periodo (ton/mes)
demanda_por_periodo = {'Ene-Mar': 700, 'Abr-Jun': 900, 'Jul-Sep': 1000, 'Oct-Dic': 800}

# Calcular demanda anual total
demanda_anual_total = 700*3 + 900*3 + 1000*3 + 800*3

# Calcular precio promedio ponderado del biodiesel
precio_promedio_ponderado = (1250*700*3 + 1300*900*3 + 1350*1000*3 + 1300*800*3) / demanda_anual_total

# Precio de subproductos y servicios
precio_glicerol = 500.0  # USD/ton
precio_electricidad = 0.12  # USD/kWh
precio_vapor_por_tonelada = 25.0  # USD/ton vapor

# Conversión de vapor a energía (kWh/ton)
factor_conversion_vapor_kwh = 2790.0
precio_vapor_por_kwh = precio_vapor_por_tonelada / factor_conversion_vapor_kwh

# =============================================
# PARÁMETROS DE RENDIMIENTO DEL PROCESO
# =============================================

rendimiento_biodiesel_aceite = 0.999  # Rendimiento de biodiesel por tonelada de aceite
consumo_metanol_por_aceite = 0.38  # Fracción másica de metanol consumido por aceite procesado
rendimiento_glicerol_por_aceite = 0.856  # ton glicerol por ton aceite procesado

# =============================================
# PARÁMETROS ENERGÉTICOS Y DE COSTOS
# =============================================

# Consumo energético base
consumo_energia_termica_base = 500.0  # kWh térmico por ton biodiesel base
consumo_energia_electrica_base = 50.0  # kWh eléctrico por ton biodiesel base

# Parámetros CAPEX para intercambiadores
area_intercambiadores_base = 1  
coeficiente_capex_intercambiadores = 500.0
costo_base_intercambiadores = 15000.0
factor_anualizacion = 0.15  # factor de anualización

# Costos específicos por tipo de equipo
costo_etapa_destilacion = 18000.0  # USD/etapa para columnas de destilación
costo_etapa_absorbedor = 12000.0   # USD/etapa para absorbedor (menor costo)
costo_unidad_flash = 80000.0       # USD por unidad flash (costo fijo por equipo)

# Configuración fija de los equipos
etapas_columna_secundaria = 12     # Etapas columna secundaria
etapas_columna_terciaria = 8       # Etapas columna terciaria  
etapas_absorbedor = 15             # Etapas del absorbedor
unidades_flash = 2                 # 2 unidades flash

# Factores de complejidad
factor_materiales_destilacion = 1.25  # Acero inoxidable
factor_materiales_absorbedor = 1.15   # Materiales menos exigentes
factor_instalacion = 1.2              # Instalación y equipos auxiliares

# Costos de almacenamiento
costo_almacenamiento_mensual = 5.0  # USD/(ton·mes)
capacidad_almacenamiento_maxima = 300.0  # ton

# Precio promedio de biodiesel para cálculos
precio_venta_biodiesel = precio_promedio_ponderado

# =============================================
# LÍMITES DE VARIABLES DE DECISIÓN
# =============================================

limites_variables = {
    'relacion_reflujo_principal': (1.2, 6.0),      # Reflujo columna principal
    'numero_etapas_principal': (6, 30),            # Número de etapas (entero)
    'relacion_reflujo_secundaria': (1.0, 8.0),     # Reflujo columna secundaria
    'factor_escala_area': (0.6, 2.0)               # Factor de escala área intercambiadores
}

def calcular_capex_equipos_separacion(numero_etapas_principal):
    """
    Calcula el CAPEX total para todos los equipos de separación
    considerando la configuración completa de la planta
    """
    
    # 1. COLUMNA DE DESTILACIÓN PRINCIPAL
    capex_principal = costo_etapa_destilacion * numero_etapas_principal * factor_materiales_destilacion
    
    # 2. COLUMNA DE DESTILACIÓN SECUNDARIA
    capex_secundaria = costo_etapa_destilacion * etapas_columna_secundaria * factor_materiales_destilacion
    
    # 3. COLUMNA DE DESTILACIÓN TERCIARIA
    capex_terciaria = costo_etapa_destilacion * etapas_columna_terciaria * factor_materiales_destilacion
    
    # 4. ABSORBEDOR
    capex_absorbedor = costo_etapa_absorbedor * etapas_absorbedor * factor_materiales_absorbedor
    
    # 5. UNIDADES FLASH 
    capex_flash = costo_unidad_flash * unidades_flash
    
    # Costo base total de los equipos
    capex_equipos_base = capex_principal + capex_secundaria + capex_terciaria + capex_absorbedor + capex_flash
    
    # Aplicar factor de instalación y equipos auxiliares
    capex_total = capex_equipos_base * factor_instalacion
    
    # Desglose para análisis
    desglose_capex = {
        'columna_principal': capex_principal,
        'columna_secundaria': capex_secundaria,
        'columna_terciaria': capex_terciaria,
        'absorbedor': capex_absorbedor,
        'unidades_flash': capex_flash,
        'total_sin_instalacion': capex_equipos_base,
        'total_con_instalacion': capex_total
    }
    
    return capex_total, desglose_capex

# =============================================
# FUNCIÓN DE EVALUACIÓN DE DISEÑO 
# =============================================

def evaluar_configuracion_planta(configuracion):
    # Extraer variables de decisión
    relacion_reflujo_principal = configuracion['relacion_reflujo_principal']
    numero_etapas_principal = int(round(configuracion['numero_etapas_principal']))
    relacion_reflujo_secundaria = configuracion['relacion_reflujo_secundaria']
    factor_escala_area = configuracion['factor_escala_area']
    
    produccion_anual = produccion_anual_objetivo
    
    # Cálculos de materiales e ingresos 
    aceite_requerido = produccion_anual / rendimiento_biodiesel_aceite
    metanol_requerido = aceite_requerido * consumo_metanol_por_aceite
    naoh_requerido = aceite_requerido * 0.008
    
    ingresos_por_biodiesel = produccion_anual * precio_venta_biodiesel
    produccion_glicerol = aceite_requerido * rendimiento_glicerol_por_aceite
    ingresos_por_glicerol = produccion_glicerol * precio_glicerol
    ingresos_totales = ingresos_por_biodiesel + ingresos_por_glicerol
    
    costo_aceite = aceite_requerido * precio_aceite
    costo_metanol = metanol_requerido * precio_metanol
    costo_naoh = naoh_requerido * precio_naoh
    
    # =============================================
    # CÁLCULOS DE CAPEX
    # =============================================
    
    # 1. CAPEX INTERCAMBIADORES 
    area_total_intercambiadores = area_intercambiadores_base * produccion_anual * factor_escala_area
    
    # Modelo más realista para intercambiadores
    if area_total_intercambiadores <= 5000:
        capex_intercambiadores = costo_base_intercambiadores + 300 * area_total_intercambiadores
    elif area_total_intercambiadores <= 20000:
        capex_intercambiadores = costo_base_intercambiadores + 250 * (area_total_intercambiadores ** 0.9)
    else:
        capex_intercambiadores = costo_base_intercambiadores + 200 * (area_total_intercambiadores ** 0.85)
    
    capex_anual_intercambiadores = capex_intercambiadores * factor_anualizacion
    
    # 2. CAPEX EQUIPOS DE SEPARACIÓN 
    capex_equipos_separacion, desglose_separacion = calcular_capex_equipos_separacion(numero_etapas_principal)
    capex_anual_separacion = capex_equipos_separacion * factor_anualizacion
    
    # 3. CONSUMO ENERGÉTICO 
    consumo_energia_termica = consumo_energia_termica_base * produccion_anual * (0.85 / factor_escala_area)
    consumo_energia_electrica = (consumo_energia_electrica_base * produccion_anual * 
                               (1.0 + 0.012*(numero_etapas_principal-10) + 
                                0.006*(relacion_reflujo_principal-2.0) + 
                                0.006*(relacion_reflujo_secundaria-2.0) +
                                0.02)) 
    
    consumo_vapor_kwh = consumo_energia_termica * 0.85
    consumo_vapor_toneladas = consumo_vapor_kwh / factor_conversion_vapor_kwh
    
    costo_electricidad = consumo_energia_electrica * precio_electricidad
    costo_vapor = consumo_vapor_toneladas * precio_vapor_por_tonelada
    
    # 4. OTROS COSTOS
    costo_almacenamiento_anual = (capacidad_almacenamiento_maxima/2.0) * costo_almacenamiento_mensual * 12.0
    costo_operacion_mantenimiento = (capex_intercambiadores * 0.02 + 
                                   capex_equipos_separacion * 0.025)  #
    
    # =============================================
    # CÁLCULOS FINALES
    # =============================================
    
    costos_totales = (costo_aceite + costo_metanol + costo_naoh + 
                     costo_electricidad + costo_vapor + capex_anual_intercambiadores + 
                     capex_anual_separacion + costo_almacenamiento_anual + costo_operacion_mantenimiento)
    
    beneficio_anual = ingresos_totales - costos_totales
    
    pureza_biodiesel = (0.9 + 0.018*(numero_etapas_principal/10.0) + 
                       0.035*math.log(relacion_reflujo_secundaria+1.0) +
                       0.015)  
    
    penalizacion_pureza = 0.0
    if pureza_biodiesel < 0.965:
        penalizacion_pureza = -2e6 * (0.965 - pureza_biodiesel)
    
    bonificacion_pureza = 0.0
    if pureza_biodiesel > 0.98:
        bonificacion_pureza = 75000 * (pureza_biodiesel - 0.98)
    
    consumo_energetico_total = consumo_energia_termica + consumo_energia_electrica
    
    # Métricas de evaluación 
    metricas_evaluacion = {
        'beneficio_anual': beneficio_anual + penalizacion_pureza + bonificacion_pureza,
        'consumo_energetico_total': consumo_energetico_total,
        'aceite_requerido_anual': aceite_requerido,
        'metanol_requerido_anual': metanol_requerido,
        'naoh_requerido_anual': naoh_requerido,
        'glicerol_producido_anual': produccion_glicerol,
        'capex_intercambiadores': capex_intercambiadores,
        'capex_equipos_separacion': capex_equipos_separacion,
        'area_intercambiadores': area_total_intercambiadores,
        'numero_etapas_principal': numero_etapas_principal,
        'pureza_estimada': pureza_biodiesel,
        'costos_operativos_anuales': costos_totales,
        'ingresos_totales': ingresos_totales,
        'costo_materias_primas': costo_aceite + costo_metanol + costo_naoh,
        'costo_energia': costo_electricidad + costo_vapor,
        'capex_total': capex_intercambiadores + capex_equipos_separacion,
        'desglose_separacion': desglose_separacion
    }
    
    return metricas_evaluacion

# =============================================
# ALGORITMO NSGA-II
# =============================================

def ordenar_soluciones_no_dominadas(objetivos_soluciones):
    """Implementa el algoritmo de ordenamiento no dominado (Non-dominated Sorting)
que clasifica las soluciones en diferentes frentes de Pareto según su 
dominancia relativa en el espacio multiobjetivo.
"""
    S = [[] for _ in range(len(objetivos_soluciones))]
    n = [0] * len(objetivos_soluciones)
    rank = [0] * len(objetivos_soluciones)
    fronts = [[]]
    
    for p in range(len(objetivos_soluciones)):
        S[p] = []
        n[p] = 0
        for q in range(len(objetivos_soluciones)):
            if p == q:
                continue
            if (objetivos_soluciones[p][0] <= objetivos_soluciones[q][0] and 
                objetivos_soluciones[p][1] <= objetivos_soluciones[q][1]) and \
               (objetivos_soluciones[p][0] < objetivos_soluciones[q][0] or 
                objetivos_soluciones[p][1] < objetivos_soluciones[q][1]):
                S[p].append(q)
            elif (objetivos_soluciones[q][0] <= objetivos_soluciones[p][0] and 
                  objetivos_soluciones[q][1] <= objetivos_soluciones[p][1]) and \
                 (objetivos_soluciones[q][0] < objetivos_soluciones[p][0] or 
                  objetivos_soluciones[q][1] < objetivos_soluciones[p][1]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()
    return fronts

def calcular_distancia_agrupamiento(front, objs):
    """Calcula la distancia de agrupamiento (crowding distance) para mantener 
diversidad en el frente de Pareto, priorizando soluciones en regiones 
menos pobladas del espacio de objetivos.
"""
    l = len(front)
    dist = {i:0.0 for i in front}
    if l <= 2:
        for i in front:
            dist[i] = float('inf')
        return dist
    for m in range(len(objs[0])):
        values = sorted(front, key=lambda x: objs[x][m])
        fmin = objs[values[0]][m]
        fmax = objs[values[-1]][m]
        dist[values[0]] = float('inf')
        dist[values[-1]] = float('inf')
        if fmax == fmin:
            continue
        for i in range(1, len(values)-1):
            prevv = objs[values[i-1]][m]
            nextv = objs[values[i+1]][m]
            dist[values[i]] += (nextv - prevv) / (fmax - fmin)
    return dist

def cruce_variables_binarias_simulado(x1, x2, lb, ub, eta=15):
    """Implementa el operador de cruce SBX (Simulated Binary Crossover) que 
simula el comportamiento del cruce single-point de algoritmos genéticos 
binarios para variables continuas.
"""
    if random.random() > 0.9:
        return x1, x2
    y1 = min(x1, x2)
    y2 = max(x1, x2)
    rand = random.random()
    if rand <= 0.5:
        beta = (2 * rand) ** (1.0/(eta + 1))
    else:
        beta = (1/(2 * (1 - rand))) ** (1.0/(eta + 1))
    child1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    child2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
    child1 = min(max(child1, lb), ub)
    child2 = min(max(child2, lb), ub)
    return child1, child2

def mutacion_variables_polinomial(x, lb, ub, eta=20, pm=0.2):
    """ Implementa el operador de mutación polinomial que perturba soluciones 
existentes introduciendo diversidad y permitiendo explorar regiones 
nuevas del espacio de búsqueda.
"""
    if random.random() > pm:
        return x
    u = random.random()
    delta1 = (x - lb) / (ub - lb)
    delta2 = (ub - x) / (ub - lb)
    mut_pow = 1.0/(eta + 1.0)
    if u < 0.5:
        xy = 1.0 - delta1
        val = 2 * u + (1 - 2 * u) * (xy ** (eta + 1))
        deltaq = val ** mut_pow - 1.0
    else:
        xy = 1.0 - delta2
        val = 2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta + 1))
        deltaq = 1.0 - val ** mut_pow
    x = x + deltaq * (ub - lb)
    x = min(max(x, lb), ub)
    return x

# =============================================
# INICIALIZACIÓN Y EVOLUCIÓN
# =============================================

tamano_poblacion = 120
numero_generaciones = 120

def generar_configuracion_aleatoria():
    return {
        'relacion_reflujo_principal': random.uniform(*limites_variables['relacion_reflujo_principal']),
        'numero_etapas_principal': random.uniform(*limites_variables['numero_etapas_principal']),
        'relacion_reflujo_secundaria': random.uniform(*limites_variables['relacion_reflujo_secundaria']),
        'factor_escala_area': random.uniform(*limites_variables['factor_escala_area'])
    }

# Crear población inicial
poblacion_configuraciones = [generar_configuracion_aleatoria() for _ in range(tamano_poblacion)]
objetivos_poblacion_actual = []
metricas_poblacion_actual = []

# Evaluar población inicial
for configuracion in poblacion_configuraciones:
    metricas_evaluacion = evaluar_configuracion_planta(configuracion)
    metricas_poblacion_actual.append(metricas_evaluacion)
    objetivos_poblacion_actual.append((-metricas_evaluacion['beneficio_anual'], 
                                     metricas_evaluacion['consumo_energetico_total']))

# Bucle evolutivo principal
for generacion_actual in range(numero_generaciones):
    nuevas_configuraciones = []
    
    while len(nuevas_configuraciones) < tamano_poblacion:
        i1, i2 = random.sample(range(tamano_poblacion), 2)
        configuracion_padre1 = poblacion_configuraciones[i1] if random.random() < 0.5 else poblacion_configuraciones[i2]
        
        i3, i4 = random.sample(range(tamano_poblacion), 2)
        configuracion_padre2 = poblacion_configuraciones[i3] if random.random() < 0.5 else poblacion_configuraciones[i4]
        
        configuracion_descendiente = {}
        
        # Cruce para cada variable
        nuevo_reflujo_principal, _ = cruce_variables_binarias_simulado(
            configuracion_padre1['relacion_reflujo_principal'], 
            configuracion_padre2['relacion_reflujo_principal'],
            limites_variables['relacion_reflujo_principal'][0],
            limites_variables['relacion_reflujo_principal'][1]
        )
        configuracion_descendiente['relacion_reflujo_principal'] = nuevo_reflujo_principal
        
        nuevas_etapas_principal, _ = cruce_variables_binarias_simulado(
            configuracion_padre1['numero_etapas_principal'], 
            configuracion_padre2['numero_etapas_principal'],
            limites_variables['numero_etapas_principal'][0],
            limites_variables['numero_etapas_principal'][1]
        )
        configuracion_descendiente['numero_etapas_principal'] = nuevas_etapas_principal
        
        nuevo_reflujo_secundaria, _ = cruce_variables_binarias_simulado(
            configuracion_padre1['relacion_reflujo_secundaria'], 
            configuracion_padre2['relacion_reflujo_secundaria'],
            limites_variables['relacion_reflujo_secundaria'][0],
            limites_variables['relacion_reflujo_secundaria'][1]
        )
        configuracion_descendiente['relacion_reflujo_secundaria'] = nuevo_reflujo_secundaria
        
        nuevo_factor_escala, _ = cruce_variables_binarias_simulado(
            configuracion_padre1['factor_escala_area'], 
            configuracion_padre2['factor_escala_area'],
            limites_variables['factor_escala_area'][0],
            limites_variables['factor_escala_area'][1]
        )
        configuracion_descendiente['factor_escala_area'] = nuevo_factor_escala
        
        # Mutación
        configuracion_descendiente['relacion_reflujo_principal'] = mutacion_variables_polinomial(
            configuracion_descendiente['relacion_reflujo_principal'],
            limites_variables['relacion_reflujo_principal'][0],
            limites_variables['relacion_reflujo_principal'][1]
        )
        
        configuracion_descendiente['numero_etapas_principal'] = mutacion_variables_polinomial(
            configuracion_descendiente['numero_etapas_principal'],
            limites_variables['numero_etapas_principal'][0],
            limites_variables['numero_etapas_principal'][1]
        )
        
        configuracion_descendiente['relacion_reflujo_secundaria'] = mutacion_variables_polinomial(
            configuracion_descendiente['relacion_reflujo_secundaria'],
            limites_variables['relacion_reflujo_secundaria'][0],
            limites_variables['relacion_reflujo_secundaria'][1]
        )
        
        configuracion_descendiente['factor_escala_area'] = mutacion_variables_polinomial(
            configuracion_descendiente['factor_escala_area'],
            limites_variables['factor_escala_area'][0],
            limites_variables['factor_escala_area'][1]
        )
        
        nuevas_configuraciones.append(configuracion_descendiente)
    
    # Evaluar nuevas configuraciones
    metricas_nuevas_configuraciones = [evaluar_configuracion_planta(config) for config in nuevas_configuraciones]
    objetivos_nuevas_configuraciones = [(-metricas['beneficio_anual'], metricas['consumo_energetico_total']) 
                                      for metricas in metricas_nuevas_configuraciones]
    
    # Combinar población actual y nuevas configuraciones
    poblacion_combinada = poblacion_configuraciones + nuevas_configuraciones
    metricas_combinadas = metricas_poblacion_actual + metricas_nuevas_configuraciones
    objetivos_combinados = objetivos_poblacion_actual + objetivos_nuevas_configuraciones
    
    # Ordenamiento no dominado
    frentes_optimizacion = ordenar_soluciones_no_dominadas(objetivos_combinados)
    
    # Crear nueva población
    nueva_poblacion_configuraciones = []
    nuevas_metricas_poblacion = []
    nuevos_objetivos_poblacion = []
    
    for frente_actual in frentes_optimizacion:
        if len(nueva_poblacion_configuraciones) + len(frente_actual) > tamano_poblacion:
            distancias_frente = calcular_distancia_agrupamiento(frente_actual, objetivos_combinados)
            frente_ordenado = sorted(frente_actual, key=lambda x: distancias_frente[x], reverse=True)
            
            for indice_solucion in frente_ordenado:
                if len(nueva_poblacion_configuraciones) >= tamano_poblacion:
                    break
                nueva_poblacion_configuraciones.append(poblacion_combinada[indice_solucion])
                nuevas_metricas_poblacion.append(metricas_combinadas[indice_solucion])
                nuevos_objetivos_poblacion.append(objetivos_combinados[indice_solucion])
            break
        else:
            for indice_solucion in frente_actual:
                nueva_poblacion_configuraciones.append(poblacion_combinada[indice_solucion])
                nuevas_metricas_poblacion.append(metricas_combinadas[indice_solucion])
                nuevos_objetivos_poblacion.append(objetivos_combinados[indice_solucion])
    
    poblacion_configuraciones = nueva_poblacion_configuraciones
    metricas_poblacion_actual = nuevas_metricas_poblacion
    objetivos_poblacion_actual = nuevos_objetivos_poblacion
    
    # Mostrar progreso
    if generacion_actual % 20 == 0 or generacion_actual == numero_generaciones-1:
        mejor_beneficio = max([metricas['beneficio_anual'] for metricas in metricas_poblacion_actual])
        minima_energia = min([metricas['consumo_energetico_total'] for metricas in metricas_poblacion_actual])
        print(f"Generación {generacion_actual}: mejor beneficio = {mejor_beneficio:,.0f} USD/año, "
              f"mínima energía = {minima_energia:,.0f} kWh/año")

# =============================================
# ANÁLISIS DE RESULTADOS Y FRONTERA DE PARETO
# =============================================

# Extraer frente de Pareto
objetivos_finales = [(-metricas['beneficio_anual'], metricas['consumo_energetico_total']) 
                     for metricas in metricas_poblacion_actual]
frentes_finales = ordenar_soluciones_no_dominadas(objetivos_finales)
indices_pareto = frentes_finales[0]

configuraciones_optimas = []
for indice_optimo in indices_pareto:
    configuracion = poblacion_configuraciones[indice_optimo]
    metricas = metricas_poblacion_actual[indice_optimo]
    configuraciones_optimas.append({
        'relacion_reflujo_principal': configuracion['relacion_reflujo_principal'],
        'numero_etapas_principal': int(round(configuracion['numero_etapas_principal'])),
        'relacion_reflujo_secundaria': configuracion['relacion_reflujo_secundaria'],
        'factor_escala_area': configuracion['factor_escala_area'],
        'beneficio_anual': metricas['beneficio_anual'],
        'consumo_energetico_total': metricas['consumo_energetico_total'],
        'aceite_requerido_anual': metricas['aceite_requerido_anual'],
        'metanol_requerido_anual': metricas['metanol_requerido_anual'],
        'naoh_requerido_anual': metricas['naoh_requerido_anual'],  # AGREGADO
        'glicerol_producido_anual': metricas['glicerol_producido_anual'],  # AGREGADO
        'pureza_estimada': metricas['pureza_estimada'],
        'capex_intercambiadores': metricas['capex_intercambiadores'],
        'capex_equipos_separacion': metricas['capex_equipos_separacion'],
        'costos_operativos_anuales': metricas['costos_operativos_anuales'],
        'ingresos_totales': metricas['ingresos_totales'],
        'costo_materias_primas': metricas['costo_materias_primas'],
        'costo_energia': metricas['costo_energia']
    })

# Ordenar configuraciones por energía creciente
configuraciones_optimas = sorted(configuraciones_optimas, key=lambda x: x['consumo_energetico_total'])

# Asegurar al menos 10 configuraciones óptimas
if len(configuraciones_optimas) < 10:
    configuraciones_adicionales = []
    for frente_secundario in frentes_finales[1:]:
        for indice_secundario in frente_secundario:
            configuracion = poblacion_configuraciones[indice_secundario]
            metricas = metricas_poblacion_actual[indice_secundario]
            configuraciones_adicionales.append({
                'relacion_reflujo_principal': configuracion['relacion_reflujo_principal'],
                'numero_etapas_principal': int(round(configuracion['numero_etapas_principal'])),
                'relacion_reflujo_secundaria': configuracion['relacion_reflujo_secundaria'],
                'factor_escala_area': configuracion['factor_escala_area'],
                'beneficio_anual': metricas['beneficio_anual'],
                'consumo_energetico_total': metricas['consumo_energetico_total'],
                'aceite_requerido_anual': metricas['aceite_requerido_anual'],
                'metanol_requerido_anual': metricas['metanol_requerido_anual'],
                'naoh_requerido_anual': metricas['naoh_requerido_anual'],  # AGREGADO
                'glicerol_producido_anual': metricas['glicerol_producido_anual'],  # AGREGADO
                'pureza_estimada': metricas['pureza_estimada'],
                'capex_intercambiadores': metricas['capex_intercambiadores'],
                'capex_equipos_separacion': metricas['capex_equipos_separacion'],
                'costos_operativos_anuales': metricas['costos_operativos_anuales'],
                'ingresos_totales': metricas['ingresos_totales'],
                'costo_materias_primas': metricas['costo_materias_primas'],
                'costo_energia': metricas['costo_energia']
            })
    configuraciones_optimas += configuraciones_adicionales[:max(0, 10 - len(configuraciones_optimas))]

# Crear DataFrame con resultados
dataframe_configuraciones_optimas = pd.DataFrame(configuraciones_optimas)
pd.options.display.float_format = '{:,.2f}'.format

# Mostrar primeras 15 configuraciones óptimas
print("\nPrimeras 15 configuraciones óptimas de la Frontera de Pareto:")
print(dataframe_configuraciones_optimas.head(15))

# =============================================
# GRÁFICA CON PUNTO DESTACADO
# =============================================

# Configurar estilo de fuente y colores
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

# Extraer datos para la gráfica
beneficios_optimos = dataframe_configuraciones_optimas['beneficio_anual'].values
consumos_energeticos = dataframe_configuraciones_optimas['consumo_energetico_total'].values

# Encontrar punto de compromiso (knee-point)
beneficio_maximo = max(beneficios_optimos)
beneficio_minimo = min(beneficios_optimos)
consumo_maximo = max(consumos_energeticos)
consumo_minimo = min(consumos_energeticos)

puntuaciones_compromiso = []
for beneficio, consumo in zip(beneficios_optimos, consumos_energeticos):
    beneficio_normalizado = (beneficio - beneficio_minimo) / (beneficio_maximo - beneficio_minimo) if beneficio_maximo > beneficio_minimo else 1.0
    consumo_normalizado = (consumo - consumo_minimo) / (consumo_maximo - consumo_minimo) if consumo_maximo > consumo_minimo else 0.0
    puntuaciones_compromiso.append(beneficio_normalizado - consumo_normalizado)

mejor_indice_compromiso = int(np.argmax(puntuaciones_compromiso))
configuracion_compromiso = dataframe_configuraciones_optimas.iloc[mejor_indice_compromiso]

# Crear figura 
plt.figure(figsize=(10, 7))

# Graficar todos los puntos de Pareto
plt.scatter(consumos_energeticos, beneficios_optimos, 
           c='lightblue', alpha=0.7, s=60, 
           edgecolors='steelblue', linewidth=0.5,
           label='Soluciones Pareto')

# Destacar el punto de compromiso 
plt.scatter(configuracion_compromiso['consumo_energetico_total'], 
           configuracion_compromiso['beneficio_anual'],
           c='red', s=200, marker='*', 
           edgecolors='darkred', linewidth=2,
           label='Solución Óptima de Compromiso')

plt.xlabel('Consumo Energético Total (kWh/año)', fontsize=12, fontweight='bold')
plt.ylabel('Beneficio Anual (USD/año)', fontsize=12, fontweight='bold')
plt.title('Frontera de Pareto - Optimización de Planta de Biodiesel\nBeneficio vs Consumo Energético', 
          fontsize=14, fontweight='bold', pad=20)

plt.annotate('Solución Óptima', 
             xy=(configuracion_compromiso['consumo_energetico_total'], 
                 configuracion_compromiso['beneficio_anual']),
             xytext=(20, 20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.grid(True, linestyle='--', alpha=0.7, color='gray')
plt.gca().set_facecolor('whitesmoke')
plt.legend()
plt.tight_layout()

plt.show()

# =============================================
# INFORMACIÓN DETALLADA DE LA SOLUCIÓN SELECCIONADA
# =============================================

def mostrar_analisis_detallado(configuracion_compromiso):
    """
    Muestra un análisis detallado de los costos de capital
    """
    print("\n" + "="*70)
    print("ANÁLISIS DETALLADO DE COSTOS DE CAPITAL")
    print("="*70)
    
    # Recalcular para obtener el desglose
    numero_etapas = int(round(configuracion_compromiso['numero_etapas_principal']))
    capex_separacion, desglose = calcular_capex_equipos_separacion(numero_etapas)
    
    # Calcular área de intercambiadores
    area_interc = area_intercambiadores_base * 10000 * configuracion_compromiso['factor_escala_area']
    
    print(f"\n--- EQUIPOS DE SEPARACIÓN (Total: ${capex_separacion:,.2f}) ---")
    print(f"  • Columna Principal ({numero_etapas} etapas): ${desglose['columna_principal']:,.2f}")
    print(f"  • Columna Secundaria ({etapas_columna_secundaria} etapas): ${desglose['columna_secundaria']:,.2f}")
    print(f"  • Columna Terciaria ({etapas_columna_terciaria} etapas): ${desglose['columna_terciaria']:,.2f}")
    print(f"  • Absorbedor ({etapas_absorbedor} etapas): ${desglose['absorbedor']:,.2f}")
    print(f"  • Unidades Flash ({unidades_flash} unidades): ${desglose['unidades_flash']:,.2f}")
    print(f"  • Instalación y auxiliares (x{factor_instalacion}): ${desglose['total_con_instalacion'] - desglose['total_sin_instalacion']:,.2f}")
    
    print(f"\n--- INTERCAMBIADORES DE CALOR ---")
    print(f"  • Área total: {area_interc:,.0f} m²")
    print(f"  • CAPEX intercambiadores: ${configuracion_compromiso['capex_intercambiadores']:,.2f}")
    
    print(f"\n--- DISTRIBUCIÓN DE COSTOS DE CAPITAL ---")
    total_capex = configuracion_compromiso['capex_intercambiadores'] + capex_separacion
    porcentaje_interc = (configuracion_compromiso['capex_intercambiadores'] / total_capex) * 100
    porcentaje_separacion = (capex_separacion / total_capex) * 100
    
    print(f"  • Intercambiadores: {porcentaje_interc:.1f}%")
    print(f"  • Equipos de separación: {porcentaje_separacion:.1f}%")
    print(f"  • TOTAL CAPEX: ${total_capex:,.2f}")

# Mostrar configuración de compromiso CON INGRESOS TOTALES
detalles_configuracion_compromiso = {
    'Relación de Reflujo Principal': configuracion_compromiso['relacion_reflujo_principal'],
    'Número de Etapas Principal': configuracion_compromiso['numero_etapas_principal'],
    'Relación de Reflujo Secundaria': configuracion_compromiso['relacion_reflujo_secundaria'],
    'Factor de Escala de Área': configuracion_compromiso['factor_escala_area'],
    'Beneficio Anual (USD)': configuracion_compromiso['beneficio_anual'],
    'Ingresos Totales (USD)': configuracion_compromiso['ingresos_totales'],
    'Costos Operativos Anuales (USD)': configuracion_compromiso['costos_operativos_anuales'],
    'Consumo Energético Total (kWh)': configuracion_compromiso['consumo_energetico_total'],
    'Aceite Requerido Anual (ton)': configuracion_compromiso['aceite_requerido_anual'],
    'Metanol Requerido Anual (ton)': configuracion_compromiso['metanol_requerido_anual'],
    'NaOH Requerido Anual (ton)': configuracion_compromiso['naoh_requerido_anual'],  # CORREGIDO
    'Glicerol Producido Anual (ton)': configuracion_compromiso['glicerol_producido_anual'],  # CORREGIDO
    'Pureza Estimada del Biodiesel': configuracion_compromiso['pureza_estimada'],
    'Costo Intercambiadores (USD)': configuracion_compromiso['capex_intercambiadores'],
    'Costo Equipos Separación (USD)': configuracion_compromiso['capex_equipos_separacion'],
    'Costo Materias Primas (USD)': configuracion_compromiso['costo_materias_primas'],
    'Costo Energía (USD)': configuracion_compromiso['costo_energia']
}

print("\n" + "="*70)
print("CONFIGURACIÓN ÓPTIMA DE COMPROMISO SELECCIONADA")
print("="*70)

# Agrupar la información por categorías
print("\n--- PARÁMETROS DE DISEÑO ---")
for param in ['Relación de Reflujo Principal', 'Número de Etapas Principal', 
              'Relación de Reflujo Secundaria', 'Factor de Escala de Área']:
    print(f" - {param}: {detalles_configuracion_compromiso[param]:.2f}")

print("\n--- RESULTADOS ECONÓMICOS ---")
for param in ['Ingresos Totales (USD)', 'Costos Operativos Anuales (USD)', 
              'Beneficio Anual (USD)', 'Costo Materias Primas (USD)', 'Costo Energía (USD)']:
    print(f" - {param}: {detalles_configuracion_compromiso[param]:,.2f}")

print("\n--- INDICADORES TÉCNICOS ---")
for param in ['Consumo Energético Total (kWh)', 'Aceite Requerido Anual (ton)',
              'Metanol Requerido Anual (ton)', 'NaOH Requerido Anual (ton)', 
              'Glicerol Producido Anual (ton)', 'Pureza Estimada del Biodiesel']:  
    if 'Pureza' in param:
        print(f" - {param}: {detalles_configuracion_compromiso[param]:.3f}")
    elif 'Consumo' in param:
        print(f" - {param}: {detalles_configuracion_compromiso[param]:,.0f}")
    else:
        print(f" - {param}: {detalles_configuracion_compromiso[param]:,.2f}")

print("\n--- INVERSIONES DE CAPITAL ---")
for param in ['Costo Intercambiadores (USD)', 'Costo Equipos Separación (USD)']:
    print(f" - {param}: {detalles_configuracion_compromiso[param]:,.2f}")

# Calcular y mostrar márgenes
margen_beneficio = (detalles_configuracion_compromiso['Beneficio Anual (USD)'] / 
                    detalles_configuracion_compromiso['Ingresos Totales (USD)']) * 100
costo_por_tonelada = detalles_configuracion_compromiso['Costos Operativos Anuales (USD)'] / 10000

print(f"\n--- INDICADORES DE RENTABILIDAD ---")
print(f" - Margen de Beneficio: {margen_beneficio:.1f}%")
print(f" - Costo por Tonelada de Biodiesel: ${costo_por_tonelada:.2f}/ton")
print(f" - Relación Beneficio/Energía: ${detalles_configuracion_compromiso['Beneficio Anual (USD)']/detalles_configuracion_compromiso['Consumo Energético Total (kWh)']:.4f}/kWh")

mostrar_analisis_detallado(configuracion_compromiso)

print("\n" + "="*70)

# =============================================
# BALANCE DE MATERIA Y ENERGÍA DEL PROCESO OPTIMIZADO
# =============================================

print("\n" + "="*80)
print("BALANCE DE MATERIA Y ENERGÍA - PROCESO OPTIMIZADO")
print("="*80)

# Convertir producción anual a horaria (asumiendo 8000 horas de operación anual)
horas_operacion_anual = 8000
produccion_horaria = produccion_anual_objetivo * 1000 / horas_operacion_anual  # kg/h

# Calcular flujos másicos horarios basados en la configuración óptima
aceite_horario = configuracion_compromiso['aceite_requerido_anual'] * 1000 / horas_operacion_anual
metanol_horario = configuracion_compromiso['metanol_requerido_anual'] * 1000 / horas_operacion_anual
naoh_horario = configuracion_compromiso['naoh_requerido_anual'] * 1000 / horas_operacion_anual
glicerol_horario = configuracion_compromiso['glicerol_producido_anual'] * 1000 / horas_operacion_anual

# Calcular consumo energético horario
consumo_energia_horario = configuracion_compromiso['consumo_energetico_total'] / horas_operacion_anual

print(f"\nPRODUCCIÓN HORARIA: {produccion_horaria:.2f} kg/h de Biodiesel")
print(f"HORAS DE OPERACIÓN ANUAL: {horas_operacion_anual:,} horas")

# =============================================
# BALANCE DE MATERIA DETALLADO
# =============================================

print("\n" + "-"*50)
print("BALANCE DE MATERIA - CORRIENTES PRINCIPALES")
print("-"*50)

# 1. MEZCLADOR M-101 y BOMBA P-101
print("\n1. MEZCLADOR M-101 y BOMBA P-101:")
print(f"   Entrada Metanol: {metanol_horario:.2f} kg/h a 25°C, 100 kPa")
print(f"   Entrada NaOH: {naoh_horario:.2f} kg/h a 25°C, 100 kPa")
print(f"   Salida Mezcla: {metanol_horario + naoh_horario:.2f} kg/h a 25°C, 400 kPa")

# 2. MEZCLADOR M-102
print(f"\n2. MEZCLADOR M-102:")
print(f"   Entrada de M-101: {metanol_horario + naoh_horario:.2f} kg/h")
print(f"   Recirculación T-201: {metanol_horario * 0.1:.2f} kg/h")  
print(f"   Salida al Reactor: {metanol_horario + naoh_horario + metanol_horario * 0.1:.2f} kg/h")

# 3. LÍNEA DE ACEITE - BOMBA P-102
print(f"\n3. LÍNEA DE ACEITE - BOMBA P-102:")
print(f"   Entrada Aceite: {aceite_horario:.2f} kg/h a 26.7°C, 100 kPa")
print(f"   Salida Aceite: {aceite_horario:.2f} kg/h a 60°C, 400 kPa")

# 4. REACTOR R-101
print(f"\n4. REACTOR R-101 (Transesterificación):")
flujo_reactor_salida = aceite_horario + metanol_horario + naoh_horario + metanol_horario * 0.1
print(f"   Entrada Total: {flujo_reactor_salida:.2f} kg/h a 60°C, 400 kPa")
print(f"   Salida Total: {flujo_reactor_salida:.2f} kg/h a 60°C, 400 kPa")
print(f"   Composición de Salida:")
print(f"     - Metanol: {flujo_reactor_salida * 0.092:.2f} kg/h (9.2%)")
print(f"     - Aceite Palma: {flujo_reactor_salida * 0.041:.2f} kg/h (4.1%)")
print(f"     - Biodiesel: {flujo_reactor_salida * 0.779:.2f} kg/h (77.9%)")
print(f"     - Glicerol: {flujo_reactor_salida * 0.081:.2f} kg/h (8.1%)")
print(f"     - NaOH: {flujo_reactor_salida * 0.008:.2f} kg/h (0.8%)")

# 5. TORRE DE DESTILACIÓN T-201
print(f"\n5. TORRE DE DESTILACIÓN T-201:")
print(f"   Entrada: {flujo_reactor_salida:.2f} kg/h")
corriente_cimas_t201 = flujo_reactor_salida * 0.086  
corriente_fondos_t201 = flujo_reactor_salida * 0.914  
print(f"   Corriente Cimas: {corriente_cimas_t201:.2f} kg/h a 28.2°C, 20 kPa")
print(f"   Corriente Fondos: {corriente_fondos_t201:.2f} kg/h a 122.34°C, 30 kPa")

# 6. BOMBA P-201 (Recirculación)
print(f"\n6. BOMBA P-201 (Recirculación):")
print(f"   Entrada: {corriente_cimas_t201:.2f} kg/h a 28.2°C, 20 kPa")
print(f"   Salida: {corriente_cimas_t201:.2f} kg/h a 28.2°C, 400 kPa")

# 7. ABSORBEDOR T-202
print(f"\n7. ABSORBEDOR T-202:")
agua_absorbedor = corriente_fondos_t201 * 0.1  # Estimado
print(f"   Entrada Fondos T-201: {corriente_fondos_t201:.2f} kg/h")
print(f"   Entrada Agua: {agua_absorbedor:.2f} kg/h")
print(f"   Corriente Superior: {corriente_fondos_t201 * 0.9:.2f} kg/h (a Flash T-203)")
print(f"   Corriente Inferior: {corriente_fondos_t201 * 0.1 + agua_absorbedor:.2f} kg/h (a Reactor R-301)")

# 8. FLASH T-203
print(f"\n8. FLASH T-203:")
flujo_flash_t203 = corriente_fondos_t201 * 0.9
print(f"   Entrada: {flujo_flash_t203:.2f} kg/h")
print(f"   Salida a T-301: {flujo_flash_t203 * 0.95:.2f} kg/h a 60°C, 110 kPa")

# 9. TORRE DE DESTILACIÓN T-301
print(f"\n9. TORRE DE DESTILACIÓN T-301 (Principal):")
flujo_entrada_t301 = flujo_flash_t203 * 0.95
biodiesel_puro = produccion_horaria
mezcla_metanol_agua = flujo_entrada_t301 * 0.007  
fondos_t301 = flujo_entrada_t301 * 0.05  

print(f"   Entrada: {flujo_entrada_t301:.2f} kg/h")
print(f"   Biodiesel Puro: {biodiesel_puro:.2f} kg/h a 193.7°C, 10 kPa")
print(f"     - Biodiesel: {biodiesel_puro * 0.997:.2f} kg/h (99.7%)")
print(f"     - Agua: {biodiesel_puro * 0.003:.2f} kg/h (0.3%)")
print(f"   Mezcla Metanol-Agua: {mezcla_metanol_agua:.2f} kg/h a 193.7°C, 10 kPa")
print(f"     - Metanol: {mezcla_metanol_agua * 0.338:.2f} kg/h (33.8%)")
print(f"     - Biodiesel: {mezcla_metanol_agua * 0.504:.2f} kg/h (50.4%)")
print(f"     - Agua: {mezcla_metanol_agua * 0.107:.2f} kg/h (10.7%)")
print(f"   Fondos T-301: {fondos_t301:.2f} kg/h a 414.7°C, 20 kPa")
print(f"     - Aceite Palma: {fondos_t301 * 0.998:.2f} kg/h (99.8%)")
print(f"     - Agua: {fondos_t301 * 0.002:.2f} kg/h (0.2%)")

# 10. REACTOR R-301 y FLASH T-302
print(f"\n10. REACTOR R-301 y FLASH T-302:")
flujo_absorbedor_inferior = corriente_fondos_t201 * 0.1 + agua_absorbedor
h3po4_entrada = flujo_absorbedor_inferior * 0.02  
print(f"   Entrada Absorbedor Inferior: {flujo_absorbedor_inferior:.2f} kg/h")
print(f"   Entrada H3PO4: {h3po4_entrada:.2f} kg/h")
print(f"   Salida Flash T-302 Inferior: {flujo_absorbedor_inferior * 0.1:.2f} kg/h a 60°C, 110 kPa")
print(f"     - H3PO4: {flujo_absorbedor_inferior * 0.1 * 0.005:.2f} kg/h (0.5%)")
print(f"     - Na3PO4: {flujo_absorbedor_inferior * 0.1 * 0.995:.2f} kg/h (99.5%)")

# 11. TORRE DE DESTILACIÓN T-303
print(f"\n11. TORRE DE DESTILACIÓN T-303:")
flujo_t303_entrada = flujo_absorbedor_inferior * 0.9
metanol_agua_t303 = flujo_t303_entrada * 0.08  
glicerol_agua_t303 = flujo_t303_entrada * 0.92  

print(f"   Entrada: {flujo_t303_entrada:.2f} kg/h")
print(f"   Cimas - Metanol/Agua: {metanol_agua_t303:.2f} kg/h a 56.2°C, 40 kPa")
print(f"     - Metanol: {metanol_agua_t303 * 0.363:.2f} kg/h (36.3%)")
print(f"     - Agua: {metanol_agua_t303 * 0.637:.2f} kg/h (63.7%)")
print(f"   Fondos - Glicerol/Agua: {glicerol_agua_t303:.2f} kg/h a 112°C, 50 kPa")
print(f"     - Glicerol: {glicerol_agua_t303 * 0.85:.2f} kg/h (85.0%)")
print(f"     - Agua: {glicerol_agua_t303 * 0.15:.2f} kg/h (15.0%)")

# =============================================
# BALANCE DE ENERGÍA
# =============================================

print("\n" + "-"*50)
print("BALANCE DE ENERGÍA")
print("-"*50)

# Consumo energético total
print(f"\nCONSUMO ENERGÉTICO TOTAL:")
print(f"   Anual: {configuracion_compromiso['consumo_energetico_total']:,.0f} kWh/año")
print(f"   Horario: {consumo_energia_horario:.2f} kWh/h")
print(f"   Específico: {configuracion_compromiso['consumo_energetico_total']/produccion_anual_objetivo:.2f} kWh/ton biodiesel")

# Desglose por equipos principales 
print(f"\nDESGLOSE ESTIMADO DE CONSUMO ENERGÉTICO:")
print(f"   Intercambiadores de Calor: {consumo_energia_horario * 0.4:.2f} kWh/h (40%)")
print(f"   Columnas de Destilación: {consumo_energia_horario * 0.35:.2f} kWh/h (35%)")
print(f"   Bombas: {consumo_energia_horario * 0.15:.2f} kWh/h (15%)")
print(f"   Reactor y Otros: {consumo_energia_horario * 0.1:.2f} kWh/h (10%)")

# Requerimientos térmicos específicos
print(f"\nREQUERIMIENTOS TÉRMICOS:")
calor_reactor = 60  # kW (estimado para mantener 60°C)
calor_destilacion = consumo_energia_horario * 0.6  # kW 
print(f"   Reactor R-101: {calor_reactor:.2f} kW (mantenimiento a 60°C)")
print(f"   Torres de Destilación: {calor_destilacion:.2f} kW")
print(f"   Intercambiadores: {consumo_energia_horario * 0.4:.2f} kW")

# =============================================
# EFICIENCIAS Y MÉTRICAS DEL PROCESO
# =============================================

print("\n" + "-"*50)
print("EFICIENCIAS Y MÉTRICAS DEL PROCESO")
print("-"*50)

# Eficiencia global
eficiencia_global = (produccion_horaria / (aceite_horario + metanol_horario)) * 100
print(f"Eficiencia Global del Proceso: {eficiencia_global:.2f}%")

# Conversión en reactor
conversion_reactor = (1 - (flujo_reactor_salida * 0.0025) / aceite_horario) * 100
print(f"Conversión en Reactor R-101: {conversion_reactor:.2f}%")

# Recuperación de productos
recuperacion_biodiesel = (biodiesel_puro / (flujo_reactor_salida * 0.779)) * 100
recuperacion_glicerol = (glicerol_agua_t303 * 0.85 / (flujo_reactor_salida * 0.081)) * 100

print(f"Recuperación de Biodiesel: {recuperacion_biodiesel:.2f}%")
print(f"Recuperación de Glicerol: {recuperacion_glicerol:.2f}%")

# Purezas
print(f"Pureza Biodiesel Final: {configuracion_compromiso['pureza_estimada']*100:.2f}%")
print(f"Pureza Glicerol Final: {85.5}%")

# =============================================
# RESUMEN DE FLUJOS MÁSICOS
# =============================================

print("\n" + "-"*50)
print("RESUMEN DE FLUJOS MÁSICOS PRINCIPALES")
print("-"*50)

print(f"\nENTRADAS:")
print(f"   Aceite de Palma: {aceite_horario:.2f} kg/h")
print(f"   Metanol Fresco: {metanol_horario:.2f} kg/h")
print(f"   NaOH: {naoh_horario:.2f} kg/h")
print(f"   Agua: {agua_absorbedor:.2f} kg/h")
print(f"   H3PO4: {h3po4_entrada:.2f} kg/h")

print(f"\nSALIDAS:")
print(f"   Biodiesel: {biodiesel_puro:.2f} kg/h")
print(f"   Glicerol: {glicerol_agua_t303 * 0.85:.2f} kg/h")
print(f"   Na3PO4: {flujo_absorbedor_inferior * 0.1 * 0.995:.2f} kg/h")
print(f"   Mezcla Metanol-Agua: {metanol_agua_t303:.2f} kg/h")
print(f"   Residuos: {fondos_t301 + flujo_absorbedor_inferior * 0.1 * 0.005:.2f} kg/h")

print(f"\nBALANCE: Entradas - Salidas = {(aceite_horario + metanol_horario + naoh_horario + agua_absorbedor + h3po4_entrada) - (biodiesel_puro + glicerol_agua_t303 * 0.85 + flujo_absorbedor_inferior * 0.1 * 0.995 + metanol_agua_t303 + fondos_t301 + flujo_absorbedor_inferior * 0.1 * 0.005):.2f} kg/h")

print("\n" + "="*80)
print("FIN DEL BALANCE DE MATERIA Y ENERGÍA")
print("="*80)

dataframe_configuraciones_optimas.head(15)

# =============================================
# FACTORES AMBIENTALES Y ANÁLISIS DE EMISIONES
# =============================================

print("\n" + "="*80)
print("ANÁLISIS AMBIENTAL - HUELLA DE CARBONO Y CRÉDITOS")
print("="*80)

# Factores de emisión
factor_emision_electricidad = 0.15  # kg CO2/kWh (Colombia)
factor_emision_gas_natural = 2.75   # kg CO2/m³
rango_creditos_carbono = (10, 15)   # USD/ton CO2

# Cálculo de emisiones
emisiones_electricidad = configuracion_compromiso['consumo_energetico_total'] * factor_emision_electricidad / 1000  # ton CO2
emisiones_vapor = (configuracion_compromiso['costo_energia'] / precio_vapor_por_kwh) * factor_emision_gas_natural / 1000  # ton CO2 (estimado)

emisiones_totales_co2 = emisiones_electricidad + emisiones_vapor

# Cálculo de créditos de carbono
creditos_minimos = emisiones_totales_co2 * rango_creditos_carbono[0]
creditos_maximos = emisiones_totales_co2 * rango_creditos_carbono[1]

print(f"\n--- EMISIONES DE CO2 ---")
print(f"Emisiones por electricidad: {emisiones_electricidad:,.2f} ton CO2/año")
print(f"Emisiones por vapor: {emisiones_vapor:,.2f} ton CO2/año")
print(f"TOTAL EMISIONES: {emisiones_totales_co2:,.2f} ton CO2/año")
print(f"Intensidad de emisiones: {emisiones_totales_co2/produccion_anual_objetivo:.2f} ton CO2/ton biodiesel")

print(f"\n--- CRÉDITOS DE CARBONO ---")
print(f"Valor potencial de créditos: ${creditos_minimos:,.2f} - ${creditos_maximos:,.2f} USD/año")
print(f"Rango de precio: ${rango_creditos_carbono[0]} - ${rango_creditos_carbono[1]} por ton CO2")

# Beneficio considerando créditos de carbono
beneficio_con_creditos_min = configuracion_compromiso['beneficio_anual'] + creditos_minimos
beneficio_con_creditos_max = configuracion_compromiso['beneficio_anual'] + creditos_maximos

print(f"\n--- IMPACTO EN RENTABILIDAD ---")
print(f"Beneficio base: ${configuracion_compromiso['beneficio_anual']:,.2f} USD/año")
print(f"Beneficio con créditos mínimos: ${beneficio_con_creditos_min:,.2f} USD/año")
print(f"Beneficio con créditos máximos: ${beneficio_con_creditos_max:,.2f} USD/año")
print(f"Incremento potencial: {((beneficio_con_creditos_max/configuracion_compromiso['beneficio_anual'])-1)*100:.1f}%")

# =============================================
# GRÁFICAS AMBIENTALES Y DE CONTROL
# =============================================

plt.style.use('seaborn-v0_8')

fig = plt.figure(figsize=(15, 10))
fig.suptitle('ANÁLISIS AMBIENTAL Y PROYECCIONES - PLANTA DE BIODIESEL',
             fontsize=12, fontweight='bold', fontfamily='Times New Roman')

gs = gridspec.GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])  
ax2 = fig.add_subplot(gs[0, 1])   

# =======================
# Gráfica 1: Diagrama de Pastel
# =======================
categorias_emisiones = ['Electricidad', 'Vapor/Gas Natural']
emisiones = [emisiones_electricidad, emisiones_vapor]
colores = ['#ff9999', '#66b3ff']

ax1.pie(
    emisiones,
    labels=categorias_emisiones,
    autopct='%1.1f%%',
    colors=colores,
    startangle=60
)
ax1.set_title('Distribución de Emisiones de CO₂',
              fontweight='bold', fontfamily='Times New Roman')

# =======================
# Gráfica 2: Diagrama de Barras
# =======================
escenarios = ['Base', 'Créditos\nMínimos', 'Créditos\nMáximos']
beneficios = [
    configuracion_compromiso['beneficio_anual'],
    beneficio_con_creditos_min,
    beneficio_con_creditos_max
]
colores_barras = ['lightgray', 'lightgreen', 'darkgreen']

barras = ax2.bar(escenarios, beneficios, color=colores_barras, alpha=0.8)
ax2.set_ylabel('Beneficio Anual (USD)', fontfamily='Times New Roman')
ax2.set_title('Impacto de Créditos de Carbono en Rentabilidad',
              fontweight='bold', fontfamily='Times New Roman')

# Etiquetas
for barra, valor in zip(barras, beneficios):
    height = barra.get_height()
    ax2.text(
        barra.get_x() + barra.get_width() / 2.,
        height + height * 0.02,
        f'${valor/1e6:.1f}M',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# =============================================
# ANÁLISIS DE SENSIBILIDAD AMBIENTAL
# =============================================

print("\n" + "-"*50)
print("ANÁLISIS DE SENSIBILIDAD AMBIENTAL")
print("-"*50)

# Escenarios de mejora de eficiencia
mejoras_eficiencia = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20% de mejora

print(f"\nEscenarios de Mejora de Eficiencia Energética:")
print("Mejora | Consumo Energético | Emisiones CO2 | Reducción")
print("-" * 65)

for mejora in mejoras_eficiencia:
    consumo_mejorado = configuracion_compromiso['consumo_energetico_total'] * (1 - mejora)
    emisiones_mejoradas = consumo_mejorado * factor_emision_electricidad / 1000
    reduccion_emisiones = emisiones_totales_co2 - emisiones_mejoradas
    reduccion_porcentaje = (reduccion_emisiones / emisiones_totales_co2) * 100
    
    print(f"{mejora*100:2.0f}%   | {consumo_mejorado:12,.0f} kWh | {emisiones_mejoradas:8.0f} ton | {-reduccion_porcentaje:6.1f}%")

# =============================================
# GRÁFICA DE SENSIBILIDAD - MEJORAS DE EFICIENCIA
# =============================================

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(1, 2, figure=fig)

# =========================
# Subgráfica 1: Comparación con otras industrias
# =========================
ax3 = fig.add_subplot(gs[0, 0])  

industrias = ['Biodiesel\n(Esta Planta)', 'Diesel\nFósil', 'Gasolina', 'Electricidad\n(Carbón)']
intensidad_emisiones = [
    emisiones_totales_co2 / produccion_anual_objetivo,
    3.2,
    2.3,
    1.0
]

colores_industrias = ['green', 'red', 'orange', 'brown']

barras_industrias = ax3.bar(industrias, intensidad_emisiones,
                            color=colores_industrias, alpha=0.7)

ax3.set_ylabel('Intensidad de Emisiones (ton CO₂/ton producto)',
               fontfamily='Times New Roman')
ax3.set_title('Comparación de Intensidad de Emisiones',
              fontweight='bold', fontfamily='Times New Roman')

ax3.set_xticklabels(industrias, rotation=45, ha='right')

# =========================
# Subgráfica 2: Trayectoria de descarbonización
# =========================
ax4 = fig.add_subplot(gs[0, 1])  

años = np.arange(2024, 2035)
emisiones_base = emisiones_totales_co2
reduccion_anual = 0.03  # 3% anual

trayectoria_emisiones = [
    emisiones_base * (1 - reduccion_anual)**(año - 2024) for año in años
]

objetivo_2030 = emisiones_base * 0.7  

ax4.plot(años, trayectoria_emisiones, 's-', linewidth=2, markersize=6,
         label='Trayectoria Proyectada')

ax4.axhline(y=objetivo_2030, color='red', linestyle='--',
            label='Objetivo 2030 (30% reducción)')

ax4.fill_between(años, trayectoria_emisiones, alpha=0.3)

ax4.set_xlabel('Año', fontfamily='Times New Roman')
ax4.set_ylabel('Emisiones de CO₂ (ton/año)', fontfamily='Times New Roman')
ax4.set_title('Trayectoria de Descarbonización 2024-2034',
              fontweight='bold', fontfamily='Times New Roman')

ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================
# RECOMENDACIONES Y PLAN DE ACCIÓN AMBIENTAL
# =============================================

print("\n" + "="*80)
print("RECOMENDACIONES Y PLAN DE ACCIÓN AMBIENTAL")
print("="*80)

print(f"\n1. GESTIÓN DE EMISIONES:")
print(f"   • Emisiones actuales: {emisiones_totales_co2:,.0f} ton CO₂/año")
print(f"   • Objetivo recomendado: {emisiones_totales_co2 * 0.8:,.0f} ton CO₂/año (20% reducción)")
print(f"   • Potencial de créditos: ${creditos_minimos:,.0f} - ${creditos_maximos:,.0f} USD/año")

print(f"\n2. ESTRATEGIAS DE MITIGACIÓN:")
print(f"   • Mejora eficiencia energética: Hasta {mejoras_eficiencia[-1]*100:.0f}% de reducción")
print(f"   • Optimización de procesos: {reduccion_porcentaje:.1f}% reducción potencial")
print(f"   • Fuentes renovables: Solar para hasta 30% del consumo eléctrico")

print(f"\n3. PROYECCIÓN FINANCIERA:")
print(f"   • Inversión recomendada en eficiencia: ${configuracion_compromiso['costos_operativos_anuales'] * 0.1:,.0f} USD")
print(f"   • Retorno por créditos carbono: 2-3 años")
print(f"   • Beneficio incremental: Hasta ${creditos_maximos - creditos_minimos:,.0f} USD/año")

print(f"\n4. CUMPLIMIENTO REGULATORIO:")
print(f"   • Intensidad emisiones actual: {emisiones_totales_co2/produccion_anual_objetivo:.2f} ton CO₂/ton biodiesel")
print(f"   • Límite recomendado: <2.0 ton CO₂/ton biodiesel")

# =============================================
# MÉTRICAS AMBIENTALES CLAVE (KPIs)
# =============================================

print("\n" + "-"*50)
print("INDICADORES AMBIENTALES CLAVE (KPIs)")
print("-"*50)

kpis_ambientales = {
    'Intensidad Emisiones CO2': f"{emisiones_totales_co2/produccion_anual_objetivo:.2f} ton CO₂/ton biodiesel",
    'Eficiencia Energética': f"{produccion_anual_objetivo/(configuracion_compromiso['consumo_energetico_total']/1000):.1f} ton biodiesel/MWh",
    'Huella de Carbono Total': f"{emisiones_totales_co2:,.0f} ton CO₂/año",
    'Valor Créditos Carbono': f"${creditos_minimos:,.0f} - ${creditos_maximos:,.0f} USD/año",
    'Reducción vs Diesel Fósil': f"{(3.2 - (emisiones_totales_co2/produccion_anual_objetivo))/3.2*100:.1f}%",
    'Potencial Mejora Eficiencia': f"Hasta {mejoras_eficiencia[-1]*100:.0f}%"
}

for kpi, valor in kpis_ambientales.items():
    print(f"   • {kpi}: {valor}")

print("\n" + "="*80)
print("ANÁLISIS AMBIENTAL COMPLETADO")
print("="*80)

# Retornar resumen final de todo el análisis
print(f"\nRESUMEN EJECUTIVO:")
print(f"✅ Producción optimizada: {produccion_anual_objetivo:,.0f} ton biodiesel/año")
print(f"✅ Beneficio económico: ${configuracion_compromiso['beneficio_anual']:,.0f} USD/año")
print(f"✅ Huella carbono: {emisiones_totales_co2:,.0f} ton CO₂/año")
print(f"✅ Potencial créditos: ${creditos_minimos:,.0f} - ${creditos_maximos:,.0f} USD/año")
print(f"✅ Rentabilidad + ambiental: {((beneficio_con_creditos_max/configuracion_compromiso['beneficio_anual'])-1)*100:.1f}% mejora potencial")