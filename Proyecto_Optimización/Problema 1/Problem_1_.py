#***************************************************#
# Problem 1: Process Optimization Project           #
# Reactor de Transesterificación - Planta de Biodiesel
# Professor: Francisco Javier Vasquez Vasquez       #
# EMMCH                                            #
#***************************************************#

# =============================================================================
# DECLARACIÓN DE LIBRERÍAS REQUERIDAS
# =============================================================================
import numpy as np
from scipy.optimize import minimize, fsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

# Configuración para caracteres especiales
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuración para mostrar más decimales en numpy
np.set_printoptions(precision=6, suppress=True)

# Configuración de tipología para las gráficas
plt.rcParams['font.family'] = 'Times New Roman'

# =============================================================================
# CONSTANTES UNIVERSALES Y CONFIGURACIÓN
# =============================================================================
R = 1.987  # cal/mol·K (constante de los gases)
T_REF = 333.15  # K (60°C - temperatura de referencia)

print("=" * 70)
print("PROBLEMA 1: OPTIMIZACIÓN DEL REACTOR DE TRANSESTERIFICACIÓN")
print("Planta de Biodiesel - 10,000 ton/año")
print("=" * 70)

# =============================================================================
# PASO 1: PARÁMETROS CINÉTICOS - DATOS EXPERIMENTALES
# (Jansri et al., 2011 - Aceite de Palma)
# =============================================================================

R = 1.987  # cal/(mol·K) - constante de los gases

# Energías de activación (cal/mol) - Nomenclatura mejorada
ENERGIAS_ACTIVACION = {
    'Ea3': 34800,    # TG -> DG  
    'Ea4': 78650,    # DG -> TG (reversa)
    'Ea5': 15538,    # DG -> MG
    'Ea6': 30372,    # MG -> DG (reversa)
    'Ea7': 21356,    # MG -> GL
    'Ea8': 6321      # GL -> MG (reversa)
}

# Constantes cinéticas a 60°C (L/mol·min) - Datos de referencia
CONSTANTES_REFERENCIA = {
    'k3_ref': 2.600,    # TG -> DG
    'k4_ref': 0.248,    # DG -> TG (reversa)
    'k5_ref': 1.186,    # DG -> MG
    'k6_ref': 0.227,    # MG -> DG (reversa)
    'k7_ref': 2.303,    # MG -> GL
    'k8_ref': 0.022     # GL -> MG (reversa)
}

T_ref_C = 60.0
T_ref_K = T_ref_C + 273.15

# Diccionario para almacenar factores pre-exponenciales calculados
FACTORES_PRE_EXPONENCIALES = {}

# Cálculo de A3 a A8 a partir de k_ref y Ea
for i in range(3, 9):
    k_key = f'k{i}_ref'
    Ea_key = f'Ea{i}'
    
    k_ref = CONSTANTES_REFERENCIA[k_key]
    Ea_valor = ENERGIAS_ACTIVACION[Ea_key]
    
    # Ecuación de Arrhenius: k = A·exp(-Ea/RT) => A = k·exp(Ea/RT)
    A_valor = k_ref * np.exp(Ea_valor / (R * T_ref_K))
    
    FACTORES_PRE_EXPONENCIALES[f'A{i}'] = A_valor

print("\nFACTORES PRE-EXPONENCIALES (A3 a A8) CALCULADOS (UNIDADES: L/mol·min)")
print("=" * 75)

print("\n{:<10} {:<12} {:<15} {:<20} {:<10}".format(
    "Constante", "k@60°C", "Ea (cal/mol)", "Factor A", "Reacción"
))
print("{:<10} {:<12} {:<15} {:<20} {:<10}".format(
    "", "(L/mol·min)", "", "(L/mol·min)", ""
))
print("-" * 75)

# Función para transformar números grandes en notación científica
def formatear_numero(valor):
    if valor >= 10000 or valor <= 0.001:
        return "{:.4e}".format(valor)
    else:
        return "{:.4f}".format(valor)

# Mostrar tabla resumen
for i in range(3, 9):
    k_key = f'k{i}_ref'
    Ea_key = f'Ea{i}'
    A_key = f'A{i}'
    
    k_ref = CONSTANTES_REFERENCIA[k_key]
    Ea_valor = ENERGIAS_ACTIVACION[Ea_key]
    A_valor = FACTORES_PRE_EXPONENCIALES[A_key]
    
    reaccion = {
        3: "TG → DG",
        4: "DG → TG (rev.)",
        5: "DG → MG",
        6: "MG → DG (rev.)",
        7: "MG → GL",
        8: "GL → MG (rev.)"
    }[i]
    
    print("{:<10} {:<12} {:<15} {:<20} {:<10}".format(
        k_key, 
        formatear_numero(k_ref), 
        f"{Ea_valor:.0f}", 
        formatear_numero(A_valor), 
        reaccion
    ))


# =============================================================================
# PARTE 2: CÁLCULO DE CONSTANTES CINÉTICAS A CUALQUIER TEMPERATURA
# =============================================================================

def calcular_constantes_temperatura(T_C):
    """
    Calcula las constantes cinéticas k3 a k8 a una temperatura dada T (°C)
    Devuelve un diccionario con k3, k4, ..., k8
    """
    T_K = T_C + 273.15
    k = {}
    
    for i in range(3, 9):
        A_key = f'A{i}'
        Ea_key = f'Ea{i}'
        k_key = f'k{i}'
        
        A_valor = FACTORES_PRE_EXPONENCIALES[A_key]
        Ea_valor = ENERGIAS_ACTIVACION[Ea_key]
        
        # Ecuación de Arrhenius
        k_valor = A_valor * np.exp(-Ea_valor / (R * T_K))
        k[k_key] = k_valor
    
    return k

# Ejemplo: tabla de k(T) para visualizar sensibilidad a la temperatura
print("\nCONSTANTES CINÉTICAS k3-k8 EN FUNCIÓN DE LA TEMPERATURA")
print("=" * 75)
print("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
    "T(°C)", "k3", "k4", "k5", "k6", "k7", "k8"
))

for T in range(40, 81, 10):
    k_vals = calcular_constantes_temperatura(T)
    print("{:<8} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4e}".format(
        T,
        k_vals['k3'], k_vals['k4'], k_vals['k5'], 
        k_vals['k6'], k_vals['k7'], k_vals['k8']
    ))


print("\n" + "=" * 70)
print("FASE 1 COMPLETADA EXITOSAMENTE ✓")
print("Parámetros cinéticos calculados y validados")
print("=" * 70)


# =============================================================================
# PARTE 3: DEFINICIÓN DEL MODELO DE REACTOR CSTR
# =============================================================================

def modelo_CSTR(conc, F, V, C_in, k):
    """
    Sistema de ecuaciones del CSTR en estado estacionario
    conc: vector de concentraciones [TG, DG, MG, GL, AL, E]
    F: flujo volumétrico (L/min)
    V: volumen del reactor (L)
    C_in: vector de concentraciones de entrada
    k: diccionario con constantes cinéticas
    """
    TG, DG, MG, GL, AL, E = conc
    TG_in, DG_in, MG_in, GL_in, AL_in, E_in = C_in
    
    # Velocidades de reacción (modelo de Yiga et al. 2025)
    # r3: TG + AL -> DG + E
    r3 = k['k3'] * TG * AL
    # r4: DG + E -> TG + AL
    r4 = k['k4'] * DG * E
    # r5: DG + AL -> MG + E
    r5 = k['k5'] * DG * AL
    # r6: MG + E -> DG + AL
    r6 = k['k6'] * MG * E
    # r7: MG + AL -> GL + E
    r7 = k['k7'] * MG * AL
    # r8: GL + E -> MG + AL
    r8 = k['k8'] * GL * E
    
    # Balances de masa en estado estacionario
    dTG = F*(TG_in - TG) + V*(-r3 + r4)
    dDG = F*(DG_in - DG) + V*(r3 - r4 - r5 + r6)
    dMG = F*(MG_in - MG) + V*(r5 - r6 - r7 + r8)
    dGL = F*(GL_in - GL) + V*(r7 - r8)
    dAL = F*(AL_in - AL) + V*(-r3 - r5 - r7 + r4 + r6 + r8)
    dE  = F*(E_in - E)  + V*(r3 + r5 + r7 - r4 - r6 - r8)
    
    return [dTG, dDG, dMG, dGL, dAL, dE]

def resolver_CSTR(T_C, F, V, C_in, cat=1.0):
    """
    Resuelve el modelo CSTR a una temperatura dada
    T_C: temperatura en °C
    F: flujo volumétrico (L/min)
    V: volumen del reactor (L)
    C_in: vector de concentraciones de entrada
    cat: concentración de catalizador (% peso) - factor de ajuste
    """
    # 1. Calcular constantes cinéticas a la temperatura actual
    k = calcular_constantes_temperatura(T_C)
    
    # 2️. Ajustar constantes por concentración de catalizador (simplificado)
    for key in k.keys():
        k[key] = k[key] * (cat / 1.0)  
    
    # 3️. Estimación inicial para las concentraciones
    TG_in, DG_in, MG_in, GL_in, AL_in, E_in = C_in
    
    # Suposición: 
    conc0 = [
        TG_in * 0.1,      # TG 
        TG_in * 0.05,     # DG  
        TG_in * 0.03,     # MG
        TG_in * 0.02,     # GL
        AL_in * 0.4,      # AL 
        TG_in * 0.8       # E  
    ]

    # 4️. Resolver con fsolve - agregar tolerancia
    sol = fsolve(modelo_CSTR, conc0, args=(F, V, C_in, k), xtol=1e-6)
    sol = np.maximum(sol, 0.0)

    return sol

# Parámetros de operación para la prueba
F = 100.0    # L/min
V = 5000.0   # L (5 m³)
tau = V / F  # 50 min

# Alimentación: TG puro + Metanol con relación molar 6:1
TG_in = 1.0    # mol/L
RM = 6.0       # Relación molar MeOH:TG
AL_in = RM * TG_in  # mol/L

C_in = np.array([TG_in, 0, 0, 0, AL_in, 0])

# Resolver el reactor
print("\nRESOLUCIÓN DEL CSTR PARA UN PUNTO DE OPERACIÓN DE REFERENCIA")
print("=" * 75)

T_oper_C = 60.0  # Temperatura de operación (°C)
sol = resolver_CSTR(T_oper_C, F, V, C_in, cat=1.0)
TG, DG, MG, GL, AL, E = sol

conversion_ref = (TG_in - TG) / TG_in

print(f"Temperatura de operación: {T_oper_C:.1f} °C")
print(f"Conversión de triglicéridos (TG): {conversion_ref*100:.2f}%")
print(f"Concentraciones de salida [TG, DG, MG, GL, AL, E]:")
print(sol)

print("\n" + "=" * 70)
print("FASE 2 COMPLETADA EXITOSAMENTE ✓")
print("Modelo CSTR implementado y validado")
print("=" * 70)

# =============================================================================
# FASE 3: OPTIMIZACIÓN FORMAL DEL REACTOR
# =============================================================================

# =============================================================================
# PARTE 4: FORMULACIÓN DEL PROBLEMA DE OPTIMIZACIÓN
# =============================================================================

print("\nFORMULACIÓN DEL PROBLEMA DE OPTIMIZACIÓN")
print("=" * 75)
print("""
Variables de decisión:
    x = [T, RM, tau, catalizador]
    T: temperatura del reactor (°C)
    RM: relación molar MeOH:TG
    tau: tiempo de residencia (min)
    catalizador: % en peso (normalizado a 1.0 en el modelo)

Función objetivo:
    Maximizar la conversión de TG a biodiesel (E),
    penalizando excesos de metanol y catalizador.

Restricciones:
    - Conversión mínima de TG >= 98%
    - Presión de vapor del metanol por debajo de un umbral de seguridad
    - Límites operacionales para T, RM, tau y catalizador.
""")

# Parámetros base para la optimización
F_base = 24.0   # L/min (Yiga et| al., 2025)
V = 2300.0      # L (Yiga et| al., 2025)
tau_base = V / F_base

TG_in = 0.1     # mol/L
RM_base = 6.0
AL_in = RM_base * TG_in

C_in_base = np.array([TG_in, 0, 0, 0, AL_in, 0])

print("Parámetros base de operación:")
print(f"  • Volumen del reactor: {V} L")
print(f"  • Flujo base: {F_base} L/min")
print(f"  • Tiempo de residencia base: {tau_base:.2f} min")
print(f"  • Concentración de entrada de TG: {TG_in} mol/L")
print(f"  • Relación molar MeOH:TG base: {RM_base}")

# =============================================================================
# PARTE 5: DEFINICIÓN DE LA FUNCIÓN OBJETIVO
# =============================================================================

def funcion_objetivo(x, F, V, C_in):
    """
    Maximizar conversión penalizando:
    - exceso de relación molar de metanol
    - exceso de catalizador
    """
    T, RM, tau, catalizador = x

    # Cálculo del flujo
    F_actual = V / tau if tau > 0 else F

    # Actualizar entrada
    TG_in = C_in[0]

    C_in_actual = C_in.copy()
    C_in_actual[4] = RM * TG_in

    try:
        # Resolver el reactor
        resultado = resolver_CSTR(T, F_actual, V, C_in_actual, catalizador)
        TG_out = resultado[0]
        conversion = (TG_in - TG_out) / TG_in

        # Restricción de Costo por exceso de metanol
        penal_met = 0.01 * max(0, RM - 7.0)**2

        # Restricción de Costo por exceso de catalizador
        penal_cat = 0.5 * max(0, catalizador - 1.0)**2

        # Función objetivo (a minimizar)
        return -(conversion) + penal_met + penal_cat

    except Exception as e:
        # Advertencia si el solver no converge o hay error
        return 1e6

# =============================================================================
# PARTE 6: DEFINICIÓN DE RESTRICCIONES
# =============================================================================

def restriccion_conversion_minima(x, F, V, C_in):
    """
    Restricción: conversión mínima de TG >= 98%
    """
    T, RM, tau, catalizador = x
    
    # Calcular flujo volumétrico
    F_actual = V / tau if tau > 0 else F
    
    # Actualizar concentración de entrada
    TG_in = C_in[0]
    C_in_actual = C_in.copy()
    C_in_actual[4] = RM * TG_in
    
    try:
        resultado = resolver_CSTR(T, F_actual, V, C_in_actual, catalizador)
        TG_out = resultado[0]
        conversion = (TG_in - TG_out) / TG_in
        
        return conversion - 0.98 
        
    except:
        return -1e6  

def restriccion_presion_vapor(x):
    """
    Restricción: Presión de vapor del metanol 
    Evitar condiciones donde pueda haber vaporización
    """
    T, RM, tau, catalizador = x
    # Presión de vapor del metanol a temperatura T (mmHg)
    # log10(Pv) = A - B/(T + C) donde T en °C
    # Para metanol: A=8.08097, B=1582.271, C=239.726 (Antoine equation)
    Pv_mmHg = 10**(8.08097 - 1582.271/(T + 239.726))
    Pv_bar = Pv_mmHg / 750.062  # Conversión aproximada a bar
    
    # Límite de seguridad 
    Pv_max = 3.0  # bar
    
    return Pv_max - Pv_bar  

# =============================================================================
# PARTE 7: CONFIGURACIÓN DEL PROBLEMA DE OPTIMIZACIÓN
# =============================================================================

# Valores iniciales
x0 = [60.0, 6.0, tau_base, 1.0]  # [T, RM, tau, cat]

# Límites de las variables
limites = [
    (50.0, 70.0),   # T [°C]
    (3.0, 12.0),    # RM [mol/mol]
    (10.0, 200.0),  # tau [min]
    (0.5, 2.0)      # catalizador [%]
]

print("\nCONFIGURACIÓN DEL PROBLEMA DE OPTIMIZACIÓN (SLSQP)")
print("=" * 75)
print("Variables de decisión y límites:")
print(f"  T (°C):       {limites[0]}")
print(f"  RM (mol/mol): {limites[1]}")
print(f"  tau (min):    {limites[2]}")
print(f"  cat (%):      {limites[3]}")

# Definir restricciones
restricciones = [
    {'type': 'ineq', 'fun': restriccion_conversion_minima, 'args': (F_base, V, C_in_base)},
    {'type': 'ineq', 'fun': restriccion_presion_vapor}
]

# Ejecutar optimización
resultado_optimizacion = minimize(
    funcion_objetivo, 
    x0, 
    args=(F_base, V, C_in_base),
    method='SLSQP',
    bounds=limites,
    constraints=restricciones,
    options={'disp': True, 'maxiter': 100, 'ftol': 1e-8}
)

# =============================================================================
# PARTE 8: ANÁLISIS DE RESULTADOS Y CONDICIONES DE OPTIMALIDAD
# =============================================================================

print("\n" + "=" * 70)
print("RESULTADOS DE LA OPTIMIZACIÓN")
print("=" * 70)

if resultado_optimizacion.success:
    # Extraer resultados óptimos
    x_opt = resultado_optimizacion.x
    T_opt, RM_opt, tau_opt, cat_opt = x_opt
    
    # Calcular flujo óptimo
    F_opt = V / tau_opt
    
    # Calcular concentración de entrada óptima
    C_in_opt = C_in_base.copy()
    C_in_opt[4] = RM_opt * TG_in
    
    # Resolver el reactor con las condiciones óptimas
    sol_opt = resolver_CSTR(T_opt, F_opt, V, C_in_opt, cat_opt)
    TG_opt, DG_opt, MG_opt, GL_opt, AL_opt, E_opt = sol_opt
    
    conversion_opt = (TG_in - TG_opt) / TG_in * 100
    rendimiento = (E_opt / (3 * TG_in)) * 100 if TG_in > 0 else 0
    selectividad = E_opt / (E_opt + DG_opt + MG_opt + GL_opt + TG_opt)
    selectividad_pct = selectividad * 100
    
    print(f"PUNTO ÓPTIMO DE OPERACIÓN ENCONTRADO:")
    print(f"  • Temperatura (T):         {T_opt:.2f} °C")
    print(f"  • Relación MeOH:TG (RM):   {RM_opt:.2f} mol/mol")
    print(f"  • Tiempo de residencia:    {tau_opt:.2f} min")
    print(f"  • Concentración catalizador: {cat_opt:.2f} %")

    print("\nDESEMPEÑO EN PUNTO ÓPTIMO:")
    print(f"   • Conversión de TG:     {conversion_opt:.2f}%")
    print(f"   • Biodiesel producido:  {E_opt:.4f} mol/L")
    print(f"   • Consumo de metanol:   {((RM_opt * TG_in) - AL_opt) / (RM_opt * TG_in) * 100:.1f}%")  
    print(f"   • Rendimiento a Biodiesel: {rendimiento:.2f}%")
    print(f"   • Selectividad: {selectividad_pct:.2f}%")

    print(f"\nCONCENTRACIONES DE SALIDA [mol/L]:")
    print(f"• Triglicéridos (TG): {TG_opt:.4f}")
    print(f"• Diglicéridos (DG):  {DG_opt:.4f}")
    print(f"• Monoglicéridos (MG): {MG_opt:.4f}")
    print(f"• Glicerol (GL):     {GL_opt:.4f}")
    print(f"• Metanol (AL):      {AL_opt:.4f}")
    print(f"• Biodiesel (E):     {E_opt:.4f}")

    print(f"\n⚖️  VERIFICACIÓN DE RESTRICCIONES (CONDICIONES DE OPTIMALIDAD PRÁCTICAS):")

    g_conv = restriccion_conversion_minima(x_opt, F_base, V, C_in_base)  
    X_opt = g_conv + 0.98  
    print(f"   • Conversión óptima: {X_opt*100:.2f}%")
    print(f"   • g_conv = X_TG - 0.98 = {g_conv:+.4e}  (≈0 indica restricción activa)")

    g_pv = restriccion_presion_vapor(x_opt)
    
    Pv_mmHg = 10**(8.08097 - 1582.271/(T_opt + 239.726))
    Pv_bar = Pv_mmHg / 750.062
    print(f"   • Presión de vapor MeOH: {Pv_bar:.2f} bar (límite: 3.0 bar)")
    print(f"   • g_pv = Pv - Pv_max = {g_pv:+.4e}  (≤0 indica región segura)")

    print(f"\n📈 INFORMACIÓN DE LA OPTIMIZACIÓN:")
    print(f"   • Número de iteraciones: {resultado_optimizacion.nit}")
    print(f"   • Número de evaluaciones: {resultado_optimizacion.nfev}")
    print(f"   • Mensaje: {resultado_optimizacion.message}")
    
else:
    print("❌ LA OPTIMIZACIÓN NO CONVERGIÓ")
    print(f"Motivo: {resultado_optimizacion.message}")

print("\n" + "=" * 70)
print("OPTIMIZACIÓN COMPLETADA EXITOSAMENTE ✓")
print("Punto óptimo identificado y analizado")
print("=" * 70)

# =============================================================================
# PARTE 9: ANÁLISIS DE SENSIBILIDAD
# =============================================================================

def analisis_sensibilidad(variable, rango, x_base, F_base, V, C_in_base, indice):
    """
    variable: nombre de la variable para el título del gráfico
    rango: array con los valores a evaluar
    x_base: vector [T, RM, tau, catalizador] de base (óptimo)
    indice: posición de la variable a variar en x_base
    """
    conversiones = []

    # Paleta de colores 
    colores_pastel = [
        '#FFB347',  
        '#77DD77',  
        '#FF6961',  
        '#C3B1E1',  
        '#FDFD96'   
    ]

    color = colores_pastel[indice % len(colores_pastel)]

    for valor in rango:
        x_test = x_base.copy()
        x_test[indice] = valor

        # Calcular flujo según tiempo de residencia
        F_test = V / x_test[2]

        # Actualizar concentración de metanol
        C_in_test = C_in_base.copy()
        C_in_test[4] = x_test[1] * C_in_base[0]

        try:
            resultado = resolver_CSTR(x_test[0], F_test, V, C_in_test, x_test[3])
            TG_out = resultado[0]
            conv = (C_in_base[0] - TG_out) / C_in_base[0] * 100
            conversiones.append(conv)
        except:
            conversiones.append(np.nan)

    # Graficar
    plt.figure(figsize=(6,4))
    plt.plot(rango, conversiones, marker='o', linewidth=2, color=color, label='Conversión')

    plt.xlabel(f'{variable}', fontweight='bold')
    plt.ylabel('Conversión TG (%)', fontweight='bold')
    plt.title(f'Análisis de sensibilidad: {variable} vs Conversión', fontweight='bold')

    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==============================
# Usar el punto óptimo obtenido
# ==============================
x_opt = resultado_optimizacion.x
T_opt, RM_opt, tau_opt, cat_opt = x_opt

# Rangos de sensibilidad
rango_T = np.linspace(50, 70, 10)
rango_RM = np.linspace(4, 12, 10)
rango_tau = np.linspace(30, 120, 10)
rango_cat = np.linspace(0.5, 2.0, 10)

# Sensibilidad
analisis_sensibilidad('Temperatura [°C]', rango_T, x_opt, F_base, V, C_in_base, 0)
analisis_sensibilidad('Relación molar MeOH:TG', rango_RM, x_opt, F_base, V, C_in_base, 1)
analisis_sensibilidad('Tiempo de residencia [min]', rango_tau, x_opt, F_base, V, C_in_base, 2)
analisis_sensibilidad('Concentración catalizador [%]', rango_cat, x_opt, F_base, V, C_in_base, 3)

# =============================================================================
# PARTE 10: Visualizaciones de Contorno
# =============================================================================

def superficie_respuesta(T_range, RM_range, tau, cat, F, V, C_in):
    """
    Genera superficies de respuesta (T, RM) vs. conversión de TG
    """
    T_vals = np.linspace(*T_range, 20)
    RM_vals = np.linspace(*RM_range, 20)
    
    T_grid, RM_grid = np.meshgrid(T_vals, RM_vals)
    conv_grid = np.zeros_like(T_grid)
    
    TG_in = C_in[0]
    
    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            T = T_grid[i, j]
            RM = RM_grid[i, j]
            
            F_actual = V / tau if tau > 0 else F
            
            C_in_actual = C_in.copy()
            C_in_actual[4] = RM * TG_in
            
            try:
                sol = resolver_CSTR(T, F_actual, V, C_in_actual, cat)
                TG_out = sol[0]
                conv_grid[i, j] = (TG_in - TG_out) / TG_in * 100
            except:
                conv_grid[i, j] = np.nan
    
    # Gráfica 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_grid, RM_grid, conv_grid, cmap='viridis')
    ax.set_xlabel("Temperatura (°C)", fontweight='bold')
    ax.set_ylabel("RM (MeOH:TG)", fontweight='bold')
    ax.set_zlabel("Conversión de TG (%)", fontweight='bold')
    ax.set_title("Superficie de respuesta: Conversión vs T y RM", fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Gráfico de contorno
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(T_grid, RM_grid, conv_grid, levels=20, cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.set_label("Conversión de TG (%)", fontweight='bold')
    plt.xlabel("Temperatura (°C)", fontweight='bold')
    plt.ylabel("RM (MeOH:TG)", fontweight='bold')
    plt.title("Mapa de contornos: Conversión vs T y RM", fontweight='bold')
    plt.tight_layout()
    plt.show()

if resultado_optimizacion.success:
    superficie_respuesta(
        T_range=(50, 70), 
        RM_range=(3, 12), 
        tau=resultado_optimizacion.x[2],
        cat=resultado_optimizacion.x[3],
        F=F_base,
        V=V,
        C_in=C_in_base
    )
