#***************************************************#
# Problem 3: Process Optimization Project           #
# Estructura de Producción - Planta de Biodiesel
# Professor: Francisco Javier Vasquez Vasquez       #
# EMMCH                                            #
#***************************************************#

import pulp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import warnings
import sys

# Configuración para caracteres especiales
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Suprimir warnings específicos de pandas
warnings.filterwarnings('ignore', category=FutureWarning)

capacidad_max = 1200
inventario_max = 300

def analisis_detallado_biodiesel():
    # Datos del problema
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    productos = ['B100', 'B20', 'B10']
    
    # Demanda por producto y mes (ton)
    demanda_B100 = [700, 700, 700, 900, 900, 900, 1000, 1000, 1000, 800, 800, 800]
    demanda_B20 = [300, 300, 300,  120, 120, 120, 140, 140, 140, 150, 150, 150]
    demanda_B10 = [200, 200, 200, 80, 80, 80, 50, 50, 50, 100, 100, 100,]
    
    # Precios de venta por producto (USD/ton)
    precios = {
        'B100': [1250, 1250, 1250, 1300, 1300, 1300, 1350, 1350, 1350, 1300, 1300, 1300],
        'B20': [1100, 1100, 1100, 1150, 1150, 1150, 1200, 1200, 1200, 1150, 1150, 1150],
        'B10': [1050, 1050, 1050, 1100, 1100, 1100, 1150, 1150, 1150, 1100, 1100, 1100]
    }
    
    precio_aceite = [850, 850, 850, 900, 900, 900, 950, 950, 950, 900, 900, 900]
    
    costo_almacenamiento = 5
    costo_diesel = 700
    costo_mezcla = 10
    
    # Costos fijos
    costo_metanol = 0.1 * 450
    costo_catalizador = 0.01 * 600
    
    # Crear problema de optimización
    prob = pulp.LpProblem("Biodiesel_MultiProducto_Detallado", pulp.LpMaximize)
    
    # Variables de decisión
    x = {}
    I = {}
    
    for prod in productos:
        for i in range(12):
            x[prod, i] = pulp.LpVariable(f"Prod_{prod}_{i}", lowBound=0)
    
    for i in range(12):
        I[i] = pulp.LpVariable(f"Inv_{i}", lowBound=0, upBound=inventario_max)
    
    y = pulp.LpVariable.dicts("Biodiesel_Mezclas", range(12), lowBound=0)
    
    # FUNCIÓN OBJETIVO
    ingresos = 0
    costos = 0
    
    for i in range(12):
        # Ingresos
        ingresos += precios['B100'][i] * x['B100', i]
        ingresos += precios['B20'][i] * x['B20', i]
        ingresos += precios['B10'][i] * x['B10', i]
        
        # Costos
        biodiesel_total = x['B100', i] + y[i]
        costos += (0.8 * precio_aceite[i] + costo_metanol + costo_catalizador) * biodiesel_total
        costos += costo_diesel * (0.8 * x['B20', i] + 0.9 * x['B10', i])
        costos += costo_mezcla * (x['B20', i] + x['B10', i])
        costos += costo_almacenamiento * I[i]
    
    prob += ingresos - costos
    
    # RESTRICCIONES
    # Balance de inventario
    prob += I[0] == x['B100', 0] + y[0] - demanda_B100[0]
    for i in range(1, 12):
        prob += I[i] == I[i-1] + x['B100', i] + y[i] - demanda_B100[i]
    
    # Biodiesel para mezclas
    for i in range(12):
        prob += y[i] == 0.2 * x['B20', i] + 0.1 * x['B10', i]
    
    # Capacidad
    for i in range(12):
        prob += x['B100', i] + x['B20', i] + x['B10', i] <= capacidad_max
    
    # Demanda
    for i in range(12):
        prob += x['B100', i] <= demanda_B100[i]
        prob += x['B20', i] <= demanda_B20[i]
        prob += x['B10', i] <= demanda_B10[i]
    
    # Resolver
    prob.solve()
    
    print("=" * 70)
    print("ANÁLISIS DETALLADO - PLAN DE PRODUCCIÓN ÓPTIMO")
    print("=" * 70)
    
    # Resultados detallados
    resultados = []
    for i in range(12):
        for prod in productos:
            prod_value = x[prod, i].varValue
            demanda_val = locals()[f"demanda_{prod}"][i]
            precio_val = precios[prod][i]
            
            # Calcular margen por producto
            if prod == 'B100':
                costo_prod = (0.8 * precio_aceite[i] + costo_metanol + costo_catalizador)
            elif prod == 'B20':
                costo_prod = (0.2 * (0.8 * precio_aceite[i] + costo_metanol + costo_catalizador) + 
                            0.8 * costo_diesel + costo_mezcla)
            else:  # B10
                costo_prod = (0.1 * (0.8 * precio_aceite[i] + costo_metanol + costo_catalizador) + 
                            0.9 * costo_diesel + costo_mezcla)
            
            margen = precio_val - costo_prod
            contribucion = margen * prod_value
            
            resultados.append({
                'Mes': meses[i],
                'Producto': prod,
                'Produccion': prod_value,
                'Demanda': demanda_val,
                'Precio_Venta': precio_val,
                'Costo_Produccion': costo_prod,
                'Margen_Unitario': margen,
                'Contribucion_Total': contribucion
            })
    
    df_detallado = pd.DataFrame(resultados)
    
    # Análisis agregado
    print(f"\n💰 BENEFICIO NETO ANUAL: ${pulp.value(prob.objective):,.2f}")
    
    # Contribución por producto
    contribucion_productos = df_detallado.groupby('Producto')['Contribucion_Total'].sum()
    print(f"\n📊 CONTRIBUCIÓN POR PRODUCTO:")
    for producto, contrib in contribucion_productos.items():
        print(f"   {producto}: ${contrib:,.2f}")
    
    # Utilización de capacidad
    produccion_total = df_detallado.groupby('Mes')['Produccion'].sum()
    utilizacion_promedio = (produccion_total.mean() / capacidad_max) * 100
    print(f"\n🏭 UTILIZACIÓN DE CAPACIDAD: {utilizacion_promedio:.1f}%")
    
    return df_detallado, pulp.value(prob.objective), meses, productos, demanda_B100, demanda_B20, demanda_B10

# Ejecución análisis detallado
df_detallado, beneficio, meses, productos, demanda_B100, demanda_B20, demanda_B10 = analisis_detallado_biodiesel()

# Definición orden correcto de meses
orden_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
df_detallado['Mes'] = pd.Categorical(df_detallado['Mes'], categories=orden_meses, ordered=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    produccion_mensual = df_detallado.groupby(['Mes', 'Producto'], observed=True)['Produccion'].sum().unstack()
    margenes_mensual = df_detallado.groupby(['Mes', 'Producto'], observed=True)['Margen_Unitario'].mean().unstack()
    capacidad_utilizada = df_detallado.groupby('Mes', observed=True)['Produccion'].sum().reindex(orden_meses)

capacidad_pct = (capacidad_utilizada / 1200) * 100


# Configuración de la salida de Gráficas

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'

colores_pastel = {
    'B100': '#FFB6C1',  
    'B20': '#87CEFA',   
    'B10': '#98FB98'    
}

fig = plt.figure(figsize=(16, 12), constrained_layout=True)

# Crear grid specification
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])  # Producción mensual
ax2 = fig.add_subplot(gs[0, 1])  # Margen unitario
ax3 = fig.add_subplot(gs[1, 0])  # Contribución porcentual
ax4 = fig.add_subplot(gs[1, 1])  # Utilización de capacidad

# ==================== GRÁFICO 1: Producción mensual por producto ====================

x = np.arange(len(orden_meses))
ancho = 0.25

for idx, producto in enumerate(productos):
    valores = [produccion_mensual.loc[mes, producto] if mes in produccion_mensual.index else 0 
               for mes in orden_meses]
    ax1.bar(x + idx * ancho, valores, width=ancho, label=producto, alpha=0.8,
            color=colores_pastel[producto], edgecolor='black', linewidth=0.5)

ax1.set_title('Producción Mensual por Tipo de Biodiesel', fontsize=11, pad=10)
ax1.set_ylabel('Toneladas', fontsize=10)
ax1.set_xlabel('Mes', fontsize=10)
ax1.set_xticks(x + ancho)
ax1.set_xticklabels(orden_meses, rotation=45, fontsize=9)
ax1.legend(title='Producto', title_fontsize=9, fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, 1200)

# ==================== GRÁFICO 2: Margen unitario por producto ====================

marcadores = ['o', 's', '^']  # Diferentes marcadores para cada producto
for idx, producto in enumerate(productos):
    valores = [margenes_mensual.loc[mes, producto] if mes in margenes_mensual.index else 0 
               for mes in orden_meses]
    ax2.plot(orden_meses, valores, marker=marcadores[idx], linewidth=2, 
             markersize=5, label=producto, color=colores_pastel[producto],
             markeredgecolor='black', markeredgewidth=0.5)

ax2.set_title('Evolución del Margen Unitario', fontsize=11, pad=10)
ax2.set_ylabel('USD por Tonelada', fontsize=10)
ax2.set_xlabel('Mes', fontsize=10)
ax2.legend(title='Producto', title_fontsize=9, fontsize=8)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=45, labelsize=9)
ax2.tick_params(axis='y', labelsize=9)

# ==================== GRÁFICO 3: Contribución porcentual por producto ====================

contribucion_productos = df_detallado.groupby('Producto')['Contribucion_Total'].sum()
colors_pastel_pie = [colores_pastel[prod] for prod in contribucion_productos.index]

wedges, texts, autotexts = ax3.pie(contribucion_productos.values, 
                                  labels=contribucion_productos.index, 
                                  autopct='%1.1f%%', 
                                  colors=colors_pastel_pie, 
                                  startangle=90,
                                  textprops={'fontsize': 9, 'fontweight': 'bold'})

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

ax3.set_title('Distribución de Contribución al Beneficio', fontsize=11, pad=10)

# ==================== GRÁFICO 4: Utilización de capacidad mensual ====================

bars = ax4.bar(orden_meses, capacidad_pct, 
               color='#87CEEB', alpha=0.8, edgecolor='navy', linewidth=1)
ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Capacidad Máxima')
ax4.axhline(y=90, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Objetivo (90%)')

for bar, valor in zip(bars, capacidad_pct):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{valor:.0f}%', ha='center', va='bottom', 
             fontsize=8, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))

ax4.set_title('Utilización de Capacidad de Planta', fontsize=11, pad=10)
ax4.set_ylabel('Porcentaje de Capacidad (%)', fontsize=10)
ax4.set_xlabel('Mes', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim(0, 115)
ax4.tick_params(axis='x', rotation=45, labelsize=9)
ax4.tick_params(axis='y', labelsize=9)
fig.suptitle('ANÁLISIS DE PRODUCCIÓN - PLANTA DE BIODIESEL', 
             fontsize=14, fontweight='bold')

plt.show()

# Calcular demanda total por mes
demanda_total = {}
for i, mes in enumerate(orden_meses):
    demanda_total[mes] = demanda_B100[i] + demanda_B20[i] + demanda_B10[i]

# Convertir a series ordenadas
demanda_series = pd.Series(demanda_total).reindex(orden_meses)
produccion_series = capacidad_utilizada.reindex(orden_meses)
deficit = demanda_series - produccion_series

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

# Gráfico 1: Comparación directa

x_pos = np.arange(len(orden_meses))
ancho = 0.35

barras_demanda = ax1.bar(x_pos - ancho/2, demanda_series.values, ancho, 
                        label='Demanda Total', alpha=0.8, color='#FFA07A',
                        edgecolor='black', linewidth=0.5)

barras_produccion = ax1.bar(x_pos + ancho/2, produccion_series.values, ancho, 
                           label='Producción Real', alpha=0.8, color='#20B2AA',
                           edgecolor='black', linewidth=0.5)

ax1.axhline(y=1200, color='red', linestyle='--', linewidth=2, label='Capacidad Máxima')

ax1.set_title('Demanda vs Producción Mensual', fontsize=11, fontweight='bold', pad=10)
ax1.set_ylabel('Toneladas', fontsize=10, fontweight='bold')
ax1.set_xlabel('Mes', fontsize=10, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(orden_meses, rotation=45, fontsize=9)
ax1.legend(fontsize=8)

# Gráfico 2: Déficit mensual mejorado

deficit_colors = ['#FF6B6B' if val > 0 else '#51CF66' for val in deficit]

bars_deficit = ax2.bar(orden_meses, deficit.values, color=deficit_colors, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_title('Déficit Mensual (Demanda No Satisfecha)', fontsize=11, fontweight='bold', pad=10)
ax2.set_ylabel('Toneladas de Déficit', fontsize=10, fontweight='bold')
ax2.set_xlabel('Mes', fontsize=10, fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=9)

for i, (mes, valor, bar) in enumerate(zip(orden_meses, deficit.values, bars_deficit)):
    if valor != 0:
        ax2.text(bar.get_x() + bar.get_width()/2, 
                valor + (-10 if valor >= 0 else -15), 
                f'{valor:.0f}', 
                ha='center', va='bottom' if valor >= 0 else 'top', 
                fontweight='bold', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

deficit_total = sum([max(0, d) for d in deficit])
ax2.text(0.02, 0.98, f'Déficit Anual: {deficit_total:,.0f} ton', 
         transform=ax2.transAxes, fontsize=9, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
         verticalalignment='top')

plt.show()

# =============================================================================
# ANÁLISIS DE SENSIBILIDAD Y RECOMENDACIONES
# =============================================================================

print("\n" + "=" * 70)
print("RECOMENDACIONES ESTRATÉGICAS")
print("=" * 70)

# Identificar el producto más rentable
margen_promedio = df_detallado.groupby('Producto')['Margen_Unitario'].mean()
producto_mas_rentable = margen_promedio.idxmax()

print(f"\n🎯 PRODUCTO MÁS RENTABLE: {producto_mas_rentable}")
print(f"   Margen promedio: ${margen_promedio[producto_mas_rentable]:.2f}/ton")

# Análisis del déficit
deficit_anual = sum(deficit)
demanda_anual = sum(demanda_series)
produccion_anual = sum(produccion_series)

print(f"\n📊 ANÁLISIS DE DÉFICIT ANUAL:")
print(f"   Demanda anual total: {demanda_anual:,} ton")
print(f"   Producción anual total: {produccion_anual:,} ton")
print(f"   Déficit anual total: {deficit_anual:,} ton")
print(f"   Tasa de satisfacción: {produccion_anual/demanda_anual*100:.1f}%")

# Análisis de capacidad
print(f"\n🏭 ANÁLISIS DE CAPACIDAD:")
print(f"   Producción promedio: {produccion_series.mean():.0f} ton/mes")
print(f"   Utilización promedio: {capacidad_pct.mean():.1f}%")
print(f"   Variación estacional: ±{(produccion_series.max() - produccion_series.min())/2:.0f} ton")

# Recomendaciones basadas en el análisis
if deficit_anual > 0:
    print(f"\n⚠️  OPORTUNIDAD DE EXPANSIÓN DETECTADA")
    print(f"   Demanda insatisfecha: {deficit_anual:,} ton/año")
    print(f"   Oportunidad de ingreso adicional: ${deficit_anual * margen_promedio.mean():,.2f}")

if capacidad_pct.min() < 80:
    print(f"\n📉 CAPACIDAD OCIOSA DETECTADA")
    mes_min = capacidad_pct.idxmin()
    print(f"   Mes con menor producción: {mes_min} ({capacidad_pct[mes_min]:.0f}% de capacidad)")
    print("   Recomendación: Buscar nuevos mercados o clientes para periodos de baja demanda")

print(f"\n✅ CONCLUSIÓN: La estrategia multi-producto genera {beneficio/2947500-1:.1%} más beneficio que solo producir B100")

# =============================================================================
# RESUMEN DE RESULTADOS
# =============================================================================

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"💰 Beneficio Neto Anual: ${beneficio:,.2f}")
print(f"📈 Tasa de Retorno Multi-Producto: +{(beneficio/2947500-1):.1%}")
print(f"🏭 Utilización Promedio: {capacidad_pct.mean():.1f}%")
print(f"🎯 Producto Estrella: {producto_mas_rentable}")
print(f"📊 Satisfacción de Demanda: {produccion_anual/demanda_anual*100:.1f}%")
print("=" * 70)


plan_produccion = df_detallado.pivot_table(
    index='Mes', 
    columns='Producto', 
    values='Produccion', 
    aggfunc='sum'
).reindex(orden_meses)


plan_produccion['Total Producción'] = plan_produccion.sum(axis=1)
plan_produccion['Capacidad Utilizada (%)'] = (plan_produccion['Total Producción'] / capacidad_max * 100).round(1)

produccion_promedio_mensual = plan_produccion['Total Producción'].mean()
max_produccion = plan_produccion['Total Producción'].max()
min_produccion = plan_produccion['Total Producción'].min()
mes_max_produccion = plan_produccion['Total Producción'].idxmax()
mes_min_produccion = plan_produccion['Total Producción'].idxmin()
variabilidad_produccion = plan_produccion['Total Producción'].std() / produccion_promedio_mensual * 100
total_produccion_2025 = plan_produccion['Total Producción'].sum()

print("\n" + "=" * 80)
print("3. PLAN DE PRODUCCIÓN ÓPTIMO 2025 - DIAGRAMA DE GANTT")
print("=" * 80)

# =============================================================================
# CONFIGURACIÓN PARA AÑO 2025
# =============================================================================

# Calendario 2025
dias_por_mes_2025 = {
    'Ene': 31, 'Feb': 28, 'Mar': 31, 'Abr': 30, 'May': 31, 'Jun': 30,
    'Jul': 31, 'Ago': 31, 'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dic': 31
}

hitos_2025 = {
    'Inicio Año': 'Ene',
    'Cierre Q1': 'Mar', 
    'Inicio Q2': 'Abr',
    'Cierre Q2': 'Jun',
    'Inicio Q3': 'Jul',
    'Cierre Q3': 'Sep',
    'Inicio Q4': 'Oct',
    'Cierre Anual': 'Dic',
    'Mantenimiento': 'Jul',
    'Auditoría': 'Nov'
}

# Capacidades
CAPACIDAD_MENSUAL = 1200  # ton/mes
CAPACIDAD_ANUAL = 10000   # ton/año


print("📊 Generando Diagrama de Gantt 2025...")

fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)

# Generar códigos de lote únicos
def generar_lote(producto, mes, consecutivo):
    return f"LOTE-{producto}-{mes}-2025-{consecutivo:03d}"

# Datos con códigos de lote
gantt_2025 = []
consecutivo_global = 1

for i, mes in enumerate(orden_meses):
    for producto in productos:
        produccion = plan_produccion.loc[mes, producto]
        if produccion > 0:
            lote = generar_lote(producto, mes, consecutivo_global)
            consecutivo_global += 1
            
            gantt_2025.append({
                'Tarea': f'{producto}',
                'Mes': mes,
                'Posicion': i,
                'Produccion': produccion,
                'Color': colores_pastel[producto],
                'Lote': lote,
                'Consecutivo': consecutivo_global - 1
            })

df_gantt = pd.DataFrame(gantt_2025)

posiciones_y = {}
contador_y = 0
for producto in productos:
    for mes in orden_meses:
        key = f"{producto}-{mes}"
        posiciones_y[key] = contador_y
        contador_y += 1

for idx, row in df_gantt.iterrows():
    key = f"{row['Tarea']}-{row['Mes']}"
    pos_y = posiciones_y[key]
    
    ax.barh(pos_y, 0.8, left=row['Posicion'], 
            color=row['Color'], alpha=0.8, edgecolor='black', linewidth=1,
            height=0.6)
    
    ax.text(row['Posicion'] + 0.4, pos_y, 
            f"{row['Produccion']} ton", 
            ha='center', va='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    ax.text(row['Posicion'] + 0.4, pos_y - 0.6, 
            row['Lote'], 
            ha='center', va='top', fontsize=7, fontweight='bold', color='darkblue',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.8))

ax.set_xticks(np.arange(0, 12, 1))
ax.set_xticklabels(orden_meses, fontsize=10, fontweight='bold')
ax.set_xlabel('MESES 2025', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlim(-0.5, 11.5)

posiciones_y_ticks = [posiciones_y[f"{p}-{m}"] for p in productos for m in orden_meses]
etiquetas_y_ticks = [m for _ in productos for m in orden_meses]

ax.set_yticks(posiciones_y_ticks)
ax.set_yticklabels(etiquetas_y_ticks, fontsize=9)
ax.set_ylabel('PRODUCTO \\ MES', fontsize=12, fontweight='bold', labelpad=10)

ax.set_ylim(-1.5, len(posiciones_y_ticks))

hitos_simplificados = {
    'Inicio': 0,
    'Q1': 3, 
    'Q2': 6,
    'Mantenimiento': 7,
    'Q3': 9,
    'Auditoría': 10,
    'Fin Año': 11
}

for hito, pos in hitos_simplificados.items():
    color = 'green' if hito == 'Inicio' else 'red' if hito == 'Mantenimiento' else 'blue' if hito == 'Auditoría' else 'purple'
    
    ax.axvline(x=pos, color=color, linestyle='--', linewidth=2, alpha=0.7)
    
    ax.text(pos, len(posiciones_y_ticks) + 0.1, hito, 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
ax.set_title('DIAGRAMA DE GANTT - PLAN DE PRODUCCIÓN 2025', 
             fontsize=14, fontweight='bold', pad=20)

legend_elements = [
    Patch(facecolor=colores_pastel['B100'], label='B100', alpha=0.8),
    Patch(facecolor=colores_pastel['B20'], label='B20', alpha=0.8),
    Patch(facecolor=colores_pastel['B10'], label='B10', alpha=0.8),
    Patch(facecolor='lightyellow', edgecolor='darkblue', label='Código de Lote', alpha=0.8)
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
          bbox_to_anchor=(1.0, 0.86))


utilizacion_anual = (total_produccion_2025 / CAPACIDAD_ANUAL) * 100

resumen_texto = (
    f"RESUMEN 2025:\n"
    f"• Producción Total: {total_produccion_2025:,.0f} ton\n"
    f"• Capacidad Utilizada: {utilizacion_anual:.1f}%\n"
    f"• Lotes Programados: {consecutivo_global - 1}\n"
    f"• Mes Pico: {mes_max_produccion}\n"
    f"• Producto Principal: {producto_mas_rentable}"
)

ax.text(0.06, 0.98, resumen_texto, transform=ax.transAxes, fontsize=10, 
        fontweight='bold', verticalalignment='top', linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9))

ax.grid(True, alpha=0.2, linestyle='-', axis='x')
ax.grid(True, alpha=0.1, linestyle='-', axis='y')

# Líneas de separación entre trimestres
for trimestre in [3, 6, 9]:
    ax.axvline(x=trimestre, color='gray', linestyle=':', alpha=0.3, linewidth=1)

plt.show()
# =============================================================================
# TABLA DETALLADA DE PRODUCCIÓN 2025
# =============================================================================

print("\n📅 PROGRAMA DE PRODUCCIÓN 2025")
print("=" * 100)

# Crear tabla detallada 2025
produccion_2025 = []
for mes in orden_meses:
    mes_data = plan_produccion.loc[mes]
    capacidad_util = mes_data['Capacidad Utilizada (%)']
    deficit = mes_data['Déficit'] if 'Déficit' in plan_produccion.columns else 0
    
    produccion_2025.append({
        'Mes': mes,
        'B100 (ton)': mes_data['B100'],
        'B20 (ton)': mes_data['B20'],
        'B10 (ton)': mes_data['B10'],
        'Total (ton)': mes_data['Total Producción'],
        'Capacidad (%)': capacidad_util,
        'Demanda (ton)': mes_data['Demanda Total'] if 'Demanda Total' in plan_produccion.columns else 0,
        'Déficit (ton)': deficit if deficit > 0 else 0
    })

df_produccion_2025 = pd.DataFrame(produccion_2025)

print(f"{'Mes':<6} {'B100':<8} {'B20':<8} {'B10':<8} {'Total':<8} {'Capacidad':<10} {'Demanda':<8} {'Déficit':<8}")
print("-" * 100)
for _, row in df_produccion_2025.iterrows():
    deficit_str = f"{row['Déficit (ton)']:.0f}" if row['Déficit (ton)'] > 0 else "0"
    demanda_str = f"{row['Demanda (ton)']:.0f}" if 'Demanda (ton)' in row else "N/A"
    print(f"{row['Mes']:<6} {row['B100 (ton)']:<8.0f} {row['B20 (ton)']:<8.0f} "
          f"{row['B10 (ton)']:<8.0f} {row['Total (ton)']:<8.0f} {row['Capacidad (%)']:<9.1f}% "
          f"{demanda_str:<8} {deficit_str:<8}")

# =============================================================================
# ANÁLISIS DE CAPACIDAD 2025
# =============================================================================

print("\n" + "=" * 80)
print("ANÁLISIS DE CAPACIDAD Y PRODUCCIÓN 2025")
print("=" * 80)

# Calcular métricas específicas para 2025
meses_sobre_90pct = plan_produccion[plan_produccion['Capacidad Utilizada (%)'] > 90].shape[0]
meses_sobre_100pct = plan_produccion[plan_produccion['Total Producción'] > CAPACIDAD_MENSUAL].shape[0]
capacidad_ociosa_anual = CAPACIDAD_ANUAL - total_produccion_2025

print(f"\n📊 MÉTRICAS DE CAPACIDAD 2025:")
print(f"   • Producción anual total: {total_produccion_2025:,.0f} ton")
print(f"   • Utilización anual: {utilizacion_anual:.1f}%")
print(f"   • Capacidad ociosa anual: {capacidad_ociosa_anual:,.0f} ton")
print(f"   • Meses >90% capacidad: {meses_sobre_90pct}")
print(f"   • Meses >100% capacidad: {meses_sobre_100pct}")

print(f"\n🎯 RENDIMIENTO OPERATIVO:")
print(f"   • Producción promedio: {produccion_promedio_mensual:.0f} ton/mes")
print(f"   • Variabilidad: {variabilidad_produccion:.1f}%")
print(f"   • Mes más productivo: {mes_max_produccion} ({max_produccion:.0f} ton)")
print(f"   • Mes menos productivo: {mes_min_produccion} ({min_produccion:.0f} ton)")

# =============================================================================
# CUMPLIMIENTO DE OBJETIVOS 2025
# =============================================================================

print("\n" + "=" * 80)
print("CUMPLIMIENTO DE OBJETIVOS 2025")
print("=" * 80)

# Verificar cumplimiento de capacidad anual
cumplimiento_capacidad = (total_produccion_2025 / CAPACIDAD_ANUAL) * 100
estado_cumplimiento = "✅ SUPERADO" if total_produccion_2025 >= CAPACIDAD_ANUAL else "⚠️  POR DEBAJO"

print(f"\n🎯 CAPACIDAD ANUAL ({CAPACIDAD_ANUAL:,} ton):")
print(f"   • Producción programada: {total_produccion_2025:,.0f} ton")
print(f"   • Cumplimiento: {cumplimiento_capacidad:.1f}% - {estado_cumplimiento}")

# Análisis por trimestres
trimestres = {
    'Q1 (Ene-Mar)': ['Ene', 'Feb', 'Mar'],
    'Q2 (Abr-Jun)': ['Abr', 'May', 'Jun'],
    'Q3 (Jul-Sep)': ['Jul', 'Ago', 'Sep'],
    'Q4 (Oct-Dic)': ['Oct', 'Nov', 'Dic']
}

print(f"\n📅 PRODUCCIÓN POR TRIMESTRE:")
for trimestre, meses_trim in trimestres.items():
    produccion_trim = plan_produccion.loc[meses_trim, 'Total Producción'].sum()
    capacidad_trim = CAPACIDAD_MENSUAL * 3
    utilizacion_trim = (produccion_trim / capacidad_trim) * 100
    
    print(f"   • {trimestre}: {produccion_trim:,.0f} ton ({utilizacion_trim:.1f}% capacidad)")

# =============================================================================
# RECOMENDACIONES OPERATIVAS 2025
# =============================================================================

print("\n" + "=" * 80)
print("RECOMENDACIONES OPERATIVAS 2025")
print("=" * 80)

if utilizacion_anual < 90:
    print(f"\n💡 OPORTUNIDAD DE MEJORA:")
    print(f"   • Capacidad ociosa disponible: {capacidad_ociosa_anual:,.0f} ton")
    print(f"   • Potencial de ingresos adicionales")
    print(f"   • Considerar expansión de mercado o nuevos clientes")

if meses_sobre_90pct > 6:
    print(f"\n⚡ ALTA DEMANDA DETECTADA:")
    print(f"   • {meses_sobre_90pct} meses operando cerca de capacidad máxima")
    print(f"   • Evaluar optimización de procesos")
    print(f"   • Considerar turnos adicionales en meses pico")

# Recomendaciones específicas por producto
print(f"\n🔧 OPTIMIZACIÓN POR PRODUCTO:")
for producto in productos:
    produccion_producto = plan_produccion[producto].sum()
    mix_producto = (produccion_producto / total_produccion_2025) * 100
    print(f"   • {producto}: {produccion_producto:,.0f} ton ({mix_producto:.1f}% del mix)")


print("\n" + "=" * 80)
print("RESUMEN - PLAN 2025")
print("=" * 80)

resumen_2025 = {
    'Indicador': [
        'Año de Operación',
        'Capacidad Instalada Anual',
        'Producción Programada',
        'Utilización de Capacidad',
        'Producción Promedio Mensual',
        'Meses a Máxima Capacidad',
        'Producto Principal',
        'Beneficio Neto Estimado'
    ],
    'Valor 2025': [
        '2025',
        f"{CAPACIDAD_ANUAL:,} ton",
        f"{total_produccion_2025:,.0f} ton",
        f"{utilizacion_anual:.1f}%",
        f"{produccion_promedio_mensual:.0f} ton",
        f"{meses_sobre_90pct} meses",
        f"{producto_mas_rentable}",
        f"${beneficio:,.0f}"
    ]
}

df_resumen_2025 = pd.DataFrame(resumen_2025)
print("\n" + df_resumen_2025.to_string(index=False))

print(f"\n💎 CONCLUSIÓN: El plan 2025 optimiza la capacidad de {CAPACIDAD_ANUAL:,} ton/año ")
print(f"               con una utilización del {utilizacion_anual:.1f}% y un mix balanceado de productos.")
print("=" * 80)

