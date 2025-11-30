import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D  # Necesario para gráficos 3D

# 1. OBTENER CONDICIONES INICIALES REALES
# Cargae el dataset
df = pd.read_csv('ece310131-sup-0004-contolini_palkovacs_drilled_mussel_data.csv')
df_clean = df.dropna(subset=['mussel.length'])

# Definir los 3 estadios basados en tallas (mm)
# J: < 30mm, S: 30-60mm, A: > 60mm
J_data = df_clean[df_clean['mussel.length'] < 30]
S_data = df_clean[(df_clean['mussel.length'] >= 30) & (df_clean['mussel.length'] <= 60)]
A_data = df_clean[df_clean['mussel.length'] > 60]

# Contar cuántos hay de cada uno para usar como punto de partida (t=0)
# Normalizar dividiendo por 10 para simular densidad por m2 (aprox)
J0 = len(J_data) / 10
S0 = len(S_data) / 10
A0 = len(A_data) / 10

print(f"--- Condiciones Iniciales Reales ---")
print(f"Juveniles (J0): {J0:.1f}")
print(f"Sub-adultos (S0): {S0:.1f}")
print(f"Adultos (A0): {A0:.1f}")


# 2. DEFINIR EL MODELO (SISTEMA DE EDOs) - Modificado con Capacidad de Carga
def modelo_estructurado(y, t, f, g1, g2, m1, m2, m3, K):
    """
    y: Vector [Juveniles, Subadultos, Adultos]
    f: Tasa de fecundidad (Larvas producidas por adulto)
    g1, g2: Tasas de crecimiento (transición entre estadios)
    m1, m2, m3: Tasas de mortalidad natural por estadio
    K: Capacidad de carga total (densidad máxima total de J+S+A)
    """
    J, S, A = y
    N_total = J + S + A  # Densidad total de la población

    # Ecuaciones diferenciales del ciclo de vida
    # Incorpora un término logístico (1 - N_total/K) en el reclutamiento
    dJdt = (f * A) * (1 - N_total / K) - (g1 * J) - (m1 * J)
    dSdt = (g1 * J) - (g2 * S) - (m2 * S)
    dAdt = (g2 * S) - (m3 * A)

    return [dJdt, dSdt, dAdt]


# 3. PARÁMETROS BIOLÓGICOS (Hipotéticos pero realistas para mejillones)
# fecundidad: alta (muchas larvas), pero pocas sobreviven al asentamiento
f = 2.5  # Recruitment rate (larvas que logran asentarse por adulto)
g1 = 0.3  # 30% de juveniles pasan a subadultos por unidad de tiempo
g2 = 0.2  # 20% de subadultos pasan a adultos
m1 = 0.5  # Alta mortalidad en juveniles (muy vulnerables)
m2 = 0.1  # Baja mortalidad en medianos
m3 = 0.15  # Mortalidad en adultos (vejez o grandes depredadores como LOM)
K = 200.0  # Densidad total máxima (J+S+A) que puede soportar el ambiente

print(f"\n--- Parámetros del Modelo (Ajustados) ---")
print(f"f (fecundidad): {f}")
print(f"g1 (crecimiento J->S): {g1}")
print(f"g2 (crecimiento S->A): {g2}")
print(f"m1 (mortalidad J): {m1}")
print(f"m2 (mortalidad S): {m2}")
print(f"m3 (mortalidad A): {m3}")
print(f"K (capacidad de carga): {K}")

# Tiempo de simulación (50 meses)
t = np.linspace(0, 50, 1000)
y0 = [J0, S0, A0]

# 4. RESOLVER EL SISTEMA
solucion = odeint(modelo_estructurado, y0, t, args=(f, g1, g2, m1, m2, m3, K))
J_sim, S_sim, A_sim = solucion.T

# 5. CALCULAR LA DENSIDAD TOTAL A LO LARGO DEL TIEMPO
N_total_sim = J_sim + S_sim + A_sim

# 6. VISUALIZACIÓN
# Crear una figura con subplots: 1 fila, 3 columnas
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Gráfico 1: Dinámica Temporal por Estadio
ax1.plot(t, J_sim, label='Juveniles (<30mm)', color='#66CCEE', linewidth=2)
ax1.plot(t, S_sim, label='Sub-adultos (30-60mm)', color='#EE6677', linewidth=2, linestyle='--')
ax1.plot(t, A_sim, label='Adultos (>60mm)', color='#228833', linewidth=2)
ax1.set_title('Dinámica Poblacional por Estadio', fontsize=12)
ax1.set_xlabel('Tiempo (unidades de tiempo)', fontsize=10)
ax1.set_ylabel('Densidad de Población', fontsize=10)
ax1.legend()
ax1.grid(alpha=0.3)

# Gráfico 2: Retrato de Fase (Espacio de Estados 3D)
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot(J_sim, S_sim, A_sim, lw=1.5, color='purple', label='Trayectoria del Sistema')
# Opcional: Marcar el punto inicial
ax2.scatter(J0, S0, A0, color='red', s=50, label='Condición Inicial', alpha=1.0)
# Opcional: Marcar el punto final
ax2.scatter(J_sim[-1], S_sim[-1], A_sim[-1], color='green', s=50, label='Condición Final', alpha=1.0)

ax2.set_xlabel('Juveniles (J)')
ax2.set_ylabel('Sub-adultos (S)')
ax2.set_zlabel('Adultos (A)')
ax2.set_title('Retrato de Fase 3D', fontsize=12)
ax2.legend()

# Gráfico 3: Densidad Total en el Tiempo
ax3.plot(t, N_total_sim, label='Densidad Total (J+S+A)', color='black', linewidth=2)
# Línea horizontal para mostrar K
ax3.axhline(y=K, color='red', linestyle=':', label=f'Capacidad de Carga (K={K})', linewidth=2)
ax3.set_title('Densidad Total de la Población', fontsize=12)
ax3.set_xlabel('Tiempo (unidades de tiempo)', fontsize=10)
ax3.set_ylabel('Densidad Total', fontsize=10)
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Análisis de estabilidad (Opcional: ver si la población colapsa o crece)
print("\n--- Resultado Final (t=50) ---")
print(f"Juveniles: {J_sim[-1]:.1f}")
print(f"Sub-adultos: {S_sim[-1]:.1f}")
print(f"Adultos: {A_sim[-1]:.1f}")
print(f"Densidad Total: {N_total_sim[-1]:.1f}")  # Usamos la nueva variable
print(f"Capacidad de Carga (K): {K}")

if N_total_sim[-1] <= 0.01:  # Usamos la nueva variable
    print("\nAdvertencia: La población total ha colapsado.")
elif abs(N_total_sim[-1] - K) < 0.05 * K:  # Usamos la nueva variable
    print("\nLa población total parece haber alcanzado la capacidad de carga (K).")
else:
    print(
        "\nLa población total ha alcanzado un equilibrio o estado estacionario (posiblemente oscilatorio amortiguada).")
