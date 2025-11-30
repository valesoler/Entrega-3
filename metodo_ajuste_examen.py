import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
import seaborn as sns

# Configurar estilo de Seaborn para que se parezca más a la gráfica del artículo
sns.set_style("white")
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.figsize': (10, 4)  # Ancho mayor para acomodar subgráficos
})

# 1. Leer los datos desde el archivo CSV
df = pd.read_csv('ece310131-sup-0004-contolini_palkovacs_drilled_mussel_data.csv')

# 2. Filtrar solo los mejillones que fueron perforados (drilled == TRUE) y que tienen una longitud medida
df_drilled = df[df['drilled'] == True].dropna(subset=['mussel.length'])


# 3. Definir la función gaussiana (modelo biológico)
def gaussian(x, A, mu, sigma):
    """
    Función Gaussiana para modelar la preferencia de tamaño de presa.

    Parámetros:
    - x: Longitud de la presa (mejillón).
    - A: Amplitud (frecuencia máxima relativa).
    - mu: Media (tamaño óptimo de presa).
    - sigma: Desviación estándar (ancho del nicho de tamaño).
    """
    # Aseguramos que sigma no sea cero o negativo para evitar errores
    sigma = abs(sigma)
    if sigma == 0:
        sigma = 1e-6
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


# 4. Ajustar el modelo para cada tratamiento
treatments = ['HOP', 'SOB', 'LOM']
colors = {'HOP': '#4477AA', 'SOB': '#CC6677', 'LOM': '#DDCC77'}

# Crear subgráficos
fig, axes = plt.subplots(1, len(treatments), figsize=(15, 4))

results = {}

for i, treatment in enumerate(treatments):
    ax = axes[i]

    # Filtrar datos para el tratamiento actual
    df_treatment = df_drilled[df_drilled['trtmnt'] == treatment]
    lengths = df_treatment['mussel.length'].values

    if len(lengths) == 0:
        print(f"No hay datos de mejillones perforados para el tratamiento {treatment}")
        continue

    # Calcular histograma para usar como datos de ajuste
    # El número de bins puede afectar el ajuste, usar uno razonable
    num_bins = min(20, len(lengths))  # Ajustar número de bins si hay pocos datos
    hist_values, bin_edges = np.histogram(lengths, bins=num_bins, density=False)  # density=False para contar
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Asegurar que no haya bins con frecuencia cero si es posible, o manejarlos
    # Filtrar los puntos donde la frecuencia es cero para el ajuste
    mask = hist_values > 0
    x_data = bin_centers[mask]
    y_data = hist_values[mask]

    if len(x_data) < 3:
        print(f"No hay suficientes puntos (más de 2) con frecuencia > 0 para ajustar {treatment}. Datos: {len(x_data)}")
        continue

    # Proporcionar valores iniciales razonables para A, mu, sigma
    # A: Valor máximo de la frecuencia observada
    # mu: Media de las longitudes observadas
    # sigma: Desviación estándar de las longitudes observadas
    initial_A = np.max(y_data) if len(y_data) > 0 else 1.0
    initial_mu = np.mean(lengths)
    initial_sigma = np.std(lengths)

    # Asegurar sigma inicial no sea cero
    if initial_sigma == 0:
        initial_sigma = 1.0

    initial_guess = [initial_A, initial_mu, initial_sigma]

    try:
        # Ajustar la curva gaussiana a los datos del histograma
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess, maxfev=5000)
        A_opt, mu_opt, sigma_opt = popt

        # Calcular los valores predichos por el modelo ajustado
        x_model = np.linspace(x_data.min(), x_data.max(), 200)
        y_model = gaussian(x_model, A_opt, mu_opt, sigma_opt)

        # Graficar
        # Dibujar puntos del histograma (frecuencia vs centro del bin)
        ax.scatter(x_data, y_data, color=colors[treatment], s=30, alpha=0.8, label='Datos (Bins)', zorder=3)

        # Dibujar la curva gaussiana ajustada
        ax.plot(x_model, y_model, color=colors[treatment], linewidth=2, label='Ajuste Gaussiano', zorder=2)

        ax.set_title(f'{treatment}\nA={A_opt:.2f}, μ={mu_opt:.2f}, σ={sigma_opt:.2f}', fontsize=10)
        ax.set_xlabel('Longitud del Mejillón (mm)', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Almacenar resultados
        results[treatment] = {'A': A_opt, 'mu': mu_opt, 'sigma': sigma_opt, 'covariance': pcov}

    except RuntimeError as e:
        print(f"Error al ajustar el modelo para {treatment}: {e}")
        # Graficar solo los datos
        ax.scatter(x_data, y_data, color=colors[treatment], s=30, alpha=0.8, label='Datos (Bins)', zorder=3)
        ax.set_title(f'{treatment}\n(Error en ajuste)', fontsize=10)
        ax.set_xlabel('Longitud del Mejillón (mm)', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# 5. Imprimir resultados de los parámetros ajustados
print("\n--- Parámetros Ajustados del Modelo Gaussiano ---")
for treatment, params in results.items():
    print(f"\nTratamiento {treatment}:")
    print(f"  Amplitud (A): {params['A']:.4f}")
    print(f"  Media (μ): {params['mu']:.4f} mm")
    print(f"  Desviación (σ): {params['sigma']:.4f} mm")

if not results:
    print("\nNo se pudo ajustar ningún modelo debido a la falta de datos adecuados.")


# --- AJUSTE EXPONENCIAL DECRECIENTE PARA LOM ---
# Filtrar datos para el tratamiento LOM
treatment = 'LOM'
df_treatment = df_drilled[df_drilled['trtmnt'] == treatment]
lengths = df_treatment['mussel.length'].values

if len(lengths) == 0:
    print(f"No hay datos de mejillones perforados para el tratamiento {treatment}")
else:
    # --- DEFINIR FUNCIONES ---
    def exponential_decay(x, A, lam, y0=0):
        """
        Función Exponencial Decreciente.
        """
        A = abs(A)
        lam = abs(lam)
        return A * np.exp(-lam * x) + y0


    # --- PREPARAR DATOS PARA EL AJUSTE ---
    num_bins = min(20, len(lengths))
    hist_values, bin_edges = np.histogram(lengths, bins=num_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = hist_values > 0
    x_data = bin_centers[mask]
    y_data = hist_values[mask]

    if len(x_data) < 3:
        print(f"No hay suficientes puntos (más de 2) con frecuencia > 0 para ajustar {treatment}. Datos: {len(x_data)}")
    else:
        # --- AJUSTAR EL MODELO EXPONENCIAL ---
        # Valores iniciales razonables para exp decay
        initial_A = np.max(y_data)  # Frecuencia máxima en el rango observado
        initial_lam = np.log(2) / (np.mean(x_data) if np.mean(x_data) > 0 else 1.0)  # Estimación burda de lambda
        initial_y0 = 0  # Empezamos sin offset
        initial_guess = [initial_A, initial_lam, initial_y0]

        try:
            # Ajustar el modelo exponencial decreciente
            popt, pcov = curve_fit(exponential_decay, x_data, y_data, p0=initial_guess, maxfev=5000)
            A_opt, lam_opt, y0_opt = popt

            # Calcular los valores predichos por el modelo ajustado
            x_model = np.linspace(x_data.min(), x_data.max(), 200)
            y_model = exponential_decay(x_model, A_opt, lam_opt, y0_opt)

            # --- VISUALIZAR ---
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, color='orange', s=50, alpha=0.8, label='Datos (Bins)', zorder=3)
            plt.plot(x_model, y_model, color='orange', linewidth=2, label='Ajuste Exponencial Decreciente', zorder=2)

            plt.title(f'{treatment}\nA={A_opt:.2f}, λ={lam_opt:.4f}, y0={y0_opt:.2f}', fontsize=10)
            plt.xlabel('Longitud del Mejillón (mm)', fontsize=10)
            plt.ylabel('Frecuencia', fontsize=10)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

            # --- IMPRIMIR RESULTADOS ---
            print(f"\n--- Parámetros Ajustados del Modelo Exponencial Decreciente para {treatment} ---")
            print(f"  Amplitud (A): {A_opt:.4f}")
            print(f"  Tasa de Decaimiento (λ): {lam_opt:.4f}")
            print(f"  Offset (y0): {y0_opt:.4f}")

        except RuntimeError as e:
            print(f"Error al ajustar el modelo exponencial decreciente para {treatment}: {e}")

# --- AJUSTE SKEW-NORMAL ---

if len(lengths) == 0:
    print(f"No hay datos de mejillones perforados para el tratamiento {treatment}")
else:
    # --- DEFINIR FUNCIONES ---
    def skewnorm_scaled(x, A, xi, omega, alpha):
        """
        Función Skew-Normal escalada para ajustar a frecuencias de histograma.

        Parámetros:
        - x: Longitud de la presa (mejillón).
        - A: Amplitud (factor de escala para la altura).
        - xi: Parámetro de ubicación (centro base).
        - omega: Parámetro de escala (ancho/variabilidad).
        - alpha: Parámetro de forma (sesgo).
        """
        # Asegurar que omega no sea negativo
        omega = abs(omega)
        # Calcular la densidad de probabilidad de la Skew-Normal
        pdf_values = skewnorm.pdf(x, a=alpha, loc=xi, scale=omega)
        # Escalarla por A para que coincida con la frecuencia
        return A * pdf_values


    # --- PREPARAR DATOS PARA EL AJUSTE ---
    num_bins = min(20, len(lengths))
    hist_values, bin_edges = np.histogram(lengths, bins=num_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filtrar puntos donde la frecuencia es cero para el ajuste
    mask = hist_values > 0
    x_data = bin_centers[mask]
    y_data = hist_values[mask]

    if len(x_data) < 3:
        print(f"No hay suficientes puntos (más de 2) con frecuencia > 0 para ajustar {treatment}. Datos: {len(x_data)}")
    else:
        # --- AJUSTAR EL MODELO SKEW-NORMAL ---
        # Valores iniciales razonables para Skew-Normal
        # A: Valor máximo de la frecuencia observada
        initial_A = np.max(y_data) if len(y_data) > 0 else 1.0
        # xi: Media de las longitudes observadas (punto central aproximado)
        initial_xi = np.mean(lengths)
        # omega: Desviación estándar de las longitudes observadas
        initial_omega = np.std(lengths)
        # alpha: Valor inicial pequeño para permitir sesgo positivo o negativo
        initial_alpha = 0.0

        if initial_omega == 0:
            initial_omega = 1.0

        initial_guess = [initial_A, initial_xi, initial_omega, initial_alpha]

        try:
            # Ajustar el modelo Skew-Normal escalado
            popt, pcov = curve_fit(skewnorm_scaled, x_data, y_data, p0=initial_guess, maxfev=5000)
            A_opt, xi_opt, omega_opt, alpha_opt = popt

            # Calcular los valores predichos por el modelo ajustado
            x_model = np.linspace(x_data.min(), x_data.max(), 200)
            y_model = skewnorm_scaled(x_model, A_opt, xi_opt, omega_opt, alpha_opt)

            # --- VISUALIZAR ---
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, color='orange', s=50, alpha=0.8, label='Datos (Bins)', zorder=3)
            plt.plot(x_model, y_model, color='orange', linewidth=2, label='Ajuste Skew-Normal', zorder=2)

            plt.title(f'{treatment}\nA={A_opt:.2f}, ξ={xi_opt:.2f}, ω={omega_opt:.2f}, α={alpha_opt:.2f}', fontsize=10)
            plt.xlabel('Longitud del Mejillón (mm)', fontsize=10)
            plt.ylabel('Frecuencia', fontsize=10)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

            # --- IMPRIMIR RESULTADOS ---
            print(f"\n--- Parámetros Ajustados del Modelo Skew-Normal para {treatment} ---")
            print(f"  Amplitud (A): {A_opt:.4f}")
            print(f"  Ubicación (ξ): {xi_opt:.4f}")
            print(f"  Escala (ω): {omega_opt:.4f}")
            print(f"  Forma (α): {alpha_opt:.4f}")

        except RuntimeError as e:
            print(f"Error al ajustar el modelo Skew-Normal para {treatment}: {e}")
