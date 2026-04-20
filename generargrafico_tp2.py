import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def generar_grafica_defuzzificacion():
    # 1. Universo de Salida (0% a 100% de Apertura)
    x_ventana = np.arange(0, 101, 1)

    # 2. Definición de los conjuntos de salida involucrados
    mitad = fuzz.trimf(x_ventana, [25, 50, 75])
    casi_cerrada = fuzz.trimf(x_ventana, [50, 75, 100])

    # 3. Simulamos un instante (ej. Z = 15) donde se activan dos reglas
    # Nivel de confianza o activación (Implicación por Mínimo)
    activacion_mitad = 0.40
    activacion_casi_cerrada = 0.60

    # 4. Recorte de los conjuntos (Mamdani IF-THEN)
    mitad_recortado = np.fmin(activacion_mitad, mitad)
    casi_cerrada_recortado = np.fmin(activacion_casi_cerrada, casi_cerrada)

    # 5. Fusión de las áreas (Agregación por Máximo)
    area_agregada = np.fmax(mitad_recortado, casi_cerrada_recortado)

    # 6. Desborrosificación (Cálculo de los dos métodos)
    # COG: Center of Gravity (Centroide) -> El que usamos en la simulación
    centroide = fuzz.defuzz(x_ventana, area_agregada, 'centroid')
    
    # MOM: Mean of Maxima (Promedio de los Máximos) -> El de tu clase
    mom = fuzz.defuzz(x_ventana, area_agregada, 'mom')

    # ==========================================
    # 7. RENDERIZADO VISUAL
    # ==========================================
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title('Análisis de Desborrosificación (Z = 15)', fontsize=14, fontweight='bold')
    
    # Dibujar los triángulos originales de fondo (líneas punteadas)
    plt.plot(x_ventana, mitad, 'b--', linewidth=1.5, alpha=0.5, label='Conjunto: MITAD')
    plt.plot(x_ventana, casi_cerrada, 'g--', linewidth=1.5, alpha=0.5, label='Conjunto: CASI CERRADA')

    # Rellenar el área agregada que evalúa el sistema
    plt.fill_between(x_ventana, 0, area_agregada, color='slateblue', alpha=0.4, label='Área de Decisión (Agregada)')
    plt.plot(x_ventana, area_agregada, color='midnightblue', linewidth=2)

    # Línea del Centroide (Continuo)
    plt.axvline(centroide, color='red', linewidth=3, 
                label=f'Centroide (COG): {centroide:.1f}%')
    
    # Línea del Máximo (Discreto)
    plt.axvline(mom, color='darkorange', linewidth=3, linestyle='-.', 
                label=f'Máximo (MOM): {mom:.1f}%')

    # Estilos
    plt.ylabel('Grado de Pertenencia (μ)')
    plt.xlabel('Apertura de Ventana (%)')
    plt.ylim(0, 1.1)
    plt.xlim(0, 100)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    generar_grafica_defuzzificacion()