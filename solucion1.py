import numpy as np
import matplotlib.pyplot as plt

def simular_control_bang_bang():
    # 1. Parámetros de simulación y tiempo
    dt = 60.0  # Paso de integración: 60 segundos (1 minuto)
    horas_totales = 24
    t = np.arange(0, horas_totales * 3600, dt) # Vector de tiempo en segundos
    n_steps = len(t)
    
    # 2. Parámetros físicos del sistema
    v0 = 25.0  # Temperatura de confort [cite: 1687]
    
    # Despejamos las constantes de tiempo (tau = R_eq * C)
    # tau_abierta = R * C
    # tau_cerrada = (R + Rvmax) * C
    tau_abierta = (2.4 * 3600) / 5.0  # 1728 segundos [cite: 1687]
    tau_cerrada = (24 * 3600) / 5.0   # 17280 segundos [cite: 1687]
    
    # 3. Generación de la Temperatura Exterior (ve)
    # El práctico pide 3 series: usaremos la que incluye la temp. de confort (Ej: 15°C a 33°C) [cite: 1688, 1689]
    # Usamos una función senoidal que simula el ciclo diario (mínima a la madrugada, máxima a la tarde)
    ve = 24.0 + 9.0 * np.sin(2 * np.pi * (t - 9 * 3600) / (24 * 3600))
    
    # 4. Inicialización de vectores de estado
    v = np.zeros(n_steps)
    v[0] = 20.0  # Supongamos que la habitación arranca a 20°C a las 00:00
    
    # Vector para guardar el estado de la ventana con fines de graficación (1=Cerrada, 0=Abierta)
    ventana_estado = np.zeros(n_steps) 

    # 5. Bucle de Integración Numérica (Método de Euler)
    for i in range(n_steps - 1):
        # A. Lógica de Control (Teorema de Lyapunov)
        Z = (v[i] - v0) * (ve[i] - v[i]) 
        
        if Z >= 0:
            # El entorno nos aleja del confort -> CERRAR ventana [cite: 
            tau_actual = tau_cerrada
            ventana_estado[i] = 1 
        else:
            # El entorno nos acerca al confort -> ABRIR ventana [cite: 1697]
            tau_actual = tau_abierta
            ventana_estado[i] = 0
            
        # B. Ecuación Diferencial de la Planta
        # dv_dt = (ve - v) / (Req * C) [cite: 1685]
        dv_dt = (ve[i] - v[i]) / tau_actual
        
        # C. Integración de Euler
        v[i+1] = v[i] + dv_dt * dt
        
    # Asignamos el último estado de la ventana para que los vectores tengan el mismo tamaño
    ventana_estado[-1] = ventana_estado[-2]

    # 6. Cálculo del Índice de Desempeño J (de 8:00 a 20:00) [cite: 1706, 1707]
    # Índices correspondientes a las 8 hs y 20 hs
    idx_8h = int((8 * 3600) / dt)
    idx_20h = int((20 * 3600) / dt)
    
    # Cálculo de la integral usando el promedio (equivalente discreto exacto al dividir por Tf-Ti)
    # Nota analítica: El TP pide (v - v0). Usaremos np.abs(v - v0) adicionalmente 
    # para tener una métrica estricta del error real.
    J_oficial = np.mean(v[idx_8h:idx_20h] - v0)
    J_absoluto = np.mean(np.abs(v[idx_8h:idx_20h] - v0))

    # 7. Graficación de Resultados
    plt.figure(figsize=(12, 8))
    
    # Gráfico de Temperaturas
    plt.subplot(2, 1, 1)
    plt.plot(t/3600, ve, label='Temp. Exterior $v_e(t)$', color='orange', linestyle='--')
    plt.plot(t/3600, v, label='Temp. Interior $v(t)$', color='blue', linewidth=2)
    plt.axhline(v0, color='green', linestyle=':', label='Confort $v_0$ (25°C)')
    plt.axvspan(8, 20, color='gray', alpha=0.1, label='Periodo de Evaluación (8h-20h)')
    plt.title(f'Control Bang-Bang | J_oficial = {J_oficial:.3f} | J_absoluto = {J_absoluto:.3f}')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True)
    
    # Gráfico del Actuador
    plt.subplot(2, 1, 2)
    plt.step(t/3600, ventana_estado, where='post', color='red')
    plt.yticks([0, 1], ['ABIERTA (0)', 'CERRADA (Rvmax)'])
    plt.xlabel('Hora del día (hs)')
    plt.title('Estado del Actuador (Ventana)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Ejecutar la simulación
if __name__ == '__main__':
    simular_control_bang_bang()