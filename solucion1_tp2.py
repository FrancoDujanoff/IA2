import numpy as np
import matplotlib.pyplot as plt

def simular_dia(T_media, T_amplitud):
    dt = 60.0  # Paso: 60 segundos
    t = np.arange(0, 24 * 3600, dt)
    n_steps = len(t)
    
    v0 = 25.0  # Confort
    tau_abierta = (2.4 * 3600) / 5.0
    tau_cerrada = (24 * 3600) / 5.0
    
    # Modelo climático senoidal
    # Desfase de 9 horas para que el pico sea a las 15:00 y el valle a las 03:00
    ve = T_media + T_amplitud * np.sin(2 * np.pi * (t - 9 * 3600) / (24 * 3600))
    
    v = np.zeros(n_steps)
    v[0] = 20.0  # Temperatura inicial
    ventana_estado = np.zeros(n_steps)

    # Bucle de Integración
    for i in range(n_steps - 1):
        # Lógica Bang-Bang
        Z = (v[i] - v0) * (ve[i] - v[i])
        
        if Z >= 0:
            tau_actual = tau_cerrada
            ventana_estado[i] = 1  # Cerrada
        else:
            tau_actual = tau_abierta
            ventana_estado[i] = 0  # Abierta
            
        # Ecuación de la planta y Euler
        dv_dt = (ve[i] - v[i]) / tau_actual
        v[i+1] = v[i] + dv_dt * dt
        
    ventana_estado[-1] = ventana_estado[-2]

    # Cálculo del Desempeño J (8:00 a 20:00)
    idx_8h = int((8 * 3600) / dt)
    idx_20h = int((20 * 3600) / dt)
    
    J_oficial = np.mean(v[idx_8h:idx_20h] - v0)
    J_absoluto = np.mean(np.abs(v[idx_8h:idx_20h] - v0))
    
    return t/3600, ve, v, ventana_estado, J_oficial, J_absoluto

def main():
    estaciones = [
        ("VERANO", 26.0, 9.0),
        ("OTOÑO", 16.0, 7.0),
        ("INVIERNO", 9.0, 6.0),
        ("PRIMAVERA", 20.0, 8.0)
    ]
    
    # Iteramos sobre cada estación
    for nombre, t_med, t_amp in estaciones:
        # 1. Simulamos el día
        t_horas, ve, v, ventana, J_of, J_abs = simular_dia(t_med, t_amp)
        
        # 2. Creamos una NUEVA ventana por cada iteración
        # figsize=(10, 8) da un buen tamaño, dpi=120 mejora la resolución (anti-pixelado)
        fig, (ax_temp, ax_act) = plt.subplots(2, 1, figsize=(10, 8), dpi=120)
        fig.suptitle(f'Control Bang-Bang - {nombre}', fontsize=16, fontweight='bold')
        
        # 3. Gráfico de Temperaturas (Arriba)
        ax_temp.plot(t_horas, ve, label='Temp. Exterior $v_e(t)$', color='orange', linestyle='--')
        ax_temp.plot(t_horas, v, label='Temp. Interior $v(t)$', color='blue', linewidth=2)
        ax_temp.axhline(25.0, color='green', linestyle=':', label='Confort $v_0$ (25°C)')
        ax_temp.axvspan(8, 20, color='gray', alpha=0.1, label='Periodo de Evaluación (8h-20h)')
        
        # Mostramos los índices J en el título del subgráfico
        ax_temp.set_title(f'Desempeño: J_oficial = {J_of:.3f} | J_absoluto = {J_abs:.3f}')
        ax_temp.set_ylabel('Temperatura (°C)')
        ax_temp.legend(loc='upper right')
        ax_temp.grid(True)
        
        # 4. Gráfico del Actuador (Abajo)
        ax_act.step(t_horas, ventana, where='post', color='red', linewidth=2)
        ax_act.set_yticks([0, 1])
        ax_act.set_yticklabels(['ABIERTA (0)', 'CERRADA (Rvmax)'])
        ax_act.set_xlabel('Hora del día (hs)')
        ax_act.set_title('Estado del Actuador (Ventana)')
        ax_act.grid(True)
        
        # 5. Ajustar el diseño para que no se corten los textos en esta ventana
        plt.tight_layout()

    # 6. Mostrar todas las ventanas creadas al mismo tiempo
    plt.show()

if __name__ == '__main__':
    main()