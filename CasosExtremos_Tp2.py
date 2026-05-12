import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DEL SISTEMA DIFUSO (Alta Ganancia)
# ==========================================
z_input = ctrl.Antecedent(np.arange(-100, 101, 1), 'Z')
hora_input = ctrl.Antecedent(np.arange(0, 25, 0.1), 'hora')
ventana_output = ctrl.Consequent(np.arange(0, 101, 1), 'ventana')

# Conjuntos Z
z_input['MUY_NEG'] = fuzz.trapmf(z_input.universe, [-100, -100, -10, -5])
z_input['NEG'] = fuzz.trimf(z_input.universe, [-10, -5, 0])
z_input['ZERO'] = fuzz.trimf(z_input.universe, [-5, 0, 5])
z_input['POS'] = fuzz.trimf(z_input.universe, [0, 5, 10])
z_input['MUY_POS'] = fuzz.trapmf(z_input.universe, [5, 10, 100, 100])

# Conjuntos HORA
hora_input['NOCHE'] = fuzz.trapmf(hora_input.universe, [0, 0, 7, 8]) + \
                      fuzz.trapmf(hora_input.universe, [19, 20, 24, 24])
hora_input['DIA'] = fuzz.trapmf(hora_input.universe, [7, 8, 19, 20])

# Conjuntos VENTANA
ventana_output['ABIERTA'] = fuzz.trimf(ventana_output.universe, [0, 0, 25])
ventana_output['CASI_ABIERTA'] = fuzz.trimf(ventana_output.universe, [0, 25, 50])
ventana_output['MITAD'] = fuzz.trimf(ventana_output.universe, [25, 50, 75])
ventana_output['CASI_CERRADA'] = fuzz.trimf(ventana_output.universe, [50, 75, 100])
ventana_output['CERRADA'] = fuzz.trapmf(ventana_output.universe, [75, 100, 100, 100])

# Base de Reglas
reglas = [
    ctrl.Rule(hora_input['DIA'] & z_input['MUY_NEG'], ventana_output['ABIERTA']),
    ctrl.Rule(hora_input['DIA'] & z_input['NEG'], ventana_output['CASI_ABIERTA']),
    ctrl.Rule(hora_input['DIA'] & z_input['ZERO'], ventana_output['MITAD']),
    ctrl.Rule(hora_input['DIA'] & z_input['POS'], ventana_output['CASI_CERRADA']),
    ctrl.Rule(hora_input['DIA'] & z_input['MUY_POS'], ventana_output['CERRADA']),
    ctrl.Rule(hora_input['NOCHE'], ventana_output['CERRADA'])
]

# ==========================================
# 2. FUNCIÓN DE SIMULACIÓN TÉRMICA DINÁMICA
# ==========================================
def simular_difuso(T_media, T_amplitud, estrategia, tau_abierta, tau_cerrada):
    if estrategia in ['centroid', 'mom']:
        ventana_output.defuzzify_method = estrategia
        controlador_ventana = ctrl.ControlSystem(reglas)
        simulador = ctrl.ControlSystemSimulation(controlador_ventana)
    
    dt = 300.0  # Paso: 5 minutos
    t = np.arange(0, 24 * 3600, dt)
    n_steps = len(t)
    
    v0 = 25.0
    
    ve = T_media + T_amplitud * np.sin(2 * np.pi * (t - 9 * 3600) / (24 * 3600))
    v = np.zeros(n_steps)
    v[0] = 20.0 
    apertura_hist = np.zeros(n_steps)

    for i in range(n_steps - 1):
        z_actual = (v[i] - v0) * (ve[i] - v[i])
        hora_actual = (t[i] / 3600) % 24
        
        # --- EL BYPASS DE LOS CASOS BASE ---
        if estrategia == 'siempre_abierta':
            alfa = 0.0
        elif estrategia == 'siempre_cerrada':
            alfa = 100.0
        else:
            simulador.input['Z'] = np.clip(z_actual, -100, 100)
            simulador.input['hora'] = hora_actual
            simulador.compute()
            alfa = simulador.output['ventana']
        # ------------------------------------

        apertura_hist[i] = alfa
        
        # Integración de Euler con las Taus dinámicas
        tau_actual = tau_abierta + (alfa / 100.0) * (tau_cerrada - tau_abierta)
        dv_dt = (ve[i] - v[i]) / tau_actual
        v[i+1] = v[i] + dv_dt * dt
        
    apertura_hist[-1] = apertura_hist[-2]

    # Desempeño (8hs a 20hs)
    idx_8h = int((8 * 3600) / dt)
    idx_20h = int((20 * 3600) / dt)
    J_oficial = np.mean(v[idx_8h:idx_20h] - v0)
    J_absoluto = np.mean(np.abs(v[idx_8h:idx_20h] - v0))
    
    return t/3600, ve, v, apertura_hist, J_oficial, J_absoluto

# ==========================================
# 3. EJECUCIÓN MULTI-ESCENARIO
# ==========================================
def main():
    t_med = 24.0
    t_amp = 8.0
    
    # 1. Definimos las 3 estrategias (Columnas)
    estrategias = [
        ('siempre_cerrada', 'Base A: CERRADA'),
        ('siempre_abierta', 'Base B: ABIERTA'),
        ('centroid', 'Control Difuso (IA)')
    ]

    # 2. Definimos las 4 Físicas de la Habitación (Figuras)
    escenarios_fisicos = [
        ("Ventana Normal (x1)", 1728.0, 17280.0),
        ("Ventana Doble (x2)", 864.0, 17280.0),
        ("Ventana Triple (x3)", 576.0, 15000.0),
        ("Pared de Cristal (Apertura Masiva)", 172.0, 8640.0)
    ]
    
    print("Iniciando simulaciones masivas. Se generarán 4 gráficos distintos...")
    
    for nombre_fisica, t_a, t_c in escenarios_fisicos:
        # Reajustado a 3 columnas y tamaño proporcional
        fig, axs = plt.subplots(2, 3, figsize=(18, 8), dpi=100)
        fig.suptitle(f'Caso 1 - Dinámica Térmica con: {nombre_fisica}', fontsize=16, fontweight='bold')
        
        for col, (codigo_estrategia, titulo) in enumerate(estrategias):
            t_horas, ve, v, ventana, J_of, J_abs = simular_difuso(t_med, t_amp, codigo_estrategia, t_a, t_c)
            
            # Fila 1: Temperaturas
            ax_temp = axs[0, col]
            ax_temp.plot(t_horas, ve, label='Temp. Exterior $v_e(t)$', color='orange', linestyle='--')
            ax_temp.plot(t_horas, v, label='Temp. Interior $v(t)$', color='blue', linewidth=2)
            ax_temp.axhline(25.0, color='green', linestyle=':', label='Confort (25°C)')
            ax_temp.axvspan(8, 20, color='gray', alpha=0.1)
            ax_temp.set_title(f'{titulo}\nJ_abs = {J_abs:.3f}')
            ax_temp.set_ylabel('Temperatura (°C)')
            ax_temp.grid(True)
            ax_temp.set_ylim(15, 35)
            if col == 0: ax_temp.legend(loc='upper left')
            
            # Fila 2: Ventanas
            ax_act = axs[1, col]
            ax_act.plot(t_horas, ventana, color='red', linewidth=2)
            ax_act.fill_between(t_horas, ventana, color='orange', alpha=0.2)
            ax_act.set_yticks([0, 25, 50, 75, 100])
            ax_act.set_yticklabels(['ABIERTA', 'CASI ABIERTA', 'MITAD', 'CASI CERRADA', 'CERRADA'])
            ax_act.set_ylim(-10, 110)
            ax_act.set_xlabel('Hora del día (hs)')
            ax_act.grid(True)
            if col == 0: ax_act.set_ylabel('Estado de la Ventana')

        plt.tight_layout()

    print("Simulaciones completadas. Revisa las 4 ventanas gráficas abiertas.")
    plt.show()

if __name__ == '__main__':
    main()