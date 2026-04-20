import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DEL SISTEMA DIFUSO
# ==========================================
z_input = ctrl.Antecedent(np.arange(-100, 101, 1), 'Z')
hora_input = ctrl.Antecedent(np.arange(0, 25, 0.1), 'hora')
ventana_output = ctrl.Consequent(np.arange(0, 101, 1), 'ventana')

# Conjuntos Z
z_input['MUY_NEG'] = fuzz.trapmf(z_input.universe, [-100, -100, -75, -25])
z_input['NEG'] = fuzz.trimf(z_input.universe, [-75, -25, 0])
z_input['ZERO'] = fuzz.trimf(z_input.universe, [-25, 0, 25])
z_input['POS'] = fuzz.trimf(z_input.universe, [0, 25, 75])
z_input['MUY_POS'] = fuzz.trapmf(z_input.universe, [25, 75, 100, 100])

# Conjuntos HORA
hora_input['NOCHE'] = fuzz.trapmf(hora_input.universe, [0, 0, 7, 8]) + \
                      fuzz.trapmf(hora_input.universe, [19, 20, 24, 24])
hora_input['DIA'] = fuzz.trapmf(hora_input.universe, [7, 8, 19, 20])

# Conjuntos VENTANA (Porcentaje de Resistencia)
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

controlador_ventana = ctrl.ControlSystem(reglas)
simulador = ctrl.ControlSystemSimulation(controlador_ventana)

# ==========================================
# 2. FUNCIÓN DE SIMULACIÓN TÉRMICA Y DESEMPEÑO
# ==========================================
def simular_difuso(T_media, T_amplitud):
    dt = 60.0  # Paso: 60 segundos
    t = np.arange(0, 24 * 3600, dt)
    n_steps = len(t)
    
    v0 = 25.0
    tau_abierta = (2.4 * 3600) / 5.0  # 1728 s
    tau_cerrada = (24 * 3600) / 5.0   # 17280 s
    
    # Clima senoidal
    ve = T_media + T_amplitud * np.sin(2 * np.pi * (t - 9 * 3600) / (24 * 3600))
    
    v = np.zeros(n_steps)
    v[0] = 20.0  # Temperatura inicial a la medianoche
    apertura_hist = np.zeros(n_steps)

    for i in range(n_steps - 1):
        # Entradas al motor difuso
        z_actual = (v[i] - v0) * (ve[i] - v[i])
        hora_actual = (t[i] / 3600) % 24
        
        simulador.input['Z'] = np.clip(z_actual, -100, 100)
        simulador.input['hora'] = hora_actual
        simulador.compute()
        
        # Salida del motor (0% a 100% de resistencia)
        alfa = simulador.output['ventana']
        apertura_hist[i] = alfa
        
        # Traducción a constante de tiempo
        tau_actual = tau_abierta + (alfa / 100.0) * (tau_cerrada - tau_abierta)
        
        # Euler
        dv_dt = (ve[i] - v[i]) / tau_actual
        v[i+1] = v[i] + dv_dt * dt
        
    apertura_hist[-1] = apertura_hist[-2]

    # Cálculo del Índice de Desempeño J (de 8:00 a 20:00)
    idx_8h = int((8 * 3600) / dt)
    idx_20h = int((20 * 3600) / dt)
    
    J_oficial = np.mean(v[idx_8h:idx_20h] - v0)
    J_absoluto = np.mean(np.abs(v[idx_8h:idx_20h] - v0))
    
    return t/3600, ve, v, apertura_hist, J_oficial, J_absoluto

# ==========================================
# 3. EJECUCIÓN: COMPARATIVA DE LAS 4 ESTACIONES
# ==========================================
def main():
    estaciones = [
        ("CASO 1", 24.0, 8.0),
        ("CASO 2", 33.0, 5.0),
        ("CASO 3", 12.0, 8.0),
    ]
    
    print("Iniciando simulaciones difusas (esto puede tomar unos segundos)...")
    
    for nombre, t_med, t_amp in estaciones:
        t_horas, ve, v, ventana, J_of, J_abs = simular_difuso(t_med, t_amp)
        
        fig, (ax_temp, ax_act) = plt.subplots(2, 1, figsize=(10, 8), dpi=120)
        fig.suptitle(f'Control Difuso Suave - {nombre}', fontsize=16, fontweight='bold')
        
        # Gráfico Temperaturas
        ax_temp.plot(t_horas, ve, label='Temp. Exterior $v_e(t)$', color='orange', linestyle='--')
        ax_temp.plot(t_horas, v, label='Temp. Interior $v(t)$', color='blue', linewidth=2)
        ax_temp.axhline(25.0, color='green', linestyle=':', label='Confort $v_0$ (25°C)')
        ax_temp.axvspan(8, 20, color='gray', alpha=0.1, label='Evaluación (8h-20h)')
        
        ax_temp.set_title(f'Desempeño: J_oficial = {J_of:.3f} | J_absoluto = {J_abs:.3f}')
        ax_temp.set_ylabel('Temperatura (°C)')
        ax_temp.legend(loc='upper right')
        ax_temp.grid(True)
        
        # Gráfico Actuador
        ax_act.plot(t_horas, ventana, color='red', linewidth=2)
        ax_act.fill_between(t_horas, ventana, color='orange', alpha=0.2)
        
        # --- CAMBIOS APLICADOS AQUÍ ---
        ax_act.set_yticks([0, 25, 50, 75, 100])
        ax_act.set_yticklabels(['ABIERTA', 'CASI ABIERTA', 'MITAD', 'CASI CERRADA', 'CERRADA'])
        ax_act.set_ylabel('Estado de la Ventana')
        ax_act.set_title('Acción del Controlador Difuso')
        # ------------------------------
        
        ax_act.set_ylim(-10, 110) # Un poco más de margen visual
        ax_act.set_xlabel('Hora del día (hs)')
        ax_act.grid(True)
        
        plt.tight_layout()

    print("Simulaciones completadas. Mostrando gráficos...")
    plt.show()

if __name__ == '__main__':
    main()