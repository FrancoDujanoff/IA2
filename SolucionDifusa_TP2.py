import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DEL SISTEMA DIFUSO
# ==========================================
z_input = ctrl.Antecedent(np.arange(-100, 101, 1), 'Z')
hora_input = ctrl.Antecedent(np.arange(0, 25, 0.1), 'hora')
t_pred_input = ctrl.Antecedent(np.arange(0, 51, 1), 't_predicha')
ventana_output = ctrl.Consequent(np.arange(0, 101, 1), 'ventana')

# ==========================================
# 2. Borrosificacion
# ==========================================
# Conjuntos Z
z_input['MUY_NEG'] = fuzz.trapmf(z_input.universe, [-100, -100, -10, -5])
z_input['NEG'] = fuzz.trimf(z_input.universe, [-10, -5, 0])
z_input['ZERO'] = fuzz.trimf(z_input.universe, [-5, 0, 5])
z_input['POS'] = fuzz.trimf(z_input.universe, [0, 5, 10])
z_input['MUY_POS'] = fuzz.trapmf(z_input.universe, [5, 10, 100, 100])

# Conjuntos HORA
hora_input['NOCHE'] = fuzz.trapmf(hora_input.universe, [0, 0, 7, 8]) + fuzz.trapmf(hora_input.universe, [19, 20, 24, 24])
hora_input['DIA'] = fuzz.trapmf(hora_input.universe, [7, 8, 19, 20])

# Conjuntos de Temperatura Predicha
t_pred_input['BAJA'] = fuzz.trapmf(t_pred_input.universe, [0, 0, 22, 28])
t_pred_input['ALTA'] = fuzz.trapmf(t_pred_input.universe, [22, 28, 50, 50])

# Conjuntos VENTANA (Porcentaje de Resistencia)
ventana_output['ABIERTA'] = fuzz.trimf(ventana_output.universe, [0, 0, 25])
ventana_output['CASI_ABIERTA'] = fuzz.trimf(ventana_output.universe, [0, 25, 50])
ventana_output['MITAD'] = fuzz.trimf(ventana_output.universe, [25, 50, 75])
ventana_output['CASI_CERRADA'] = fuzz.trimf(ventana_output.universe, [50, 75, 100])
ventana_output['CERRADA'] = fuzz.trapmf(ventana_output.universe, [75, 100, 100, 100])

# ==========================================
# 3. Inferencia Difusa
# ==========================================
# Base de Reglas
reglas = [
   # --- REGLAS REACTIVAS DE DÍA (Sin pronóstico) ---
    ctrl.Rule(hora_input['DIA'] & z_input['MUY_NEG'], ventana_output['ABIERTA']),
    ctrl.Rule(hora_input['DIA'] & z_input['NEG'], ventana_output['CASI_ABIERTA']),
    ctrl.Rule(hora_input['DIA'] & z_input['ZERO'], ventana_output['MITAD']),
    ctrl.Rule(hora_input['DIA'] & z_input['POS'], ventana_output['CASI_CERRADA']),
    ctrl.Rule(hora_input['DIA'] & z_input['MUY_POS'], ventana_output['CERRADA']),

    # --- REGLAS PROACTIVAS DE DÍA (Con pronóstico) ---
    ctrl.Rule(hora_input['DIA'] & z_input['ZERO'] & t_pred_input['ALTA'], ventana_output['CASI_CERRADA']),
    ctrl.Rule(hora_input['DIA'] & z_input['POS'] & t_pred_input['ALTA'], ventana_output['CERRADA']),
    ctrl.Rule(hora_input['DIA'] & z_input['POS'] & t_pred_input['BAJA'], ventana_output['CASI_CERRADA']),

    # --- REGLAS DE SEGURIDAD NOCTURNA ---
    ctrl.Rule(hora_input['NOCHE'] & z_input['ZERO'], ventana_output['CERRADA']),

    # --- REGLAS PROACTIVAS DE NOCHE (Amenaza de CALOR mañana) ---
    ctrl.Rule(hora_input['NOCHE'] & z_input['MUY_NEG'] & t_pred_input['ALTA'], ventana_output['ABIERTA']),
    ctrl.Rule(hora_input['NOCHE'] & z_input['NEG'] & t_pred_input['ALTA'], ventana_output['CASI_ABIERTA']),
    ctrl.Rule(hora_input['NOCHE'] & z_input['POS'] & t_pred_input['ALTA'], ventana_output['CERRADA']),
    ctrl.Rule(hora_input['NOCHE'] & z_input['MUY_POS'] & t_pred_input['ALTA'], ventana_output['CERRADA']),

    # --- REGLAS PROACTIVAS DE NOCHE (Amenaza de FRÍO mañana) ---
    ctrl.Rule(hora_input['NOCHE'] & z_input['MUY_NEG'] & t_pred_input['BAJA'], ventana_output['ABIERTA']),
    ctrl.Rule(hora_input['NOCHE'] & z_input['NEG'] & t_pred_input['BAJA'], ventana_output['CERRADA']),
    ctrl.Rule(hora_input['NOCHE'] & z_input['POS'] & t_pred_input['BAJA'], ventana_output['CERRADA']),
    ctrl.Rule(hora_input['NOCHE'] & z_input['MUY_POS'] & t_pred_input['BAJA'], ventana_output['CERRADA'])
]


controlador_ventana = ctrl.ControlSystem(reglas)
simulador = ctrl.ControlSystemSimulation(controlador_ventana)

# ==========================================
# 4. Desborrosificacion
# ==========================================
def simular_difuso(T_media, T_amplitud, metodo_desborrosificacion):
    # Asignamos el método de desborrosificación elegido para esta corrida
    ventana_output.defuzzify_method = metodo_desborrosificacion
    
    dt = 60.0  # Paso: 60 segundos
    t = np.arange(0, 24 * 3600, dt)
    n_steps = len(t)
    
    v0 = 25.0
    tau_abierta = (2.4 * 3600) / 5.0
    tau_cerrada = (24 * 3600) / 5.0
    
    # Clima senoidal
    ve = T_media + T_amplitud * np.sin(2 * np.pi * (t - 9 * 3600) / (24 * 3600))
    
    v = np.zeros(n_steps)
    v[0] = 20.0  # Temperatura inicial
    apertura_hist = np.zeros(n_steps)
    T_pronostico = 6 * 3600  # Miramos 6 horas hacia el futuro
    pasos_futuro = int(T_pronostico / dt)

    for i in range(n_steps - 1):
        z_actual = (v[i] - v0) * (ve[i] - v[i])
        hora_actual = (t[i] / 3600) % 24
        
        idx_futuro = min(i + pasos_futuro, n_steps - 1)
        temp_manana = ve[idx_futuro]

        simulador.input['Z'] = np.clip(z_actual, -100, 100)
        simulador.input['hora'] = hora_actual
        simulador.input['t_predicha'] = temp_manana
        simulador.compute()
        
        try:
            alfa = simulador.output['ventana']
        except KeyError:
            # ¡Atrapamos el bug en el acto!
            print(f"ZONA MUERTA DETECTADA en el paso {i}:")
            print(f"   Hora = {hora_actual:.2f}")
            print(f"   Z = {z_actual:.2f}")
            print(f"   T_Pred = {temp_manana:.2f}")
            print("   Ninguna regla difusa cubre esta combinación.")
            
            # Valor por defecto (Fallback) para que no crashee y termine de graficar
            alfa = 50.0
        
        apertura_hist[i] = alfa
        tau_actual = tau_abierta + (alfa / 100.0) * (tau_cerrada - tau_abierta)
        dv_dt = (ve[i] - v[i]) / tau_actual
        v[i+1] = v[i] + dv_dt * dt
        
    apertura_hist[-1] = apertura_hist[-2]

    # Cálculo del Índice de Desempeño J (8h a 20h)
    idx_8h = int((8 * 3600) / dt)
    idx_20h = int((20 * 3600) / dt)
    J_oficial = np.mean(v[idx_8h:idx_20h] - v0)
    J_absoluto = np.mean(np.abs(v[idx_8h:idx_20h] - v0))
    
    return t/3600, ve, v, apertura_hist, J_oficial, J_absoluto

# ==========================================
# 3. EJECUCIÓN: COMPARATIVA DE LOS 3 CASOS TP
# ==========================================
def main():
    # Los 3 casos obligatorios del Trabajo Práctico
    escenarios_tp = [
        ("Caso 1: Cruza el Confort (Oscila 16°C a 32°C)", 24.0, 8.0),
        ("Caso 2: Siempre por Encima (Oscila 28°C a 38°C)", 33.0, 5.0),
        ("Caso 3: Siempre por Debajo (Oscila 4°C a 20°C)", 12.0, 8.0)
    ]
    
    metodos = [
        ('centroid', 'Centroide (Continuo)'),
        ('mom', 'Máximo - MOM (Saltos Discretos)')
    ]
    
    print("Iniciando simulaciones comparativas...")
    
    for nombre_caso, t_med, t_amp in escenarios_tp:
        # Creamos una ventana por cada caso, con 2 filas y 2 columnas
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
        fig.suptitle(f'{nombre_caso}', fontsize=16, fontweight='bold')
        
        for col, (codigo_metodo, titulo_metodo) in enumerate(metodos):
            t_horas, ve, v, ventana, J_of, J_abs = simular_difuso(t_med, t_amp, codigo_metodo)
            
            # Gráficos de Temperatura (Fila 0)
            ax_temp = axs[0, col]
            ax_temp.plot(t_horas, ve, label='Temp. Exterior', color='orange', linestyle='--')
            ax_temp.plot(t_horas, v, label='Temp. Interior', color='blue', linewidth=2)
            ax_temp.axhline(25.0, color='green', linestyle=':', label='Confort (25°C)')
            ax_temp.axvspan(8, 20, color='gray', alpha=0.1)
            ax_temp.set_title(f'{titulo_metodo}\nJ_abs = {J_abs:.3f}')
            ax_temp.set_ylabel('Temperatura (°C)')
            ax_temp.grid(True)
            if col == 0: ax_temp.legend(loc='best')
            
            # Gráficos del Actuador (Fila 1)
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

    print("Simulaciones completadas. Mostrando ventanas...")
    plt.show()

if __name__ == '__main__':
    main()