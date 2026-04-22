import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ==========================================
# 1. DEFINICIÓN MANUAL DEL SISTEMA DIFUSO
# ==========================================
# Universos
x_z = np.arange(-100, 101, 1)
x_hora = np.arange(0, 25, 0.1)
x_tpred = np.arange(0, 51, 1)
x_out = np.arange(0, 101, 1)

# Conjuntos Z
z_muy_neg = fuzz.trapmf(x_z, [-100, -100, -10, -5])
z_neg = fuzz.trimf(x_z, [-10, -5, 0])
z_zero = fuzz.trimf(x_z, [-5, 0, 5])
z_pos = fuzz.trimf(x_z, [0, 5, 10])
z_muy_pos = fuzz.trapmf(x_z, [5, 10, 100, 100])

# Conjuntos HORA
hora_noche = fuzz.trapmf(x_hora, [0, 0, 7, 8]) + fuzz.trapmf(x_hora, [19, 20, 24, 24])
hora_noche = np.clip(hora_noche, 0, 1)
hora_dia = fuzz.trapmf(x_hora, [7, 8, 19, 20])

# Conjutos T Predicha
t_baja = fuzz.trapmf(x_tpred, [0, 0, 22, 28])
t_alta = fuzz.trapmf(x_tpred, [22, 28, 50, 50])

# Conjuntos SALIDA (VENTANA)
out_abierta = fuzz.trimf(x_out, [0, 0, 25])
out_casi_abierta = fuzz.trimf(x_out, [0, 25, 50])
out_mitad = fuzz.trimf(x_out, [25, 50, 75])
out_casi_cerrada = fuzz.trimf(x_out, [50, 75, 100])
out_cerrada = fuzz.trapmf(x_out, [75, 100, 100, 100])

# ==========================================
# 2. SIMULACIÓN DE LOS 3 CASOS (Pre-cálculo)
# ==========================================
dt = 60.0
t_horas = np.arange(0, 24 * 3600, dt) / 3600
n_steps = len(t_horas)

def pre_simular(T_media, T_amplitud):
    # --- VARIABLES INICIALES RECUPERADAS ---
    v0 = 25.0
    tau_abierta = 1728.0
    tau_cerrada = 17280.0
    
    # Onda de clima senoidal
    ve = T_media + T_amplitud * np.sin(2 * np.pi * (t_horas - 9) / 24)
    v = np.zeros(n_steps)
    v[0] = 20.0
    z_hist = np.zeros(n_steps)
    
    # Máquina del tiempo del pronóstico
    T_pronostico = 6 * 3600  # Miramos 6 horas al futuro
    pasos_futuro = int(T_pronostico / dt)

    for i in range(n_steps - 1):
        z_actual = (v[i] - v0) * (ve[i] - v[i])
        z_hist[i] = np.clip(z_actual, -100, 100)
        
        # 1. Mirar al futuro
        idx_futuro = min(i + pasos_futuro, n_steps - 1)
        temp_manana = ve[idx_futuro]
        
        # 2. Fuzzificación rápida
        mu_dia = fuzz.interp_membership(x_hora, hora_dia, t_horas[i])
        mu_noche = fuzz.interp_membership(x_hora, hora_noche, t_horas[i])
        
        mt_baja = fuzz.interp_membership(x_tpred, t_baja, temp_manana)
        mt_alta = fuzz.interp_membership(x_tpred, t_alta, temp_manana)
        
        mz_mn = fuzz.interp_membership(x_z, z_muy_neg, z_hist[i])
        mz_n = fuzz.interp_membership(x_z, z_neg, z_hist[i])
        mz_z = fuzz.interp_membership(x_z, z_zero, z_hist[i])
        mz_p = fuzz.interp_membership(x_z, z_pos, z_hist[i])
        mz_mp = fuzz.interp_membership(x_z, z_muy_pos, z_hist[i])
        
        # 3. Las 5 Reglas Maestras Condensadas (AND=min, OR=max)
        act_abierta = max(mz_mn, min(mu_noche, mt_baja, mz_mp))
        
        act_c_abierta = max(min(mu_dia, mz_n), 
                            min(mu_noche, mt_alta, mz_n), 
                            min(mu_noche, mt_alta, mz_mp), 
                            min(mu_noche, mt_baja, mz_p))
                            
        act_mitad = max(min(mu_dia, mz_z), 
                        min(mu_dia, mz_p, mt_baja))
                        
        act_c_cerrada = max(min(mu_dia, mz_p), 
                            min(mu_dia, mz_z, mt_alta))
                            
        act_cerrada = max(min(mu_dia, mz_mp), 
                          min(mz_p, mt_alta), 
                          min(mu_noche, mt_baja, mz_n), 
                          min(mu_noche, mz_z))
        
        # 4. Agregación y Desborrosificación
        area = np.fmax(np.fmin(act_abierta, out_abierta),
               np.fmax(np.fmin(act_c_abierta, out_casi_abierta),
               np.fmax(np.fmin(act_mitad, out_mitad),
               np.fmax(np.fmin(act_c_cerrada, out_casi_cerrada),
                       np.fmin(act_cerrada, out_cerrada)))))
        
        alfa = fuzz.defuzz(x_out, area, 'centroid') if np.sum(area) > 0 else 50.0
        
        # 5. Física Térmica
        tau_actual = tau_abierta + (alfa / 100.0) * (tau_cerrada - tau_abierta)
        v[i+1] = v[i] + ((ve[i] - v[i]) / tau_actual) * dt
        
    # Guardar el último valor para alinear arreglos
    z_hist[-1] = z_hist[-2]
    return ve, v, z_hist

print("Pre-calculando las ecuaciones diferenciales de los 3 casos...")
datos_casos = {
    "Caso 1\nCruza Confort": pre_simular(24.0, 8.0),
    "Caso 2\nSiempre > 25": pre_simular(33.0, 5.0),
    "Caso 3\nSiempre < 25": pre_simular(12.0, 8.0)
}
# ==========================================
# 3. INTERFAZ GRÁFICA INTERACTIVA (Rediseñada)
# ==========================================
fig = plt.figure(figsize=(15, 9), dpi=100)
fig.canvas.manager.set_window_title('Simulador Analítico - Lógica Difusa Proactiva')

# --- NUEVA DISTRIBUCIÓN DE PANELES ---
# Paneles principales (Derecha)
ax_temp = plt.axes([0.25, 0.65, 0.72, 0.25])
ax_fuzzy = plt.axes([0.25, 0.20, 0.72, 0.35])
ax_slider = plt.axes([0.25, 0.08, 0.72, 0.03], facecolor='lightgray')

# Paneles de control y visualización de entradas (Izquierda)
ax_radio = plt.axes([0.02, 0.75, 0.18, 0.15], facecolor='lightgray')
ax_z = plt.axes([0.02, 0.52, 0.18, 0.15])      # Panel para Z
ax_hora = plt.axes([0.02, 0.29, 0.18, 0.15])   # Panel para Hora
ax_tpred = plt.axes([0.02, 0.06, 0.18, 0.15])  # Panel para T_Predicha

# Controles UI
radio = RadioButtons(ax_radio, list(datos_casos.keys()))
slider_tiempo = Slider(ax_slider, 'Hora del Día', 0.0, 23.99, valinit=12.0, valstep=0.1)

line_ext, = ax_temp.plot(t_horas, datos_casos["Caso 1\nCruza Confort"][0], '--', color='orange', label='T. Exterior')
line_int, = ax_temp.plot(t_horas, datos_casos["Caso 1\nCruza Confort"][1], '-', color='blue', linewidth=2, label='T. Interior')
ax_temp.axhline(25, color='green', linestyle=':', label='Confort (25°C)')
vline_t = ax_temp.axvline(12.0, color='black', linewidth=2)
ax_temp.set_ylabel('Temp (°C)')
ax_temp.legend(loc='upper right')
ax_temp.grid(True)
ax_temp.set_title('Termodinámica de la Habitación', fontweight='bold')

def update(val):
    caso_actual = radio.value_selected
    hora_actual = slider_tiempo.val
    idx = int(hora_actual * 60)
    
    ve, v, z_hist = datos_casos[caso_actual]
    z_actual = z_hist[idx]
    
    # 1. Actualizar Gráfico de Temperaturas
    line_ext.set_ydata(ve)
    line_int.set_ydata(v)
    vline_t.set_xdata([hora_actual, hora_actual])
    ax_temp.relim()
    ax_temp.autoscale_view()
    
    # 2. Inferencia Difusa Matemática
    pasos_futuro = int((6 * 3600) / dt)
    idx_futuro = min(idx + pasos_futuro, n_steps - 1)
    temp_manana = ve[idx_futuro]

    mu_dia = fuzz.interp_membership(x_hora, hora_dia, hora_actual)
    mu_noche = fuzz.interp_membership(x_hora, hora_noche, hora_actual)
    
    mt_baja = fuzz.interp_membership(x_tpred, t_baja, temp_manana)
    mt_alta = fuzz.interp_membership(x_tpred, t_alta, temp_manana)
    
    mz_mn = fuzz.interp_membership(x_z, z_muy_neg, z_actual)
    mz_n = fuzz.interp_membership(x_z, z_neg, z_actual)
    mz_z = fuzz.interp_membership(x_z, z_zero, z_actual)
    mz_p = fuzz.interp_membership(x_z, z_pos, z_actual)
    mz_mp = fuzz.interp_membership(x_z, z_muy_pos, z_actual)
    
    # Evaluar Reglas Maestras
    act_abierta = max(mz_mn, min(mu_noche, mt_baja, mz_mp))
    act_c_abierta = max(min(mu_dia, mz_n), min(mu_noche, mt_alta, mz_n), min(mu_noche, mt_alta, mz_mp), min(mu_noche, mt_baja, mz_p))
    act_mitad = max(min(mu_dia, mz_z), min(mu_dia, mz_p, mt_baja))
    act_c_cerrada = max(min(mu_dia, mz_p), min(mu_dia, mz_z, mt_alta))
    act_cerrada = max(min(mu_dia, mz_mp), min(mz_p, mt_alta), min(mu_noche, mt_baja, mz_n), min(mu_noche, mz_z))

    # Recorte
    area = np.fmax(np.fmin(act_abierta, out_abierta),
           np.fmax(np.fmin(act_c_abierta, out_casi_abierta),
           np.fmax(np.fmin(act_mitad, out_mitad),
           np.fmax(np.fmin(act_c_cerrada, out_casi_cerrada),
                   np.fmin(act_cerrada, out_cerrada)))))
    
    cog = fuzz.defuzz(x_out, area, 'centroid') if np.sum(area) > 0 else 50.0
    mom = fuzz.defuzz(x_out, area, 'mom') if np.sum(area) > 0 else 50.0
    
    # 3. Redibujar el panel difuso PRINCIPAL (Salida)
    ax_fuzzy.clear()    
    str_pred = f"T_Pred: {temp_manana:.1f}°C "
    str_pred += f"(BAJA: {mt_baja*100:.0f}% | ALTA: {mt_alta*100:.0f}%)"
    
    ax_fuzzy.set_title(f'Desborrosificación a las {hora_actual:.1f} hs | Z = {z_actual:.1f} | {str_pred}', fontweight='bold')
    
    ax_fuzzy.plot(x_out, out_abierta, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_casi_abierta, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_mitad, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_casi_cerrada, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_cerrada, 'gray', linestyle='--', alpha=0.5)
    
    ax_fuzzy.fill_between(x_out, 0, area, color='slateblue', alpha=0.5)
    ax_fuzzy.plot(x_out, area, color='midnightblue', linewidth=2)
    
    ax_fuzzy.axvline(cog, color='red', linewidth=3, label=f'Centroide: {cog:.1f}%')
    ax_fuzzy.axvline(mom, color='orange', linewidth=2, linestyle='-.', label=f'Máximo: {mom:.1f}%')
    
    ax_fuzzy.set_xticks([0, 25, 50, 75, 100])
    ax_fuzzy.set_xticklabels(['ABIERTA', 'CASI ABIERTA', 'MITAD', 'CASI CERRADA', 'CERRADA'])
    ax_fuzzy.set_ylim(0, 1.1)
    ax_fuzzy.legend(loc='upper right')
    ax_fuzzy.grid(True, alpha=0.3)

    # ==================================================
    # 4. REDIBUJAR LOS PANELES DE ENTRADA (Con Intersecciones Dinámicas)
    # ==================================================
    # Visualizar variable Z
    ax_z.clear()
    ax_z.set_title(f'Entrada Z: {z_actual:.1f}', fontsize=10, fontweight='bold', color='darkblue')
    ax_z.plot(x_z, z_muy_neg, color='tab:blue', label='M.NEG')
    ax_z.plot(x_z, z_neg, color='tab:orange', label='NEG')
    ax_z.plot(x_z, z_zero, color='tab:green', label='ZERO')
    ax_z.plot(x_z, z_pos, color='tab:red', label='POS')
    ax_z.plot(x_z, z_muy_pos, color='tab:purple', label='M.POS')
    ax_z.axvline(z_actual, color='black', linewidth=1.5)
    
    # --- MAGIA VISUAL: Intersecciones de Z ---
    for mu, col in [(mz_mn, 'tab:blue'), (mz_n, 'tab:orange'), (mz_z, 'tab:green'), (mz_p, 'tab:red'), (mz_mp, 'tab:purple')]:
        if mu > 0: # Solo mostramos el texto si la regla está activa
            ax_z.plot(z_actual, mu, 'ko', markersize=5) # Dibuja el punto de choque
            ax_z.hlines(mu, xmin=-20, xmax=z_actual, colors=col, linestyles='--', alpha=0.8) # Línea punteada
            ax_z.text(-19.5, mu + 0.05, f'{mu:.2f}', color=col, fontweight='bold', fontsize=9) # Texto flotante

    ax_z.set_xlim(-20, 20)
    ax_z.set_ylim(0, 1.25) # Un poco más alto para que el texto no se corte
    ax_z.set_yticks([0, 0.5, 1.0])
    ax_z.set_ylabel('Grado (μ)', fontsize=8)
    ax_z.grid(True, alpha=0.3)

    # Visualizar variable Hora
    ax_hora.clear()
    ax_hora.set_title(f'Entrada Hora: {hora_actual:.1f} hs', fontsize=10, fontweight='bold', color='darkblue')
    ax_hora.plot(x_hora, hora_noche, color='navy', label='NOCHE')
    ax_hora.plot(x_hora, hora_dia, color='orange', label='DIA')
    ax_hora.fill_between(x_hora, 0, hora_noche, color='navy', alpha=0.1)
    ax_hora.fill_between(x_hora, 0, hora_dia, color='orange', alpha=0.1)
    ax_hora.axvline(hora_actual, color='black', linewidth=1.5)

    # --- MAGIA VISUAL: Intersecciones de Hora ---
    for mu, col in [(mu_noche, 'navy'), (mu_dia, 'orange')]:
        if mu > 0:
            ax_hora.plot(hora_actual, mu, 'ko', markersize=5)
            ax_hora.hlines(mu, xmin=0, xmax=hora_actual, colors=col, linestyles='--', alpha=0.8)
            ax_hora.text(0.5, mu + 0.05, f'{mu:.2f}', color=col, fontweight='bold', fontsize=9)

    ax_hora.set_xlim(0, 24)
    ax_hora.set_xticks([0, 8, 16, 24])
    ax_hora.set_ylim(0, 1.25)
    ax_hora.set_yticks([0, 0.5, 1.0])
    ax_hora.set_ylabel('Grado (μ)', fontsize=8)
    ax_hora.grid(True, alpha=0.3)

    # Visualizar variable T_Predicha
    ax_tpred.clear()
    ax_tpred.set_title(f'Entrada T_Pred: {temp_manana:.1f}°C', fontsize=10, fontweight='bold', color='darkblue')
    ax_tpred.plot(x_tpred, t_baja, color='cyan', label='BAJA')
    ax_tpred.plot(x_tpred, t_alta, color='red', label='ALTA')
    ax_tpred.fill_between(x_tpred, 0, t_baja, color='cyan', alpha=0.1)
    ax_tpred.fill_between(x_tpred, 0, t_alta, color='red', alpha=0.1)
    ax_tpred.axvline(temp_manana, color='black', linewidth=1.5)

    # --- MAGIA VISUAL: Intersecciones de T_Pred ---
    for mu, col in [(mt_baja, 'cyan'), (mt_alta, 'red')]:
        if mu > 0:
            ax_tpred.plot(temp_manana, mu, 'ko', markersize=5)
            ax_tpred.hlines(mu, xmin=10, xmax=temp_manana, colors=col, linestyles='--', alpha=0.8)
            text_col = 'darkcyan' if col == 'cyan' else col # Oscurecemos un poco el cyan para que se lea
            ax_tpred.text(10.5, mu + 0.05, f'{mu:.2f}', color=text_col, fontweight='bold', fontsize=9)

    ax_tpred.set_xlim(10, 40)
    ax_tpred.set_ylim(0, 1.25)
    ax_tpred.set_yticks([0, 0.5, 1.0])
    ax_tpred.set_ylabel('Grado (μ)', fontsize=8)
    ax_tpred.grid(True, alpha=0.3)

    fig.canvas.draw_idle()

# Conectar eventos
slider_tiempo.on_changed(update)
radio.on_clicked(update)

# Forzar primer renderizado
update(12.0)
plt.show()