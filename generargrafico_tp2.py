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
    v0 = 25.0
    tau_abierta = 1728.0
    tau_cerrada = 17280.0
    ve = T_media + T_amplitud * np.sin(2 * np.pi * (t_horas - 9) / 24)
    v = np.zeros(n_steps)
    v[0] = 20.0
    z_hist = np.zeros(n_steps)
    
    for i in range(n_steps - 1):
        z_actual = (v[i] - v0) * (ve[i] - v[i])
        z_hist[i] = np.clip(z_actual, -100, 100)
        
        # Inferencia rápida para avanzar el estado
        mu_dia = fuzz.interp_membership(x_hora, hora_dia, t_horas[i])
        mu_noche = fuzz.interp_membership(x_hora, hora_noche, t_horas[i])
        
        mz_mn = fuzz.interp_membership(x_z, z_muy_neg, z_hist[i])
        mz_n = fuzz.interp_membership(x_z, z_neg, z_hist[i])
        mz_z = fuzz.interp_membership(x_z, z_zero, z_hist[i])
        mz_p = fuzz.interp_membership(x_z, z_pos, z_hist[i])
        mz_mp = fuzz.interp_membership(x_z, z_muy_pos, z_hist[i])
        
        act_abierta = min(mu_dia, mz_mn)
        act_c_abierta = min(mu_dia, mz_n)
        act_mitad = min(mu_dia, mz_z)
        act_c_cerrada = min(mu_dia, mz_p)
        act_cerrada = max(min(mu_dia, mz_mp), mu_noche) # Regla de noche incluida
        
        area = np.fmax(np.fmin(act_abierta, out_abierta),
               np.fmax(np.fmin(act_c_abierta, out_casi_abierta),
               np.fmax(np.fmin(act_mitad, out_mitad),
               np.fmax(np.fmin(act_c_cerrada, out_casi_cerrada),
                       np.fmin(act_cerrada, out_cerrada)))))
        
        alfa = fuzz.defuzz(x_out, area, 'centroid') if np.sum(area) > 0 else 50.0
        
        tau_actual = tau_abierta + (alfa / 100.0) * (tau_cerrada - tau_abierta)
        v[i+1] = v[i] + ((ve[i] - v[i]) / tau_actual) * dt
        
    z_hist[-1] = z_hist[-2]
    return ve, v, z_hist

print("Pre-calculando las ecuaciones diferenciales de los 3 casos...")
datos_casos = {
    "Caso 1\nCruza Confort": pre_simular(24.0, 8.0),
    "Caso 2\nSiempre > 25": pre_simular(33.0, 5.0),
    "Caso 3\nSiempre < 25": pre_simular(12.0, 8.0)
}

# ==========================================
# 3. INTERFAZ GRÁFICA INTERACTIVA
# ==========================================
fig = plt.figure(figsize=(14, 8), dpi=100)
fig.canvas.manager.set_window_title('Simulador Analítico - Lógica Difusa')

# Distribución de Subplots
ax_temp = plt.axes([0.25, 0.65, 0.7, 0.25])
ax_fuzzy = plt.axes([0.25, 0.15, 0.7, 0.4])
ax_radio = plt.axes([0.02, 0.5, 0.15, 0.25], facecolor='lightgray')
ax_slider = plt.axes([0.25, 0.05, 0.7, 0.03], facecolor='lightgray')

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
    idx = int(hora_actual * 60) # 1 minuto por paso
    
    ve, v, z_hist = datos_casos[caso_actual]
    z_actual = z_hist[idx]
    
    # 1. Actualizar Gráfico de Temperaturas
    line_ext.set_ydata(ve)
    line_int.set_ydata(v)
    vline_t.set_xdata([hora_actual, hora_actual])
    ax_temp.relim()
    ax_temp.autoscale_view()
    
    # 2. Inferencia Difusa Matemática (Instante actual)
    mu_dia = fuzz.interp_membership(x_hora, hora_dia, hora_actual)
    mu_noche = fuzz.interp_membership(x_hora, hora_noche, hora_actual)
    
    act_abierta = min(mu_dia, fuzz.interp_membership(x_z, z_muy_neg, z_actual))
    act_c_abierta = min(mu_dia, fuzz.interp_membership(x_z, z_neg, z_actual))
    act_mitad = min(mu_dia, fuzz.interp_membership(x_z, z_zero, z_actual))
    act_c_cerrada = min(mu_dia, fuzz.interp_membership(x_z, z_pos, z_actual))
    act_cerrada = max(min(mu_dia, fuzz.interp_membership(x_z, z_muy_pos, z_actual)), mu_noche)
    
    # Recorte
    area = np.fmax(np.fmin(act_abierta, out_abierta),
           np.fmax(np.fmin(act_c_abierta, out_casi_abierta),
           np.fmax(np.fmin(act_mitad, out_mitad),
           np.fmax(np.fmin(act_c_cerrada, out_casi_cerrada),
                   np.fmin(act_cerrada, out_cerrada)))))
    
    cog = fuzz.defuzz(x_out, area, 'centroid') if np.sum(area) > 0 else 50.0
    mom = fuzz.defuzz(x_out, area, 'mom') if np.sum(area) > 0 else 50.0
    
    # 3. Redibujar el panel difuso
    ax_fuzzy.clear()
    ax_fuzzy.set_title(f'Desborrosificación a las {hora_actual:.1f} hs | Entrada Z = {z_actual:.2f}', fontweight='bold')
    
    # Triángulos base
    ax_fuzzy.plot(x_out, out_abierta, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_casi_abierta, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_mitad, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_casi_cerrada, 'gray', linestyle='--', alpha=0.5)
    ax_fuzzy.plot(x_out, out_cerrada, 'gray', linestyle='--', alpha=0.5)
    
    # Área agregada
    ax_fuzzy.fill_between(x_out, 0, area, color='slateblue', alpha=0.5)
    ax_fuzzy.plot(x_out, area, color='midnightblue', linewidth=2)
    
    # Líneas de salida
    ax_fuzzy.axvline(cog, color='red', linewidth=3, label=f'Centroide: {cog:.1f}%')
    ax_fuzzy.axvline(mom, color='orange', linewidth=2, linestyle='-.', label=f'Máximo: {mom:.1f}%')
    
    ax_fuzzy.set_xticks([0, 25, 50, 75, 100])
    ax_fuzzy.set_xticklabels(['ABIERTA', 'CASI ABIERTA', 'MITAD', 'CASI CERRADA', 'CERRADA'])
    ax_fuzzy.set_ylabel('Grado de Membresía')
    ax_fuzzy.set_ylim(0, 1.1)
    ax_fuzzy.legend(loc='upper right')
    ax_fuzzy.grid(True, alpha=0.3)
    
    fig.canvas.draw_idle()

# Conectar eventos
slider_tiempo.on_changed(update)
radio.on_clicked(update)

# Forzar primer renderizado
update(12.0)
plt.show()