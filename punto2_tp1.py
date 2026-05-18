import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import heapq

# =============================================================================
# DEFINICIÓN MATRICIAL DEL ENTORNO DE PLANTA
# =============================================================================
matriz_almacen = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
    [ 0,  0,  1,  2,  0,  0,  9, 10,  0,  0, 17, 18,  0 ],
    [ 0,  0,  3,  4,  0,  0, 11, 12,  0,  0, 19, 20,  0 ],
    [ 0,  0,  5,  6,  0,  0, 13, 14,  0,  0, 21, 22,  0 ],
    [ 0,  0,  7,  8,  0,  0, 15, 16,  0,  0, 23, 24,  0 ],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
    [ 0,  0, 25, 26,  0,  0, 33, 34,  0,  0, 41, 42,  0 ],
    [ 0,  0, 27, 28,  0,  0, 35, 36,  0,  0, 43, 44,  0 ],
    [ 0,  0, 29, 30,  0,  0, 37, 38,  0,  0, 45, 46,  0 ],
    [ 0,  0, 31, 32,  0,  0, 39, 40,  0,  0, 47, 48,  0 ],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ]
])

BUFFER_TIEMPO = 1     # Tolerancia inercial para colisiones de seguimiento
RANGO_SENSOR = 2      # Rango del escáner LiDAR estocástico

# =============================================================================
# INFRAESTRUCTURA MATEMÁTICA: NODOS Y HEURÍSTICA
# =============================================================================
class Node:
    def __init__(self, x, y, g=0, h=0, parent=None, time=0):
        self.x = x; self.y = y; self.time = time
        self.g = g; self.h = h; self.f = g + h      
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def heuristic(x1, y1, x2, y2):
    """
    Distancia de Manhattan Ponderada (Bounded Suboptimal A*).
    Aplica una inflación de 1.001 para forzar vectores rectilíneos.
    """
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx + dy) * 1.001

def buscar_objetivo_adyacente(estanteria):
    """ Restricción cinemática: El AGV solo accede ortogonalmente por los laterales """
    pos = np.where(matriz_almacen == estanteria)
    if len(pos[0]) == 0: return None
    r, c = pos[0][0], pos[1][0]
    
    if c - 1 >= 0 and matriz_almacen[r, c-1] == 0: return (r, c-1)
    if c + 1 < matriz_almacen.shape[1] and matriz_almacen[r, c+1] == 0: return (r, c+1)
    return None

def a_star_dynamic(grid, start, goal, time_start, reservation_table):
    open_set = []
    closed_set = set()
    start_node = Node(start[0], start[1], 0, heuristic(start[0], start[1], goal[0], goal[1]), None, time_start)
    heapq.heappush(open_set, start_node)

    while open_set:
        current = heapq.heappop(open_set)

        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y, current.time))
                current = current.parent
            return path[::-1]

        state = (current.x, current.y, current.time)
        if state in closed_set: continue
        closed_set.add(state)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            nx, ny = current.x + dx, current.y + dy
            ntime = current.time + 1

            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
                # Verificación contra tabla temporal
                if (nx, ny, ntime) in reservation_table: continue
                # Filtro matemático de Edge Conflict
                if (nx, ny, current.time) in reservation_table and (current.x, current.y, ntime) in reservation_table: continue

                # Disociación Energética (Tracción vs. Ralentí)
                costo_accion = 0.2 if (dx == 0 and dy == 0) else 1.0
                g_cost = current.g + costo_accion
                
                neighbor = Node(nx, ny, g_cost, heuristic(nx, ny, goal[0], goal[1]), current, ntime)
                heapq.heappush(open_set, neighbor)
    return []

# =============================================================================
# CONTROLADOR CINEMÁTICO, FSM Y TELEMETRÍA
# =============================================================================
def ejecutar_simulacion(nombre_caso, configuracion_agentes, obstaculos_ocultos):
    print(f"\n[+] {nombre_caso} | Calculando matriz espacial...")
    agentes = []
    for conf in configuracion_agentes:
        meta_estanteria = buscar_objetivo_adyacente(conf['estanteria_id'])
        agentes.append({
            'id': conf['id'],
            'inicio': conf['inicio'],
            'estanteria_id': conf['estanteria_id'], 
            'meta_actual': meta_estanteria,
            'estado': 'ida',
            'ruta': [],
            'terminado': False
        })
    
    tiempo_maximo = 250 
    tabla_reservas = set()
    historial_posiciones = {a['id']: [] for a in agentes}
    grid_dinamico = np.copy(matriz_almacen)
    obs_pendientes = list(obstaculos_ocultos)

    for t in range(tiempo_maximo):
        if all(a['terminado'] for a in agentes):
            print(f"    -> Lazo cerrado. Convergencia lograda en t={t}.")
            break

        replanificar_global = False
        
        # 1. Escaneo Ambiental
        for agente in agentes:
            if agente['terminado']: continue
            pos_actual = historial_posiciones[agente['id']][-1] if historial_posiciones[agente['id']] else agente['inicio']
            detectados = []
            for obs in obs_pendientes:
                if (abs(pos_actual[0] - obs[0]) + abs(pos_actual[1] - obs[1])) <= RANGO_SENSOR:
                    grid_dinamico[obs[0], obs[1]] = -1 
                    detectados.append(obs)
                    replanificar_global = True
            for obs in detectados: obs_pendientes.remove(obs)
                
        # 2. Interrupción Sensorial
        if replanificar_global:
            tabla_reservas.clear()
            for agente in agentes: 
                if not agente['terminado']: agente['ruta'] = []
                
        # 3. Máquina de Estados Finitos (FSM)
        for agente in agentes:
            if agente['terminado']:
                pos_actual = historial_posiciones[agente['id']][-1]
                historial_posiciones[agente['id']].append(pos_actual)
                for dt in range(1, 5): 
                    tabla_reservas.add((pos_actual[0], pos_actual[1], t + dt))
                continue

            pos_actual = historial_posiciones[agente['id']][-1] if historial_posiciones[agente['id']] else agente['inicio']
            
            # Conmutación IDA -> VUELTA con HARD-RESET Inercial
            if pos_actual == agente['meta_actual'] and agente['estado'] == 'ida' and len(agente['ruta']) == 0:
                agente['estado'] = 'vuelta'
                agente['meta_actual'] = agente['inicio'] 
                
                # Eliminación estricta de la sombra temporal
                tabla_reservas.clear()
                for a in agentes:
                    if not a['terminado']: a['ruta'] = []
            
            # Conmutación VUELTA -> REPOSO
            if pos_actual == agente['meta_actual'] and agente['estado'] == 'vuelta' and len(agente['ruta']) == 0:
                agente['terminado'] = True
                historial_posiciones[agente['id']].append(pos_actual)
                continue

            # Invocador del Optimizador
            if len(agente['ruta']) == 0:
                ruta_calculada = a_star_dynamic(grid_dinamico, pos_actual, agente['meta_actual'], t, tabla_reservas)
                if ruta_calculada:
                    agente['ruta'] = ruta_calculada[1:] 
                    for r in ruta_calculada:
                        for inflacion in range(BUFFER_TIEMPO + 1):
                            tabla_reservas.add((r[0], r[1], r[2] + inflacion))
            
            # Escribir la salida en el actuador
            if agente['ruta']:
                siguiente = agente['ruta'].pop(0)
                historial_posiciones[agente['id']].append((siguiente[0], siguiente[1]))
            else:
                historial_posiciones[agente['id']].append(pos_actual) 

    # 4. Compensación de Pre-Carga (2 segundos inactivos iniciales)
    FRAMES_ESPERA = 5
    for agente in agentes:
        if len(historial_posiciones[agente['id']]) > 0:
            pad = [historial_posiciones[agente['id']][0]] * FRAMES_ESPERA
            historial_posiciones[agente['id']] = pad + historial_posiciones[agente['id']]

    # 5. Generación de Telemetría Analítica
    nombre_archivo_base = nombre_caso.replace(' ', '_').lower()
    log_filename = f"telemetria_{nombre_archivo_base}.txt"
    with open(log_filename, 'w', encoding='utf-8') as file:
        file.write(f"=== REPORTE CINEMÁTICO: {nombre_caso} ===\n")
        file.write("Búsqueda Global Heurística - Secuencia Nodal (x, y)\n")
        file.write(f"Obstáculos descubiertos: {obstaculos_ocultos}\n\n")
        
        for agente in agentes:
            file.write(f"--- AGV ID {agente['id']} ---\n")
            file.write(f"Posición Origen: {agente['inicio']} | Tarea: Estantería {agente['estanteria_id']}\n")
            trayectoria_real = historial_posiciones[agente['id']][FRAMES_ESPERA:]
            t_str = " -> ".join([f"({r},{c})" for r, c in trayectoria_real])
            file.write(f"Integración Temporal (Ciclos consumidos: {len(trayectoria_real)}):\n{t_str}\n\n")

    # 6. Compilador de Renderizado Visual
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ['#FF0000', '#00FF00', '#FFA500', '#800080', '#00FFFF']
    
    def update(frame):
        ax.clear()
        ax.imshow(grid_dinamico == -1, cmap='Reds', alpha=0.4)   
        ax.imshow(matriz_almacen > 0, cmap='Blues', alpha=0.2)   
        
        # Etiquetado topológico
        for r in range(matriz_almacen.shape[0]):
            for c in range(matriz_almacen.shape[1]):
                if matriz_almacen[r, c] > 0: 
                    ax.text(c, r, str(matriz_almacen[r, c]), ha='center', va='center', fontsize=8, color='black')

        # Inyección de HMI: Orígenes y Metas
        for i, agente in enumerate(agentes):
            color_agente = colores[i % len(colores)]
            r_ini, c_ini = agente['inicio']
            ax.add_patch(patches.Rectangle((c_ini - 0.5, r_ini - 0.5), 1, 1, fill=True, color=color_agente, alpha=0.25))
            
            pos_est = np.where(matriz_almacen == agente['estanteria_id'])
            if len(pos_est[0]) > 0:
                r_est, c_est = pos_est[0][0], pos_est[1][0]
                ax.add_patch(patches.Rectangle((c_est - 0.5, r_est - 0.5), 1, 1, fill=True, color=color_agente, alpha=0.6, hatch='//'))

        # Ploteo de coordenadas dinámicas
        for i, agente in enumerate(agentes):
            if frame < len(historial_posiciones[agente['id']]):
                pos = historial_posiciones[agente['id']][frame]
                ax.plot(pos[1], pos[0], marker='s', markersize=12, markeredgecolor='black', markeredgewidth=1.5, color=colores[i % len(colores)], label=f'AGV {agente["id"]}')
        
        ax.set_title(f'{nombre_caso} | Reloj del Sistema: t={frame}')
        ax.set_xticks(np.arange(-.5, matriz_almacen.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, matriz_almacen.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.legend(loc='upper right', prop={'size': 8})

    max_frames = max(len(h) for h in historial_posiciones.values())
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=400, repeat=False)
    archivo_salida_gif = f"{nombre_archivo_base}.gif"
    ani.save(archivo_salida_gif, writer='pillow')
    print(f"    -> Artefactos exportados: {log_filename} y {archivo_salida_gif}")
    plt.close(fig)

# =============================================================================
# BANCO DE PRUEBAS (UNIT TESTING SUITE)
# =============================================================================
casos_de_prueba = [
    {
        "nombre": "Caso 1 - Evasion de Intercambio (Fisica Real)",
        "agentes": [
            {'id': 1, 'inicio': (0, 0), 'estanteria_id': 18},
            {'id': 2, 'inicio': (0, 12), 'estanteria_id': 1}
        ],
        "obstaculos": []
    },
    {
        "nombre": "Caso 2 - Colision Frontal",
        "agentes": [
            {'id': 1, 'inicio': (0, 5), 'estanteria_id': 47}, 
            {'id': 2, 'inicio': (5, 0), 'estanteria_id': 34}  
        ],
        "obstaculos": [] 
    },
    {
        "nombre": "Caso 3 - Replanificacion por Obstaculos",
        "agentes": [
            {'id': 1, 'inicio': (0, 0), 'estanteria_id': 48}
        ],
        "obstaculos": [(5, 12), (6, 12), (7, 12), (8, 12)] 
    },
    {
        "nombre": "Caso 4 - Cuello de Botella Operativo",
        "agentes": [
            {'id': 1, 'inicio': (0, 0), 'estanteria_id': 27}, 
            {'id': 2, 'inicio': (1, 0), 'estanteria_id': 29},
            {'id': 3, 'inicio': (2, 0), 'estanteria_id': 31}  
        ],
        "obstaculos": []
    },
    {
        "nombre": "Caso 5 - Encierro y Backtracking",
        "agentes": [
            {'id': 1, 'inicio': (10, 0), 'estanteria_id': 24}
        ],
        "obstaculos": [(5, 10), (5, 11), (5, 12), (4, 10), (3, 10)] 
    }
]

if __name__ == '__main__':
    print("INICIALIZANDO PLATAFORMA DE VALIDACIÓN MAPF...")
    for caso in casos_de_prueba:
        ejecutar_simulacion(caso["nombre"], caso["agentes"], caso["obstaculos"])
    print("\n[+] EJECUCIÓN DEL MODELO FINALIZADA. PROCEDA CON LA DEPURACIÓN GRÁFICA.")