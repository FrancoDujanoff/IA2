import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import heapq
import random

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

BUFFER_TIEMPO = 1     
RANGO_SENSOR = 2      

# =============================================================================
# INFRAESTRUCTURA MATEMÁTICA Y A*
# =============================================================================
class Node:
    def __init__(self, x, y, g=0, h=0, parent=None, time=0):
        self.x = x; self.y = y; self.time = time
        self.g = g; self.h = h; self.f = g + h      
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def heuristic(x1, y1, x2, y2):
    return (abs(x1 - x2) + abs(y1 - y2)) * 1.001

def buscar_objetivo_adyacente(estanteria):
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
                if (nx, ny, ntime) in reservation_table: continue
                if (nx, ny, current.time) in reservation_table and (current.x, current.y, ntime) in reservation_table: continue

                costo_accion = 0.2 if (dx == 0 and dy == 0) else 1.0
                neighbor = Node(nx, ny, current.g + costo_accion, heuristic(nx, ny, goal[0], goal[1]), current, ntime)
                heapq.heappush(open_set, neighbor)
    return []

def get_pos(agente, offset=0, historial=None):
    if offset == 0:
        return historial[agente['id']][-1] if historial[agente['id']] else agente['inicio']
    else:
        # Extraemos solo X e Y (evitamos la dimensión temporal para el V2V)
        return (agente['ruta'][0][0], agente['ruta'][0][1]) if len(agente['ruta']) > 0 else get_pos(agente, 0, historial)

# =============================================================================
# MOTOR DE SIMULACIÓN Y DESPACHADOR DE TAREAS (ROBUSTO)
# =============================================================================
def ejecutar_turno_completo(nombre_caso, configuracion_agentes, obstaculos_ocultos):
    print(f"\n[+] INICIANDO: {nombre_caso}")
    agentes = []
    for conf in configuracion_agentes:
        primera_meta = buscar_objetivo_adyacente(conf['tareas'][0])
        agentes.append({
            'id': conf['id'],
            'inicio': conf['inicio'],
            'tareas_pendientes': conf['tareas'], 
            'tarea_actual_id': conf['tareas'][0],
            'meta_actual': primera_meta,
            'estado': 'trabajando', 
            'ruta': [],
            'terminado': False
        })

    tiempo_maximo = 1500 
    tabla_reservas = set()
    historial_posiciones = {a['id']: [] for a in agentes}
    grid_dinamico = np.copy(matriz_almacen)
    obs_pendientes = list(obstaculos_ocultos)

    for t in range(tiempo_maximo):
        if all(a['terminado'] for a in agentes):
            print(f"\n[OK] Turno operativo finalizado en t={t} ciclos.")
            break

        replanificar_obstaculo = False
        replanificar_colision = False
        
        # 1. Sensores Estáticos
        for agente in agentes:
            if agente['terminado']: continue
            pos_actual = get_pos(agente, 0, historial_posiciones)
            detectados = []
            for obs in obs_pendientes:
                if (abs(pos_actual[0] - obs[0]) + abs(pos_actual[1] - obs[1])) <= RANGO_SENSOR:
                    grid_dinamico[obs[0], obs[1]] = -1 
                    detectados.append(obs)
                    replanificar_obstaculo = True
            for obs in detectados: obs_pendientes.remove(obs)
                
        # 2. Sensores V2V (Predicción Cinemática)
        for i in range(len(agentes)):
            if agentes[i]['terminado']: continue
            pos_i = get_pos(agentes[i], 0, historial_posiciones)
            next_i = get_pos(agentes[i], 1, historial_posiciones)

            for j in range(i+1, len(agentes)):
                if agentes[j]['terminado']: continue
                pos_j = get_pos(agentes[j], 0, historial_posiciones)
                next_j = get_pos(agentes[j], 1, historial_posiciones)

                if (abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])) <= RANGO_SENSOR + 1:
                    if next_i == next_j or (next_i == pos_j and next_j == pos_i):
                        replanificar_colision = True

        if replanificar_obstaculo or replanificar_colision:
            for a in agentes: 
                if not a['terminado']: a['ruta'] = [] 
                
        tabla_reservas.clear()
        for a in agentes:
            if a['terminado']:
                pos = get_pos(a, 0, historial_posiciones)
                for dt in range(1, 5): tabla_reservas.add((pos[0], pos[1], t + dt))

        # 3. Planificación Reactiva con CUERPOS SÓLIDOS TEMPORALES
        for agente in agentes:
            if agente['terminado']:
                historial_posiciones[agente['id']].append(get_pos(agente, 0, historial_posiciones))
                continue

            pos_actual = get_pos(agente, 0, historial_posiciones)
            
            # FSM (Cola de Tareas)
            if pos_actual == agente['meta_actual'] and len(agente['ruta']) == 0:
                if agente['estado'] == 'trabajando':
                    agente['tareas_pendientes'].pop(0) 
                    if len(agente['tareas_pendientes']) > 0:
                        sig_tarea = agente['tareas_pendientes'][0]
                        agente['tarea_actual_id'] = sig_tarea
                        agente['meta_actual'] = buscar_objetivo_adyacente(sig_tarea)
                    else:
                        agente['estado'] = 'regresando_base'
                        agente['meta_actual'] = agente['inicio']
                        agente['tarea_actual_id'] = "BASE"
                    for a in agentes:
                        if not a['terminado']: a['ruta'] = []
                
                elif agente['estado'] == 'regresando_base':
                    agente['terminado'] = True
                    historial_posiciones[agente['id']].append(pos_actual)
                    continue

            # Invocador del Planificador
            if len(agente['ruta']) == 0:
                
                # --- CAPA DE SEGURIDAD 1: PROYECCIÓN DE CUERPOS ---
                grid_temp = np.copy(grid_dinamico)
                if replanificar_colision:
                    for otro in agentes:
                        if otro['id'] != agente['id']:
                            p_otro = get_pos(otro, 0, historial_posiciones)
                            dist = abs(pos_actual[0] - p_otro[0]) + abs(pos_actual[1] - p_otro[1])
                            if dist <= RANGO_SENSOR + 1:
                                grid_temp[p_otro[0], p_otro[1]] = -1 # Levanta muro virtual

                # Cálculo de Rutas
                if replanificar_colision:
                    ruta = a_star_dynamic(grid_temp, pos_actual, agente['meta_actual'], t, tabla_reservas)
                    if ruta:
                        agente['ruta'] = ruta[1:] 
                        for r in ruta:
                            for inf in range(BUFFER_TIEMPO + 1): tabla_reservas.add((r[0], r[1], r[2] + inf))
                else:
                    tabla_ciega = set(tabla_reservas) 
                    ruta = a_star_dynamic(grid_temp, pos_actual, agente['meta_actual'], t, tabla_ciega)
                    if ruta:
                        agente['ruta'] = ruta[1:] 
            
            # --- CAPA DE SEGURIDAD 2: INTERBLOQUEO FÍSICO (BUG SOLUCIONADO) ---
            if agente['ruta']:
                # [CORRECCIÓN CRÍTICA]: Aislamos la coordenada espacial (X, Y) ignorando el Tiempo.
                siguiente_espacial = (agente['ruta'][0][0], agente['ruta'][0][1])
                
                bloqueado_fisicamente = False
                for otro in agentes:
                    if otro['id'] != agente['id']:
                        p_otro = get_pos(otro, 0, historial_posiciones)
                        # Comparamos X, Y contra X, Y. Ahora el sensor es infalible.
                        if siguiente_espacial == p_otro:
                            bloqueado_fisicamente = True
                            break
                
                if bloqueado_fisicamente:
                    # El motor se niega a girar y frena en seco
                    historial_posiciones[agente['id']].append(pos_actual)
                    agente['ruta'] = [] 
                else:
                    # Movimiento libre
                    siguiente = agente['ruta'].pop(0)
                    historial_posiciones[agente['id']].append((siguiente[0], siguiente[1]))
            else:
                historial_posiciones[agente['id']].append(pos_actual) 

    # RENDERIZADO VISUAL
    print("    Compilando artefacto gráfico. Por favor, espere...")
    fig, ax = plt.subplots(figsize=(12, 7))
    colores = ['#FF0000', '#00FF00', '#FFA500'] 
    
    def update(frame):
        ax.clear()
        ax.imshow(grid_dinamico == -1, cmap='Reds', alpha=0.4)   
        ax.imshow(matriz_almacen > 0, cmap='Blues', alpha=0.2)   
        
        for r in range(matriz_almacen.shape[0]):
            for c in range(matriz_almacen.shape[1]):
                if matriz_almacen[r, c] > 0: ax.text(c, r, str(matriz_almacen[r, c]), ha='center', va='center', fontsize=8)

        for i, agente in enumerate(agentes):
            col = colores[i % len(colores)]
            r_ini, c_ini = agente['inicio']
            ax.add_patch(patches.Rectangle((c_ini - 0.5, r_ini - 0.5), 1, 1, fill=True, color=col, alpha=0.3))
            
            if not agente['estado'] == 'regresando_base':
                pos_est = np.where(matriz_almacen == agente['tarea_actual_id'])
                if len(pos_est[0]) > 0:
                    ax.add_patch(patches.Rectangle((pos_est[1][0] - 0.5, pos_est[0][0] - 0.5), 1, 1, fill=True, color=col, alpha=0.7, hatch='//'))

            if frame < len(historial_posiciones[agente['id']]):
                pos = historial_posiciones[agente['id']][frame]
                ax.plot(pos[1], pos[0], marker='s', markersize=14, markeredgecolor='black', color=col, label=f"AGV {agente['id']} (Quedan: {len(agente['tareas_pendientes'])})")
        
        ax.set_title(f'OPERACIÓN MAPF ROBUSTA (10 Tareas) | Ciclo: t={frame}')
        ax.set_xticks(np.arange(-.5, 13, 1), minor=True); ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=1)
        ax.legend(loc='upper right', prop={'size': 9})

    max_frames = max(len(h) for h in historial_posiciones.values())
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=400, repeat=False)
    ani.save(f"simulacion_turno_robusto.gif", writer='pillow')
    print(f"    -> Archivo 'simulacion_turno_robusto.gif' exportado exitosamente.")
    plt.close(fig)

if __name__ == '__main__':
    todas_las_estanterias = [i for i in range(1, 49)]
    
    configuracion_operativa = [
        {'id': 1, 'inicio': (0, 0),  'tareas': random.sample(todas_las_estanterias, 10)},
        {'id': 2, 'inicio': (10, 0), 'tareas': random.sample(todas_las_estanterias, 10)},
        {'id': 3, 'inicio': (5, 12), 'tareas': random.sample(todas_las_estanterias, 10)}
    ]
    
    obstaculos_sorpresa = [(5, 5), (5, 6)] 
    
    ejecutar_turno_completo("Prueba de Estrés Industrial", configuracion_operativa, obstaculos_sorpresa)