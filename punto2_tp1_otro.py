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

BUFFER_TIEMPO = 1     
RANGO_SENSOR = 2      

# =============================================================================
# CLASES Y LÓGICA HEURÍSTICA A* PONDERADA
# =============================================================================
class Node:
    def __init__(self, x, y, g=0, h=0, parent=None, time=0):
        self.x = x; self.y = y; self.time = time
        self.g = g; self.h = h; self.f = g + h      
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def heuristic(x1, y1, x2, y2):
    dx = abs(x1 - x2); dy = abs(y1 - y2)
    return (dx + dy) * 1.001

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
        return (agente['ruta'][0][0], agente['ruta'][0][1]) if len(agente['ruta']) > 0 else get_pos(agente, 0, historial)

# =============================================================================
# MOTOR DE SIMULACIÓN Y CONTROLADOR DESCENTRALIZADO
# =============================================================================
def ejecutar_simulacion(nombre_caso, configuracion_agentes, obstaculos_ocultos):
    print(f"\n[+] {nombre_caso} | Entorno Descentralizado Inicializado...")
    agentes = []
    for conf in configuracion_agentes:
        agentes.append({
            'id': conf['id'],
            'inicio': conf['inicio'],
            'estanteria_id': conf['estanteria_id'], 
            'meta_actual': buscar_objetivo_adyacente(conf['estanteria_id']),
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

        replanificar_obstaculo = False
        replanificar_colision = False
        
        #Escanea obstaculos cercanos
        for agente in agentes:
            if agente['terminado']: continue
            #Obtiene la posicion del montacargas
            pos_actual = get_pos(agente, 0, historial_posiciones)
            detectados = []
            for obs in obs_pendientes:
                #Si la caja esta a 2 metros o menos
                if (abs(pos_actual[0] - obs[0]) + abs(pos_actual[1] - obs[1])) <= RANGO_SENSOR:
                    #El sensor ve la caja y actualiza el mapa 
                    grid_dinamico[obs[0], obs[1]] = -1 
                    detectados.append(obs)
                    #Dispara la alarma de que el mapa cambio
                    replanificar_obstaculo = True
            for obs in detectados: obs_pendientes.remove(obs)
                
        for i in range(len(agentes)):
            if agentes[i]['terminado']: continue
            pos_i = get_pos(agentes[i], 0, historial_posiciones) #Posicion actual del agente i
            next_i = get_pos(agentes[i], 1, historial_posiciones) #Posicion futura del agente i

            for j in range(i+1, len(agentes)):
                if agentes[j]['terminado']: continue
                pos_j = get_pos(agentes[j], 0, historial_posiciones) #Posicion actual del agente j
                next_j = get_pos(agentes[j], 1, historial_posiciones) #Posicion futura del agente j

                #Si estan fisicamente cerca, verifica si van a chocar
                if (abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])) <= RANGO_SENSOR:
                    if next_i == next_j or (next_i == pos_j and next_j == pos_i):
                        replanificar_colision = True #Dispara la alarma de choque

        if replanificar_obstaculo or replanificar_colision:
            for a in agentes: 
                if not a['terminado']: a['ruta'] = [] 
                
        tabla_reservas.clear()
        for a in agentes:
            if a['terminado']:
                pos = get_pos(a, 0, historial_posiciones)
                for dt in range(1, 5): tabla_reservas.add((pos[0], pos[1], t + dt))

        for agente in agentes:
            if agente['terminado']:
                historial_posiciones[agente['id']].append(get_pos(agente, 0, historial_posiciones))
                continue

            pos_actual = get_pos(agente, 0, historial_posiciones)
            
            if pos_actual == agente['meta_actual'] and agente['estado'] == 'ida' and len(agente['ruta']) == 0:
                agente['estado'] = 'vuelta'
                agente['meta_actual'] = agente['inicio'] 
                for a in agentes:
                    if not a['terminado']: a['ruta'] = [] 
            
            if pos_actual == agente['meta_actual'] and agente['estado'] == 'vuelta' and len(agente['ruta']) == 0:
                agente['terminado'] = True
                historial_posiciones[agente['id']].append(pos_actual)
                continue

            #Inicia el Planificador
            if len(agente['ruta']) == 0:
                
                if replanificar_colision: #Verifica si no hay alerta de choque, sino planifica normal
                    ruta = a_star_dynamic(grid_dinamico, pos_actual, agente['meta_actual'], t, tabla_reservas)
                    
                    if ruta:
                        agente['ruta'] = ruta[1:] 
                        #Como hay peligro de choque, el montacargas de mayor prioridad escribe su futuro en la tabla global para que el otro
                        #se vea forzado a esquivarlo o ceder el paso
                        for r in ruta:
                            for inf in range(BUFFER_TIEMPO + 1): tabla_reservas.add((r[0], r[1], r[2] + inf))
                else:
                    #Si no hay riesgo de colision, asume que esta solo y planifica
                    tabla_ciega = set(tabla_reservas) 
                    ruta = a_star_dynamic(grid_dinamico, pos_actual, agente['meta_actual'], t, tabla_ciega)
                    if ruta:
                        agente['ruta'] = ruta[1:] 
            
            if agente['ruta']:
                siguiente = agente['ruta'].pop(0)
                historial_posiciones[agente['id']].append((siguiente[0], siguiente[1]))
            else:
                historial_posiciones[agente['id']].append(pos_actual) 

    FRAMES_ESPERA = 5
    for agente in agentes:
        if len(historial_posiciones[agente['id']]) > 0:
            historial_posiciones[agente['id']] = [historial_posiciones[agente['id']][0]] * FRAMES_ESPERA + historial_posiciones[agente['id']]

    nombre_base = nombre_caso.replace(' ', '_').lower()
    log_filename = f"telemetria_descentralizada_{nombre_base}.txt"
    with open(log_filename, 'w', encoding='utf-8') as file:
        file.write(f"=== REPORTE MAPF DESCENTRALIZADO: {nombre_caso} ===\n")
        file.write(f"Obstáculos descubiertos: {obstaculos_ocultos}\n\n")
        for agente in agentes:
            file.write(f"--- AGV ID {agente['id']} ---\n")
            trayectoria = historial_posiciones[agente['id']][FRAMES_ESPERA:]
            file.write(f"Nodos Transitados: {' -> '.join([f'({r},{c})' for r, c in trayectoria])}\n\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ['#FF0000', '#00FF00', '#FFA500', '#800080', '#00FFFF']
    
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
            ax.add_patch(patches.Rectangle((c_ini - 0.5, r_ini - 0.5), 1, 1, fill=True, color=col, alpha=0.25))
            pos_est = np.where(matriz_almacen == agente['estanteria_id'])
            if len(pos_est[0]) > 0:
                ax.add_patch(patches.Rectangle((pos_est[1][0] - 0.5, pos_est[0][0] - 0.5), 1, 1, fill=True, color=col, alpha=0.6, hatch='//'))

            if frame < len(historial_posiciones[agente['id']]):
                pos = historial_posiciones[agente['id']][frame]
                ax.plot(pos[1], pos[0], marker='s', markersize=12, markeredgecolor='black', color=col, label=f'AGV {agente["id"]}')
        
        ax.set_title(f'{nombre_caso} | t={frame} (Arquitectura Descentralizada)')
        ax.set_xticks(np.arange(-.5, 13, 1), minor=True); ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=1)
        ax.legend(loc='upper right', prop={'size': 8})

    max_frames = max(len(h) for h in historial_posiciones.values())
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=400, repeat=False)
    ani.save(f"{nombre_base}.gif", writer='pillow')
    print(f"    -> Componentes guardados exitosamente en directorio local.")
    plt.close(fig)

# =============================================================================
# DEFINICIÓN DE CASOS DE PRUEBA CIENTÍFICOS (INTEGRALES)
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
    print("INICIALIZANDO MOTOR MAPF DESCENTRALIZADO (REACTIVO)...")
    for caso in casos_de_prueba: ejecutar_simulacion(caso["nombre"], caso["agentes"], caso["obstaculos"])