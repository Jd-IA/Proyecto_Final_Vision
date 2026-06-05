import numpy as np
import os
from skimage import measure
from scipy.spatial.distance import cdist

class Voxel:
    
    def __init__(self):
        self.capa = 0
        self.fila = 0
        self.columna = 0
        self.matriz_3d = []
        self.nombre_objeto = ""
        self.ruta_archivo_og = ""
        self.matriz_3d_N4 = []
        self.perimetro_3d_N4 = 0
        self.matriz_codigos_F8 = []
        self.matriz_codigos_AF8 = []
        self.matriz_contornos_coords = []
        self.matriz_esquinas_3d = []
        self.nubes_puntos = {}
        self.p = 0
        self.q = 0
        self.r = 0

    def set_fila(self, fila):
        self.fila = fila

    def set_columna(self, columna):
        self.columna = columna

    def set_capa(self, capa):
        self.capa = capa
    
    def set_matriz_3d(self, matriz):
        self.matriz_3d = matriz

    def set_matriz_3d_N4(self, matriz, perimetro):
        self.matriz_3d_N4 = matriz
        self.perimetro_3d_N4 = perimetro

    def cargar_matriz(self, ruta_archivo=None):
        """Carga la matriz 3D desde un archivo .txt seleccionado por dialogo o por ruta. Inicializa las dimensiones del objeto."""
        
        if ruta_archivo is None:
            try:
                from tkinter import filedialog
                ruta_archivo = filedialog.askopenfilename(
                    title="Seleccionar archivo txt.",
                    filetypes=[("Archivos de texto", "*.txt")]
                )
                if not ruta_archivo:
                    print("No se seleccionó ningún archivo.")
                    return False
            except ImportError:
                print("Error: Proporciona ruta_archivo como parámetro")
                return False
        
        if not os.path.exists(ruta_archivo):
            print(f"Error: Archivo no encontrado: {ruta_archivo}")
            return False

        self.nombre_objeto = os.path.splitext(os.path.basename(ruta_archivo))[0]
        self.ruta_archivo_og = ruta_archivo

        with open(ruta_archivo, 'r') as f:
            contenido = f.read()

        lineas = [l.strip() for l in contenido.replace('\r', '').split('\n') if l.strip()]

        if len(lineas) == 0:
            print("Error: Archivo vacío")
            return False

        columnas = len(lineas[0].split(','))
        filas = columnas
        total_capas = len(lineas) // filas

        capas = []
        for i in range(total_capas):
            bloque = lineas[i * filas : (i + 1) * filas]
            capa = [list(map(int, linea.split(','))) for linea in bloque]
            capas.append(capa)

        matriz = np.array(capas, dtype=np.uint8)

        self.set_matriz_3d(matriz)
        self.set_capa(matriz.shape[0])
        self.set_fila(matriz.shape[1])
        self.set_columna(matriz.shape[2])

        print(f"Matriz cargada: {self.nombre_objeto}")
        print(f"  Shape: {matriz.shape}")
        return True

    def vecindad_N4(self):
        """Calcula el perimetro N4 del objeto voxelizado. Un voxel pertenece al perimetro si al menos uno de sus 4 vecinos 
        (arriba, abajo, izquierda, derecha) esta vacio. Guarda el resultado en self.matriz_3d_N4."""
        if len(self.matriz_3d) == 0:
            print("Error: No se ha cargado una matriz.")
            return

        perimetro = 0
        matriz_n4 = np.zeros_like(self.matriz_3d)

        for i in range(self.capa):
            for j in range(self.fila):
                for k in range(self.columna):
                    if self.matriz_3d[i, j, k] == 1:
                        is_borde = (
                            j == 0 or j == self.fila - 1 or
                            k == 0 or k == self.columna - 1 or
                            self.matriz_3d[i, j - 1, k] == 0 or
                            self.matriz_3d[i, j + 1, k] == 0 or
                            self.matriz_3d[i, j, k - 1] == 0 or
                            self.matriz_3d[i, j, k + 1] == 0
                        )
                        
                        if is_borde:
                            matriz_n4[i, j, k] = 1
                            perimetro += 1

        self.set_matriz_3d_N4(matriz_n4, perimetro)
        print(f"Perimetro N4: {perimetro} voxeles")

    def detectar_componentes_optimizado(self, capa_idx):
        """Detecta las componentes conexas del perimetro N4 de una capa usando conectividad de 8 vecinos. 
        Retorna una lista de arrays de coordenadas, una por componente."""
        if capa_idx >= len(self.matriz_3d_N4):
            return []

        # Usar el perimetro N4
        capa = self.matriz_3d_N4[capa_idx]

        if capa.sum() == 0:
            return []


        labeled = measure.label(capa, connectivity=2)
        regiones = measure.regionprops(labeled)

        if len(regiones) == 0:
            return []

        componentes = [region.coords for region in regiones]
        return sorted(componentes, key=lambda c: np.min(np.sum(c**2, axis=1)))

    def f8(self):
        """Calcula el codigo de cadena F8 (Freeman chain code) sobre el perimetro N4 para cada capa.
         Recorre el contorno de cada componente y codifica las direcciones de movimiento."""
        if len(self.matriz_3d_N4) == 0:
            print("Error: primero ejecuta vecindad_N4()")
            return

        self.matriz_codigos_F8 = []
        self.matriz_contornos_coords = []

        for capa_idx in range(self.capa):

            componentes = self.detectar_componentes_optimizado(capa_idx)

            if len(componentes) == 0:
                self.matriz_codigos_F8.append([])
                self.matriz_contornos_coords.append([])
                continue

            componentes_codificadas = []

            for coords in componentes:
                f8_code = self._codificar_f8_opt(coords)

                if len(f8_code) > 0:
                    componentes_codificadas.append({
                        'codigo': f8_code,
                        'coords': coords,
                        'num_pixeles': len(coords)
                    })

            self.matriz_codigos_F8.append(componentes_codificadas)
            coords_lista = [c['coords'] for c in componentes_codificadas]
            self.matriz_contornos_coords.append(coords_lista)

    def _codificar_f8_opt(self, coords):
        """Recorre el contorno de una componente y genera su codigo F8.
        Implementa el algoritmo de seguimiento de contorno de Pavlidis partiendo del pixel mas arriba-izquierda."""
        if len(coords) < 2:
            return []

        coords_set = set(map(tuple, coords))
        # Punto inicial: mas arriba (menor fila), luego mas a la izquierda (menor col)
        coords_arr = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        r0, c0 = int(coords_arr[0, 0]), int(coords_arr[0, 1])

        # Deltas por direccion Freeman
        dr = [ 0, -1, -1, -1,  0,  1,  1,  1]
        dc = [ 1,  1,  0, -1, -1, -1,  0,  1]

        codigo = []
        r, c = r0, c0

        # El pixel inicial es el mas arriba-izquierda.
        # Asumimos que venimos del Oeste (dir 4), asi que empezamos
        # buscando desde (4 + 6) % 8 = 2 (Norte) en sentido antihorario
        # es decir incrementando la direccion
        dir_actual = (4 + 6) % 8  # = 2

        max_iter = len(coords) * 2 + 8

        for _ in range(max_iter):
            encontrado = False

            for _ in range(8):
                nr = r + dr[dir_actual]
                nc = c + dc[dir_actual]

                if (nr, nc) in coords_set:
                    codigo.append(dir_actual)
                    r, c = nr, nc
                    # Proxima busqueda empieza desde (dir_actual + 6) % 8
                    dir_actual = (dir_actual + 6) % 8
                    encontrado = True
                    break

                dir_actual = (dir_actual + 1) % 8

            if not encontrado:
                break

            if r == r0 and c == c0:
                break

        return codigo

    def af8(self):
        """Convierte los codigos F8 a AF8 (Angle Freeman chain code). Calcula la diferencia modulo 8 entre simbolos F8 consecutivos."""
        if len(self.matriz_codigos_F8) == 0:
            print("Error: primero ejecuta f8()")
            return

        self.matriz_codigos_AF8 = []

        for componentes_f8 in self.matriz_codigos_F8:
            if len(componentes_f8) == 0:
                self.matriz_codigos_AF8.append([])
                continue
            
            componentes_af8 = []
            
            for comp_data in componentes_f8:
                f8 = comp_data['codigo']
                
                if len(f8) == 0:
                    continue
                
                f8_array = np.array(f8, dtype=int)

                diffs = np.diff(f8_array) % 8
                af8 = [int(f8_array[0])] + diffs.tolist()
                
                componentes_af8.append({
                    'codigo_af8': af8,
                    'coords': comp_data['coords'],
                    'num_pixeles': comp_data['num_pixeles']
                })
            
            self.matriz_codigos_AF8.append(componentes_af8)

    def mostrar_af8_por_capa(self):
        """Imprime en consola un resumen del codigo AF8 por capa: numero de componentes,
        pixeles totales y longitud del codigo."""
        
        if len(self.matriz_codigos_AF8) == 0:
            print("Error: primero ejecuta af8()")
            return
        
        print("\n" + "="*80)
        print("INFORMACIÓN AF8 POR CAPA")
        print("="*80)
        print(f"{'Capa':>4} {'Componentes':>12} {'Píxeles':>10} {'AF8 Total':>12}")
        print("-"*80)
        
        for capa_idx, componentes_af8 in enumerate(self.matriz_codigos_AF8):
            if len(componentes_af8) == 0:
                print(f"{capa_idx:4d} {'(vacía)':>12}")
            else:
                total_pixeles = sum(c['num_pixeles'] for c in componentes_af8)
                total_af8 = sum(len(c['codigo_af8']) for c in componentes_af8)
                print(f"{capa_idx:4d} {len(componentes_af8):12d} {total_pixeles:10d} {total_af8:12d}")
        
        print("="*80 + "\n")

    def detectar_esquinas_dss(self, p, q, r):
        """Detecta los key points del objeto aplicando la gramatica sobre el codigo AF8. 
        Para cada componente busca los simbolos que inician un DSS valido con los parametros dados."""
        
        self.p = p
        self.q = q
        self.r = r
        
        if len(self.matriz_codigos_AF8) == 0:
            print("Error: primero ejecuta af8()")
            return

        if not self._validar_parametros_dss(p, q, r):
            return

        self.matriz_esquinas_3d = []

        for capa_idx, componentes_af8 in enumerate(self.matriz_codigos_AF8):
            
            if len(componentes_af8) == 0:
                self.matriz_esquinas_3d.append({
                    'capa': capa_idx,
                    'esquinas': [],
                    'num_esquinas': 0,
                    'es_vacia': True,
                    'break_points': [],
                    'N4': 0,
                    'DP': 0,
                    'ISE': 0.0,
                    'CR': 0.0,
                    'FOM': 0.0
                })
                continue

            for comp_data in componentes_af8:
                af8 = comp_data['codigo_af8']
                coords = comp_data['coords']
                
                break_points = self._detectar_breakpoints_dss(af8, p, q, r)

                if coords is None or len(coords) == 0:
                    self.matriz_esquinas_3d.append({
                        'capa': capa_idx,
                        'esquinas': [],
                        'num_esquinas': 0,
                        'es_vacia': True,
                        'break_points': break_points,
                        'N4': 0,
                        'DP': 0,
                        'ISE': 0.0,
                        'CR': 0.0,
                        'FOM': 0.0
                    })
                    continue

                N4_capa = len(coords)
                DP = len(break_points)
                ISE = self._calcular_ise_total(coords, break_points)
                CR = N4_capa / DP if DP > 0 else 0
                FOM = N4_capa / (DP * ISE) if (DP > 0 and ISE > 0) else 0

                esquinas = self._extraer_esquinas(capa_idx, coords, break_points)

                self.matriz_esquinas_3d.append({
                    'capa': capa_idx,
                    'esquinas': esquinas,
                    'break_points': break_points,
                    'num_esquinas': len(esquinas),
                    'es_vacia': False,
                    'N4': N4_capa,
                    'DP': DP,
                    'ISE': ISE,
                    'CR': CR,
                    'FOM': FOM
                })

    def mostrar_resumen_metricas_por_capa(self):
        """Imprime en consola una tabla con las metricas por capa: N4, DP, ISE, CR y FOM."""
        
        if not self.matriz_esquinas_3d:
            print("Error: primero ejecuta detectar_esquinas_dss()")
            return

        print("\n" + "="*80)
        print(f"RESUMEN DE MÉTRICAS POR CAPA - DSS: Xa^{self.p}(Ya^{self.q})^{self.r}")
        print("="*80)
        print(f"{'Capa':>4} {'N4':>6} {'DP':>5} {'ISE':>10} {'CR':>8} {'FOM':>10}")
        print("-"*80)

        total_n4 = 0
        total_dp = 0

        for capa_data in self.matriz_esquinas_3d:
            if capa_data['es_vacia']:
                continue

            capa = capa_data['capa']
            N4 = capa_data['N4']
            DP = capa_data['DP']
            ISE = capa_data['ISE']
            CR = capa_data['CR']
            FOM = capa_data['FOM']

            print(f"{capa:4d} {N4:6d} {DP:5d} {ISE:10.2f} {CR:8.2f} {FOM:10.4f}")

            total_n4 += N4
            total_dp += DP

        print("-"*80)
        print(f"{'TOTAL':>4} {total_n4:6d} {total_dp:5d}")
        print("="*80 + "\n")

    def exportar_metricas(self):
        """Exporta las metricas por capa a un archivo .txt en la carpeta del script con el nombre del objeto y los parametros p, q, r."""
        
        if not self.matriz_esquinas_3d:
            print("Error: primero ejecuta detectar_esquinas_dss()")
            return
        
        # Obtener carpeta del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        archivo_salida = os.path.join(script_dir, f"{self.nombre_objeto}_p{self.p}_q{self.q}_r{self.r}_metricas.txt")
        
        try:
            with open(archivo_salida, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"MÉTRICAS DSS: Xa^{self.p}(Ya^{self.q})^{self.r}\n")
                f.write(f"Objeto: {self.nombre_objeto}\n")
                f.write("="*80 + "\n")
                f.write(f"{'Capa':>4} {'N4':>6} {'DP':>5} {'ISE':>10} {'CR':>8} {'FOM':>10}\n")
                f.write("-"*80 + "\n")

                total_n4 = 0
                total_dp = 0

                for capa_data in self.matriz_esquinas_3d:
                    if capa_data['es_vacia']:
                        continue

                    capa = capa_data['capa']
                    N4 = capa_data['N4']
                    DP = capa_data['DP']
                    ISE = capa_data['ISE']
                    CR = capa_data['CR']
                    FOM = capa_data['FOM']

                    f.write(f"{capa:4d} {N4:6d} {DP:5d} {ISE:10.2f} {CR:8.2f} {FOM:10.4f}\n")

                    total_n4 += N4
                    total_dp += DP

                f.write("-"*80 + "\n")
                f.write(f"{'TOTAL':>4} {total_n4:6d} {total_dp:5d}\n")
                f.write("="*80 + "\n")
            
            print(f"Metricas exportadas: {archivo_salida}")
            return archivo_salida
            
        except Exception as e:
            print(f"Error al exportar: {e}")
            return None

    def generar_nubes_puntos(self, delta_values):
        """Genera nubes de puntos filtrando las capas segun el parametro delta. Solo incluye capas cuyos indices son multiplos de delta."""
        
        if not self.matriz_esquinas_3d:
            print("Error: primero ejecuta detectar_esquinas_dss()")
            return

        self.nubes_puntos = {}

        for delta in delta_values:
            puntos_nube = []
            capas_procesadas = []

            for capa_idx in range(self.capa):
                if capa_idx % delta != 0:
                    continue

                capas_procesadas.append(capa_idx)

                # Buscar todas las entradas que pertenecen a esta capa
                # (puede haber varias por componentes multiples)
                entradas_capa = [e for e in self.matriz_esquinas_3d if e['capa'] == capa_idx]

                for capa_data in entradas_capa:
                    if capa_data['es_vacia'] or capa_data['num_esquinas'] == 0:
                        continue
                    for esquina in capa_data['esquinas']:
                        puntos_nube.append({
                            'x': esquina['x'],
                            'y': esquina['y'],
                            'z': esquina['z']
                        })

            num_puntos_original = sum(c['num_esquinas'] for c in self.matriz_esquinas_3d if not c['es_vacia'])
            num_puntos_nube = len(puntos_nube)
            simplificacion = 100 * (1 - num_puntos_nube / num_puntos_original) if num_puntos_original > 0 else 0

            self.nubes_puntos[delta] = {
                'delta': delta,
                'puntos': puntos_nube,
                'num_puntos': num_puntos_nube,
                'num_puntos_original': num_puntos_original,
                'simplificacion': simplificacion,
                'capas_procesadas': capas_procesadas,
                'num_capas': len(capas_procesadas)
            }

    def exportar_nubes_con_nombre(self):
        """Exporta las nubes de puntos generadas a archivos .txt en la carpeta del script. Cada archivo contiene las coordenadas x y z por linea."""
        
        if not self.nubes_puntos:
            print("Error: primero ejecuta generar_nubes_puntos()")
            return []

        # Obtener carpeta del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        archivos = []
        
        # Asegurar que siempre se exporte delta=1 (original) + las solicitadas
        deltas_a_exportar = sorted(set([1] + list(self.nubes_puntos.keys())))
        
        for delta in deltas_a_exportar:
            archivo = os.path.join(script_dir, f"cloud_{self.nombre_objeto}_p{self.p}_q{self.q}_r{self.r}_delta={delta}.txt")
            
            try:
                # Si delta=1 no fue generado, saltarlo (ya que debe estar en nubes_puntos)
                if delta not in self.nubes_puntos:
                    continue
                
                puntos = self.nubes_puntos[delta]['puntos']
                simplificacion = self.nubes_puntos[delta]['simplificacion']
                
                with open(archivo, 'w') as f:
                    for punto in puntos:
                        f.write(f"{punto['x']} {punto['y']} {punto['z']}\n")
                
                print(f"Nube exportada: {archivo}")
                archivos.append(archivo)
                
            except Exception as e:
                print(f"Error al exportar: {e}")
        
        return archivos

    def calcular_hausdorff_n4_vs_dss(self, delta):
        """Calcula la distancia de Hausdorff y el error geometrico promedio entre la nube N4 del objeto y la nube DSS generada con el delta indicado."""
        
        # V1: Coordenadas N4
        V1 = np.argwhere(self.matriz_3d_N4 == 1)
        
        # V2: Esquinas DSS filtradas por delta
        V2 = []
        for capa_idx in range(self.capa):
            if capa_idx % delta != 0:
                continue

            # Buscar todas las entradas que pertenecen a esta capa
            entradas_capa = [e for e in self.matriz_esquinas_3d if e['capa'] == capa_idx]

            for capa_data in entradas_capa:
                if capa_data['es_vacia']:
                    continue
                for esquina in capa_data['esquinas']:
                    V2.append([esquina['z'], esquina['y'], esquina['x']])
        
        if len(V2) == 0:
            print(f"Error: No hay esquinas con delta={delta}")
            return None
        
        V1 = np.array(V1, dtype=float)
        V2 = np.array(V2, dtype=float)
        
       
        print("\n" + "="*80)
        print("PASO 4: CALCULAR HAUSDORFF DISTANCE")
        print("="*80)
        
        # Calcular distancias
        distances = cdist(V1, V2, metric='euclidean')
        min_distances_V1 = np.min(distances, axis=1)
        min_distances_V2 = np.min(distances, axis=0)
        
        max_v1 = np.max(min_distances_V1)
        max_v2 = np.max(min_distances_V2)
        hau = max(max_v1, max_v2)
        
        print(f"\nHau(V1, V2) = max{{max(min_distances_V1), max(min_distances_V2)}}")
        print(f"  max(min_distances_V1) = {max_v1:.3f}")
        print(f"  max(min_distances_V2) = {max_v2:.3f}")
        print(f"\n  Hau(V1, V2) = max{{{max_v1:.3f}, {max_v2:.3f}}}")
        print(f"  Hau(V1, V2) = {hau:.3f}")
        
        print("\n" + "="*80)
        print("PASO 5: AVERAGE GEOMETRIC ERROR")
        print("="*80)
        
        avg_error = np.mean(min_distances_V1)
        sum_distances = np.sum(min_distances_V1)
        
        print(f"\nA_avg(V1, V2) = (1/|V1|) x sum(min_distances)")
        print(f"  A_avg(V1, V2) = (1/{len(V1)}) x ({sum_distances:.3f})")
        print(f"  A_avg(V1, V2) = {avg_error:.3f}")
        
        print("\n" + "="*80)
        
        return {
            'hausdorff_distance': hau,
            'avg_error': avg_error,
            'total_v1': len(V1),
            'total_v2': len(V2)
        }

    def exportar_n4(self):
        """Exporta las coordenadas de los voxeles del perimetro N4 a un archivo .txt. Cada linea contiene x y z sin encabezados."""
        if len(self.matriz_3d_N4) == 0:
            print("Error: primero ejecuta vecindad_N4()")
            return None

        coords = np.argwhere(self.matriz_3d_N4 == 1)

        if len(coords) == 0:
            print("Error: la matriz N4 no contiene voxeles de perimetro")
            return None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        archivo_salida = os.path.join(script_dir, f"cloud_{self.nombre_objeto}_N4.txt")

        try:
            with open(archivo_salida, 'w') as f:
                for capa, fila, col in coords:
                    f.write(f"{int(col)} {int(fila)} {int(capa)}\n")

            print(f"Nube N4 exportada: {archivo_salida}")
            print(f"  Total voxeles: {len(coords)}")
            return archivo_salida

        except Exception as e:
            print(f"Error al exportar N4: {e}")
            return None

    def _validar_parametros_dss(self, p, q, r):
        """Verifica que los parametros p, q y r sean no negativos."""
        if p < 0 or q < 0 or r < 0:
            print(f"Error: p, q, r deben ser no-negativos")
            return False
        return True

    def _detectar_breakpoints_dss(self, af8, p, q, r):
        """Recorre el codigo AF8 de una componente y retorna las posiciones de los key points que inician un DSS valido con los parametros dados."""
        n = len(af8)
        if n == 0:
            return []

        key_points = []
        i = 0

        while i < n:
            fin = self._match_dss(af8, i, p, q, r)

            if fin > i:
                # X inicia un DSS valido es key point; saltar el DSS completo
                key_points.append(i)
                i = fin
            else:
                # No inicia un DSS valido NO es key point
                i += 1

        return key_points

    def _es_par_Y(self, af8, j):
        """Verifica si en la posicion j del codigo AF8 hay un par diagonal Y, es decir (1,7) o (7,1)."""
        n = len(af8)
        if j + 1 >= n:
            return False
        return ((af8[j] == 1 and af8[j + 1] == 7) or
                (af8[j] == 7 and af8[j + 1] == 1))

    def _match_dss(self, af8, start, p, q, r):
        """Intenta hacer match del patron Xa^p(Ya^q)^r en el codigo AF8 a partir de la posicion start. Retorna la posicion final del DSS si hay match, o start si no."""
        n = len(af8)
        i = start

        if i >= n:
            return start

        # X: consumir simbolo inicial (cualquier direccion)
        i += 1

        # a^p: AL MENOS p ceros consecutivos
        count_p = 0
        while count_p < p:
            if i < n and af8[i] == 0:
                i += 1
                count_p += 1
            else:
                return start  # no hay suficientes ceros

        # Si r == 0, el DSS es Xa^p — cualquier X seguido de p ceros es key point
        if r == 0:
            return i

        # (Ya^q)^r: exactamente r repeticiones de (par Y + q ceros)
        for _ in range(r):
            # Y = par diagonal bh=(1,7) o hb=(7,1)
            if self._es_par_Y(af8, i):
                i += 2  # consumir el par (2 simbolos)
            else:
                return start  # no encontro par Y requerido

            # a^q: almenos q ceros
            count_q = 0
            while count_q < q:
                if i < n and af8[i] == 0:
                    i += 1
                    count_q += 1
                else:
                    return start  # no hay suficientes ceros

        return i  # match completo


    def _calcular_ise_total(self, coords, break_points):
        """Calcula el error cuadratico integral (ISE) sumando el cuadrado de la distancia de cada pixel del contorno a su segmento de aproximacion poligonal."""
        if len(coords) < 2 or len(break_points) < 2:
            return 0.0

        ISE = 0.0
        n = len(coords)

        for idx in range(len(break_points)):
            i_ini = break_points[idx]
            i_fin = break_points[(idx + 1) % len(break_points)]

            if i_fin <= i_ini:
                i_fin += n

            xk = float(coords[i_ini % n][1])
            yk = float(coords[i_ini % n][0])
            xk1 = float(coords[i_fin % n][1])
            yk1 = float(coords[i_fin % n][0])

            dxk = xk1 - xk
            dyk = yk1 - yk

            for k in range(i_ini, min(i_fin + 1, n)):
                xi = float(coords[k % n][1])
                yi = float(coords[k % n][0])

                if dxk == 0 and dyk == 0:
                    continue

                numerador = ((xi - xk) * dyk - (yi - yk) * dxk) ** 2
                denominador = dxk ** 2 + dyk ** 2

                if denominador > 0:
                    d2 = numerador / denominador
                    ISE += d2

        return ISE

    def _extraer_esquinas(self, capa_idx, coords, break_points):
        """Extrae las coordenadas 3D (x, y, z) de los key points detectados en una capa y las retorna como lista de diccionarios."""
        esquinas = []

        for bp_idx in break_points:
            if bp_idx < len(coords):
                fila, col = coords[bp_idx]
                esquinas.append({
                    'indice': bp_idx,
                    'fila': int(fila),
                    'columna': int(col),
                    'capa': capa_idx,
                    'x': int(col),
                    'y': int(fila),
                    'z': capa_idx
                })

        return esquinas

if __name__ == "__main__":

    
    # flujo completo
    #voxel = Voxel()
    #voxel.cargar_matriz()
    #voxel.vecindad_N4()
    #voxel.exportar_n4()
    #voxel.f8()
    #voxel.af8()
    
    # Mostrar AF8 por capa
    #voxel.mostrar_af8_por_capa()
    
    # Detectar esquinas
    #voxel.detectar_esquinas_dss(p=13, q=9, r=7)
    
    # Mostrar resumen de métricas por capa
    #voxel.mostrar_resumen_metricas_por_capa()
    
    # Generar nubes
    #original = 1
    #d1 = 5
    #voxel.generar_nubes_puntos([original,d1])
    
    # Exportar métricas
    #voxel.exportar_metricas()
    
    # Exportar nubes
    #voxel.exportar_nubes_con_nombre()
    
    # Calcular Hausdorff Distance
    #voxel.calcular_hausdorff_n4_vs_dss(delta=d1)
#========================================================================
    voxel = Voxel()
    voxel.cargar_matriz()
    voxel.vecindad_N4()
    voxel.exportar_n4()
    voxel.f8()
    voxel.af8()
    voxel.detectar_esquinas_dss(p=1, q=1, r=1)
    #voxel.mostrar_resumen_metricas_por_capa()
    voxel.generar_nubes_puntos([2])  

    #voxel.exportar_metricas()
    #voxel.exportar_nubes_con_nombre()
    voxel.calcular_hausdorff_n4_vs_dss(delta=2)

    voxel_2 = Voxel()
    voxel_2.cargar_matriz()
    voxel_2.vecindad_N4()
    voxel_2.exportar_n4()
    voxel_2.f8()
    voxel_2.af8()

    voxel_2.detectar_esquinas_dss(p=2, q=2, r=2)
    #voxel_2.mostrar_resumen_metricas_por_capa()
    voxel_2.generar_nubes_puntos([3])  
    #voxel_2.exportar_metricas()
    #voxel_2.exportar_nubes_con_nombre()
    voxel_2.calcular_hausdorff_n4_vs_dss(delta=3)