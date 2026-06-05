
import bpy
import os


RUTA_ARCHIVO = r"Ruta del archivo"
RADIO_PUNTO = 0.3

COLOR = (0.2, 0.7, 1.0, 1.0)

def cargar_nube_puntos(ruta_archivo):
    """Lee el archivo .txt y retorna lista de coordenadas (x, y, z)"""
    puntos = []
    if not os.path.exists(ruta_archivo):
        print(f"ERROR: Archivo no encontrado: {ruta_archivo}")
        return puntos

    with open(ruta_archivo, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith('#') or linea == '':
                continue  
            partes = linea.split()
            if len(partes) >= 3:
                try:
                    x, y, z = float(partes[0]), float(partes[1]), float(partes[2])
                    puntos.append((x, y, z))
                except ValueError:
                    pass  
    return puntos


def crear_material(nombre, color):
    """Crea un material de color solido para los puntos"""
    mat = bpy.data.materials.new(name=nombre)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = 2.0

    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def importar_como_mesh(puntos, radio, material):
    """
    Crea la nube como un mesh de instancias usando geometry nodes.
    Metodo eficiente para grandes nubes de puntos.
    """
    import bmesh

    # Crear mesh con un vertice por punto
    mesh = bpy.data.meshes.new("NubePuntos_Mesh")
    bm = bmesh.new()
    for p in puntos:
        bm.verts.new(p)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("NubePuntos", mesh)
    bpy.context.collection.objects.link(obj)

    # Agregar Geometry Nodes para instanciar esferas en cada vertice
    mod = obj.modifiers.new("PuntosGN", 'NODES')
    node_group = bpy.data.node_groups.new("PuntosGN", 'GeometryNodeTree')
    mod.node_group = node_group

    nodes = node_group.nodes
    links = node_group.links

    # Nodos necesarios
    node_input  = nodes.new('NodeGroupInput')
    node_output = nodes.new('NodeGroupOutput')
    node_points = nodes.new('GeometryNodeMeshToPoints')
    node_sphere = nodes.new('GeometryNodeMeshIcoSphere')
    node_inst   = nodes.new('GeometryNodeInstanceOnPoints')
    node_realize = nodes.new('GeometryNodeRealizeInstances')

    # Interfaces (Blender 4.x)
    try:
        node_group.interface.new_socket('Geometry', in_out='INPUT',  socket_type='NodeSocketGeometry')
        node_group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    except AttributeError:
        # Blender 3.x
        node_group.inputs.new('NodeSocketGeometry', 'Geometry')
        node_group.outputs.new('NodeSocketGeometry', 'Geometry')

    # Configurar esfera (radio del punto)
    node_sphere.inputs['Radius'].default_value = radio
    node_sphere.inputs['Subdivisions'].default_value = 2

    # Posicionar nodos en el editor
    node_input.location   = (-400, 0)
    node_points.location  = (-200, 0)
    node_sphere.location  = (-200, -200)
    node_inst.location    = (0, 0)
    node_realize.location = (200, 0)
    node_output.location  = (400, 0)

    # Conectar nodos
    links.new(node_input.outputs['Geometry'],    node_points.inputs['Mesh'])
    links.new(node_points.outputs['Points'],     node_inst.inputs['Points'])
    links.new(node_sphere.outputs['Mesh'],       node_inst.inputs['Instance'])
    links.new(node_inst.outputs['Instances'],    node_realize.inputs['Geometry'])
    links.new(node_realize.outputs['Geometry'],  node_output.inputs['Geometry'])

    # Asignar material
    obj.data.materials.append(material)
    return obj


def main():
    print("="*60)
    print("Importando nube de puntos en Blender...")
    print(f"Archivo: {RUTA_ARCHIVO}")
    print("="*60)

    # Cargar puntos
    puntos = cargar_nube_puntos(RUTA_ARCHIVO)
    if not puntos:
        print("ERROR: No se cargaron puntos. Verifica la ruta del archivo.")
        return

    print(f"Puntos cargados: {len(puntos)}")

    # Crear material
    material = crear_material("Material_NubePuntos", COLOR)

    # Crear objeto en Blender
    obj = importar_como_mesh(puntos, RADIO_PUNTO, material)

    # Centrar vista en el objeto
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.view3d.view_selected()

    print(f"Listo. Objeto '{obj.name}' creado con {len(puntos)} puntos.")
    print("="*60)


main()
