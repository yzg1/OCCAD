import bpy
import os
import math
from mathutils import Vector
import time
import numpy as np

# 设置路径
BASE_DIR = r"D:\workspace\data\ABC\obj\abc_0000_1"
OUTPUT_DIR = r"D:\workspace\data\ABC\obj\blender_new"

# 确保输出目录存在，不存在就生成相应目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 渲染设置
render_settings = {
    "resolution_x": 512,
    "resolution_y": 512,
    "samples": 256,
    "output_format": 'PNG'
}

# 相机视角设置保持不变
def generate_camera_positions():
    n_views = 12  # 12个视角
    elevation = math.pi / 10  # 设定高度
    distance = 2.2  # 相机距离中心的距离
    
    camera_positions = []
    for i in range(n_views):
        azimuth = -math.pi + (2 * math.pi * i / n_views)  # 每30°一个视角
        x = distance * math.cos(elevation) * math.cos(azimuth)
        y = distance * math.cos(elevation) * math.sin(azimuth)
        z = distance * math.sin(elevation)
        camera_positions.append((x, y, z))
    
    return camera_positions

CAMERA_VIEWS = generate_camera_positions()

def create_blue_material():
    """创建高对比度的深蓝色材质"""
    material = bpy.data.materials.new(name="Enhanced_Blue_Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    
    # 清理默认节点
    nodes.clear()
    
    # 创建主要着色器节点
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    
    # 深蓝色设置 (RGBA 格式)
    principled.inputs['Base Color'].default_value = (0.0, 0.1, 0.6, 1.0)  # 深蓝色
    
    # 调整材质属性以增强层次
    if 'Metallic' in principled.inputs:
        principled.inputs['Metallic'].default_value = 0.0
    
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.5  # 中等粗糙度
    
    if 'Specular' in principled.inputs:
        principled.inputs['Specular'].default_value = 0.0
    
    # 增加边缘锐度 - 添加法线贴图
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.8  # 控制边缘突出强度
    bump.inputs['Distance'].default_value = 0.5  # 控制法线细节
    
    # 将 Bump 节点链接到法线
    material.node_tree.links.new(bump.outputs['Normal'], principled.inputs['Normal'])
    
    # 连接主要着色器
    material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return material

def setup_lighting():
    # 删除现有灯光
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # 主光源 - 无阴影
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 7))
    sun = bpy.context.active_object
    sun.data.energy = 3.0  # 降低光照强度
    sun.rotation_euler = (math.radians(45), math.radians(45), 0)
    sun.data.use_shadow = False  # 禁用阴影
    
    # 填充光 - 无阴影
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 5))
    fill = bpy.context.active_object
    fill.data.energy = 2.0  # 降低填充光强度
    fill.rotation_euler = (math.radians(-45), math.radians(-45), 0)
    fill.data.size = 5.0  # 增加光源面积，使光更柔和
    fill.data.use_shadow = False  # 禁用阴影
    
    # 环境光 - 无阴影
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    env = bpy.context.active_object
    env.data.energy = 1.5  # 降低环境光强度
    env.rotation_euler = (0, 0, 0)
    env.data.size = 5.0  # 增加光源面积
    env.data.use_shadow = False  # 禁用阴影

def setup_scene():
    """设置场景基本参数"""
    scene = bpy.context.scene
    
    # 设置渲染引擎
    scene.render.engine = 'CYCLES'
    
    # 渲染设置
    scene.cycles.samples = render_settings["samples"]
    scene.cycles.use_denoising = True
    
    # 调整曝光和伽马值以减少过度曝光
    scene.view_settings.exposure = -0.5  # 降低曝光
    scene.view_settings.gamma = 0.9  # 稍微降低伽马值
    
    # 分辨率设置
    scene.render.resolution_x = render_settings["resolution_x"]
    scene.render.resolution_y = render_settings["resolution_y"]
    scene.render.resolution_percentage = 100
    
    # 输出设置
    scene.render.image_settings.file_format = render_settings["output_format"]
    scene.render.image_settings.color_mode = 'RGBA'
    
    # 设置白色背景
    world = bpy.data.worlds.new(name="New World")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs[0].default_value = (1, 1, 1, 1)

def apply_material_to_objects():
    """将蓝色材质应用到所有网格对象"""
    blue_material = create_blue_material()
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # 清除现有材质
            obj.data.materials.clear()
            # 添加新材质
            obj.data.materials.append(blue_material)

def normalize_to_unit_cube():
    """标准化对象到单位立方体"""
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.select_all(action='DESELECT')
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("No mesh objects found")
        return
    
    bpy.context.view_layer.objects.active = mesh_objects[0]
    for obj in mesh_objects:
        obj.select_set(True)
    
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    bounds = []
    for obj in mesh_objects:
        for point in obj.bound_box:
            bounds.append(obj.matrix_world @ Vector(point))
    
    min_point = Vector((min(v[0] for v in bounds), min(v[1] for v in bounds), min(v[2] for v in bounds)))
    max_point = Vector((max(v[0] for v in bounds), max(v[1] for v in bounds), max(v[2] for v in bounds)))
    dimensions = max_point - min_point
    
    scale_factor = 1.0 / max(dimensions)
    
    for obj in mesh_objects:
        obj.scale = (scale_factor, scale_factor, scale_factor)
        obj.location = (0, 0, 0)
    
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def setup_camera():
    """设置相机"""
    bpy.ops.object.select_all(action='DESELECT')
    
    # 删除现有相机
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # 创建新相机
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    
    # 设置相机参数
    camera.data.lens = 50
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000
    
    return camera

def clean_scene():
    """清理场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)

def check_rendering_complete(obj_path):
    """检查是否已经渲染完所有视角"""
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    parent_folder = os.path.basename(os.path.dirname(obj_path))
    obj_output_dir = os.path.join(OUTPUT_DIR, parent_folder)
    
    # 检查是否已经生成了所有12个视角的图片
    existing_renders = [f for f in os.listdir(obj_output_dir) 
                        if f.startswith(f"{obj_name}_view_") and f.endswith('.png')]
    
    return len(existing_renders) == len(CAMERA_VIEWS)

def render_from_views(obj_path):
    """从多个角度渲染物体"""
    try:
        # 检查是否已经渲染完成
        parent_folder = os.path.basename(os.path.dirname(obj_path))
        obj_output_dir = os.path.join(OUTPUT_DIR, parent_folder)
        os.makedirs(obj_output_dir, exist_ok=True)
        
        # 如果已经渲染完成，跳过
        if check_rendering_complete(obj_path):
            print(f"Skipping {obj_path} - all views already rendered")
            return
        
        # 清理场景
        clean_scene()
        
        # 设置场景
        setup_scene()
        setup_lighting()
        camera = setup_camera()
        
        # 导入OBJ文件
        print(f"Importing: {obj_path}")
        bpy.ops.wm.obj_import(filepath=obj_path)
        
        # 标准化物体并应用蓝色材质
        normalize_to_unit_cube()
        apply_material_to_objects()
        
        # 创建输出子目录
        obj_name = os.path.splitext(os.path.basename(obj_path))[0]
        
        # 从每个视角渲染
        for i, view_pos in enumerate(CAMERA_VIEWS):
            output_path = os.path.join(obj_output_dir, f"{obj_name}_view_{i+1}.png")
            
            # 如果特定视角的渲染已存在，则跳过
            if os.path.exists(output_path):
                print(f"Skipping view {i+1} for {obj_name} - already rendered")
                continue
            
            print(f"Rendering view {i+1}/12 for {obj_name}")
            
            camera.location = view_pos
            direction = Vector((0, 0, 0)) - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
            
            bpy.context.scene.camera = camera
            
            bpy.context.scene.render.filepath = output_path
            
            bpy.ops.render.render(write_still=True)
            print(f"Saved render to: {output_path}")
            
    except Exception as e:
        print(f"Error processing {obj_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    start_time = time.time()
    print("开始处理...")
    
    for folder_name in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.obj'):
                obj_path = os.path.join(folder_path, file_name)
                print(f"\n处理文件: {obj_path}")
                render_from_views(obj_path)
    
    end_time = time.time()
    print(f"\n处理完成！总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()