import os

import kaolin as kal
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
import cv2

from .mesh import Mesh
from .render import Renderer
from src.configs.train_config import GuideConfig


def build_cotan_laplacian_torch(points_tensor: torch.Tensor, tris_tensor: torch.Tensor) -> np.ndarray:
    tris, points = tris_tensor.cpu().numpy(), points_tensor.cpu().numpy()

    a, b, c = (tris[:, 0], tris[:, 1], tris[:, 2])
    A = np.take(points, a, axis=1)
    B = np.take(points, b, axis=1)
    C = np.take(points, c, axis=1)

    eab, ebc, eca = (B - A, C - B, A - C)
    eab = eab / np.linalg.norm(eab, axis=0)[None, :]
    ebc = ebc / np.linalg.norm(ebc, axis=0)[None, :]
    eca = eca / np.linalg.norm(eca, axis=0)[None, :]

    alpha = np.arccos(-np.sum(eca * eab, axis=0))
    beta = np.arccos(-np.sum(eab * ebc, axis=0))
    gamma = np.arccos(-np.sum(ebc * eca, axis=0))

    wab, wbc, wca = (1.0 / np.tan(gamma), 1.0 / np.tan(alpha), 1.0 / np.tan(beta))
    rows = np.concatenate((a, b, a, b, b, c, b, c, c, a, c, a), axis=0)
    cols = np.concatenate((a, b, b, a, b, c, c, b, c, a, a, c), axis=0)
    vals = np.concatenate((wab, wab, -wab, -wab, wbc, wbc, -wbc, -wbc, wca, wca, -wca, -wca), axis=0)
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(points.shape[1], points.shape[1]), dtype=float).tocsc()
    return L


def build_graph_laplacian_torch(tris_tensor: torch.Tensor) -> np.ndarray:
    tris = tris_tensor.cpu().numpy()
    n_verts = tris.max() + 1
    v2v = [[] for _ in range(n_verts)]
    for face in tris:
        for i in face:
            for j in face:
                if i != j:
                    if j not in v2v[i]:
                        v2v[i].append(j)

    valency = [len(x) for x in v2v]
    I, J, vals = [], [], []
    for i in range(n_verts):
        I.append(i)
        J.append(i)
        vals.append(1)

        for neighbor in v2v[i]:
            I.append(i)
            J.append(neighbor)
            vals.append(-1 / valency[i])
    L = sparse.csr_matrix((vals, (I, J)), shape=(n_verts, n_verts))
    return L


def eigen_problem(Lap, k=20, e=0.0) -> (torch.Tensor, torch.Tensor):
    shift = 1e-4
    eigenvalues, eigenvectors = eigsh(
        Lap + shift * scipy.sparse.eye(Lap.shape[0]),
        k=k + 1, which='LM', sigma=e, tol=1e-3)
    eigenvalues += shift

    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    return torch.from_numpy(eigenvalues).float(), torch.from_numpy(eigenvectors.T).float()


def choose_multi_modal(n: int, k: int):
    interval_length = n // k
    n_intervals = n / interval_length
    if n_intervals == int(n_intervals):
        n_intervals = int(n_intervals)
    else:
        n_intervals = int(n_intervals) + 1
    chosen_numbers = []
    for i in range(n_intervals):
        current_interval_length = min(interval_length, n - i * interval_length)
        chosen_numbers.append(np.random.choice(current_interval_length) + i * interval_length)
    return chosen_numbers

def save_tensor_as_image(tensor, filename, output_dir="output_images"):
    """
    Saves a PyTorch tensor as an image.

    Args:
        tensor (torch.Tensor or torch.nn.Parameter): Input tensor of shape (1, 3, H, W).
        filename (str): Name of the file to save (e.g., "step1_original.png").
        output_dir (str): Directory where images will be saved.

    Returns:
        None (Saves image to disk)
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Convert to tensor if it's a Parameter
    if isinstance(tensor, torch.nn.Parameter):
        tensor = tensor.data  # Access raw tensor

    # Ensure it's a tensor before calling `.detach()`
    if isinstance(tensor, torch.Tensor):
        image_np = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    else:
        raise TypeError("Expected a PyTorch tensor but got a different type.")

    # Normalize if needed (ensure values are in 0-255 range)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    # Save image using OpenCV
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    print(f"Saved: {output_path}")  # Debug message


class TexturedMeshModel(nn.Module):
    def __init__(self,
                 opt: GuideConfig,
                 render_grid_size=1024,
                 texture_resolution=1024,
                 initial_texture_path=None,
                 cache_path=None,
                 device=torch.device('cpu'),
                 augmentations=False,
                 augment_prob=0.5):

        super().__init__()
        self.device = device
        self.augmentations = augmentations
        self.augment_prob = augment_prob
        self.opt = opt
        self.dy = self.opt.dy
        self.mesh_scale = self.opt.shape_scale
        self.texture_resolution = texture_resolution
        if initial_texture_path is not None:
            self.initial_texture_path = initial_texture_path
        else:
            self.initial_texture_path = self.opt.initial_texture
        self.cache_path = cache_path
        self.num_features = 3

        self.renderer = Renderer(device=self.device, dim=(render_grid_size, render_grid_size),
                                 interpolation_mode=self.opt.texture_interpolation_mode)
        self.env_sphere, self.mesh = self.init_meshes()
        self.default_color = [0.8, 0.1, 0.8]
        self.background_sphere_colors, self.texture_img = self.init_paint()
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))

        """self.background_sphere_colors_initial = self.background_sphere_colors.clone()
        self.texture_img_initial = self.texture_img.clone()
        self.meta_texture_img_initial = self.meta_texture_img.clone()

        self.saved_texture_img = []
        self.saved_meta_texture_img = []"""

        if self.opt.reference_texture:
            base_texture = torch.Tensor(np.array(Image.open(self.opt.reference_texture).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
            change_mask = (
                    (base_texture.to(self.device) - self.texture_img).abs().sum(axis=1) > 0.1).float()
            with torch.no_grad():
                self.meta_texture_img[:, 1] = change_mask
        self.vt, self.ft = self.init_texture_map()

        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0),
            self.ft.long()).detach()

        self.n_eigen_values = 20
        self._L = None
        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def L(self) -> np.ndarray:
        if self._L is None:
            self._L = build_cotan_laplacian_torch(self.mesh.vertices.T, self.mesh.faces)
        return self._L

    def eigens(self, k: int, e: float) -> (torch.Tensor, torch.Tensor):
        if self._eigenvalues is None or self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = eigen_problem(self.L, k, e)
            self._eigenvalues, self._eigenvectors = \
                self._eigenvalues.to(self.device), self._eigenvectors.to(self.device)

        return self._eigenvalues, self._eigenvectors

    @staticmethod
    def normalize_vertices(vertices: torch.Tensor, mesh_scale: float = 1.0, dy: float = 0.0) -> torch.Tensor:
        vertices -= vertices.mean(dim=0)[None, :]
        vertices /= vertices.norm(dim=1).max()
        vertices *= mesh_scale
        vertices[:, 1] += dy
        return vertices

    def spectral_augmentations(self, vertices: torch.Tensor) -> torch.Tensor:
        eigen_values, basis_functions = self.eigens(self.n_eigen_values, 0.0)
        basis_functions /= basis_functions.max(dim=-1)[0][:, None] - basis_functions.min(dim=-1)[0][:, None]

        chosen_basis_function = choose_multi_modal(basis_functions.shape[0], 2)
        coeffs = torch.zeros(basis_functions.shape[0]).to(self.device)
        signs = ((torch.rand(len(chosen_basis_function), device=self.device) > 0.5).float() - 0.5) * 2
        coeffs[chosen_basis_function] = signs

        reconstructed = coeffs @ basis_functions.float()

        directions = vertices / torch.norm(vertices, dim=1)[:, None]
        deformed_v = vertices + 0.25 * reconstructed[:, None] * directions
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def axis_augmentations(self, vertices: torch.Tensor, stretch_factor: float = 1.6, squish_factor: float = 0.7):
        axis_indices = np.arange(0, 3)
        axis_indices = np.random.permutation(axis_indices)
        stretch_axis = axis_indices[0]
        squish_axis = axis_indices[1]

        deformed_v = vertices.clone()
        deformed_v[:, stretch_axis] *= stretch_factor
        deformed_v[:, squish_axis] *= squish_factor
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def augment_vertices(self):
        verts = self.mesh.vertices.clone()
        if np.random.rand() < 0.5:
            verts = self.spectral_augmentations(verts)
        if np.random.rand() < 0.5:
            verts = self.axis_augmentations(verts)
        return verts

    def init_meshes(self, env_sphere_path='shapes/env_sphere.obj'):
        env_sphere = Mesh(env_sphere_path, self.device)

        mesh = Mesh(self.opt.shape_path, self.device)
        mesh.normalize_mesh(inplace=True, target_scale=self.mesh_scale, dy=self.dy)

        return env_sphere, mesh

    def zero_meta(self):
        with torch.no_grad():
            self.meta_texture_img[:] = 0

    def init_paint(self, num_backgrounds=1):
        # random color face attributes for background sphere
        init_background_bases = torch.rand(num_backgrounds, 3).to(self.device)
        modulated_init_background_bases_latent = init_background_bases[:, None, None, :] * 0.8 + 0.2 * torch.randn(
            num_backgrounds, self.env_sphere.faces.shape[0],
            3, self.num_features, dtype=torch.float32).cuda()
        background_sphere_colors = nn.Parameter(modulated_init_background_bases_latent.cuda())

        if self.initial_texture_path is not None:
            texture = torch.Tensor(np.array(Image.open(self.initial_texture_path).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
        else:
            texture = torch.ones(1, 3, self.texture_resolution, self.texture_resolution).cuda() * torch.Tensor(
                self.default_color).reshape(1, 3, 1, 1).cuda()
        texture_img = nn.Parameter(texture)
        return background_sphere_colors, texture_img

    def invert_color(self, color: torch.Tensor) -> torch.Tensor:
        # inverse linear approx to find latent
        A = self.linear_rgb_estimator.T
        regularizer = 1e-2

        pinv = (torch.pinverse(A.T @ A + regularizer * torch.eye(4).cuda()) @ A.T)
        if len(color) == 1 or type(color) is torch.Tensor:
            init_color_in_latent = color @ pinv.T
        else:
            init_color_in_latent = pinv @ torch.tensor(
                list(color)).float().to(A.device)
        return init_color_in_latent

    def change_default_to_median(self):
        diff = (self.texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        default_mask = (diff < 0.1).float().unsqueeze(0)
        median_color = self.texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(axis=1)
        with torch.no_grad():
            self.texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)

    def init_texture_map(self):
        cache_path = self.cache_path
        if cache_path is None:
            cache_exists_flag = False
        else:
            vt_cache, ft_cache = cache_path / 'vt.pth', cache_path / 'ft.pth'
            cache_exists_flag = vt_cache.exists() and ft_cache.exists()

        if self.mesh.vt is not None and self.mesh.ft is not None \
                and self.mesh.vt.shape[0] > 0 and self.mesh.ft.min() > -1:
            logger.info('Mesh includes UV map')
            vt = self.mesh.vt.cuda()
            ft = self.mesh.ft.cuda()
        elif cache_exists_flag:
            logger.info(f'running cached UV maps from {vt_cache}')
            vt = torch.load(vt_cache).cuda()
            ft = torch.load(ft_cache).cuda()
        else:
            logger.info(f'running xatlas to unwrap UVs for mesh')
            # unwrap uvs
            import xatlas
            v_np = self.mesh.vertices.cpu().numpy()
            f_np = self.mesh.faces.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()
            if cache_path is not None:
                os.makedirs(cache_path, exist_ok=True)
                torch.save(vt.cpu(), vt_cache)
                torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        return [self.background_sphere_colors, self.texture_img, self.meta_texture_img]

    @torch.no_grad()
    def export_mesh(self, path):
        v, f = self.mesh.vertices, self.mesh.faces.int()
        h0, w0 = 256, 256
        ssaa, name = 1, ''

        # v, f: torch Tensor
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        colors = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        colors = colors[0].cpu().detach().numpy()
        colors = (colors * 255).astype(np.uint8)

        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        colors = Image.fromarray(colors)

        if ssaa > 1:
            colors = colors.resize((w0, h0), Image.LINEAR)

        colors.save(os.path.join(path, f'{name}albedo.png'))

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'{name}mesh.obj')
        mtl_file = os.path.join(path, f'{name}mesh.mtl')

        logger.info('writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {name}mesh.mtl \n')

            logger.info('writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            logger.info('writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                # fp.write(f'vt {v[0]} {1 - v[1]} \n')
                fp.write(f'vt {v[0]} {v[1]} \n')

            logger.info('writing faces {f_np.shape}')
            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {name}albedo.png \n')

    # 여기서 자동으로 previous UV Map을 반영하는데 어디일까
    def render(self, theta=None, phi=None, radius=None, background=None,
               use_meta_texture=False, render_cache=None, use_median=False, dims=None):
        if render_cache is None:
            assert theta is not None and phi is not None and radius is not None
        background_sphere_colors = self.background_sphere_colors[
            torch.randint(0, self.background_sphere_colors.shape[0], (1,))]
        if use_meta_texture:
            texture_img = self.meta_texture_img
        else:
            texture_img = self.texture_img

        if self.augmentations:
            augmented_vertices = self.augment_vertices()
        else:
            augmented_vertices = self.mesh.vertices

        if use_median:
            diff = (texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
                self.device)).abs().sum(axis=1)
            default_mask = (diff < 0.1).float().unsqueeze(0)
            median_color = texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(
                axis=1)
            texture_img = texture_img.clone()
            with torch.no_grad():
                texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)
        background_type = 'none'
        use_render_back = False
        if background is not None and type(background) == str:
            background_type = background
            use_render_back = True
        pred_features, mask, depth, normals, render_cache = self.renderer.render_single_view_texture(augmented_vertices,
            self.mesh.faces,
            self.face_attributes,
            texture_img,
            elev=theta,
            azim=phi,
            radius=radius,
            look_at_height=self.dy,
            render_cache=render_cache,
            dims=dims,
            background_type=background_type)

        mask = mask.detach()

        if use_render_back:
            pred_map = pred_features
            pred_back = pred_features
        else:
            if background is None:
                pred_back, _, _ = self.renderer.render_single_view(self.env_sphere,
                background_sphere_colors,
                elev=theta,
                azim=phi,
                radius=radius,
                dims=dims,
                look_at_height=self.dy, calc_depth=False)
            elif len(background.shape) == 1:
                pred_back = torch.ones_like(pred_features) * background.reshape(1, 3, 1, 1)
            else:
                pred_back = background

            pred_map = pred_back * (1 - mask) + pred_features * mask

        if not use_meta_texture:
            pred_map = pred_map.clamp(0, 1)
            pred_features = pred_features.clamp(0, 1)

        return {'image': pred_map, 'mask': mask, 'background': pred_back,
                'foreground': pred_features, 'depth': depth, 'normals': normals, 'render_cache': render_cache,
                'texture_map': texture_img}

    def draw(self, theta, phi, radius, target_rgb):
        # failed attempt to draw on the texture image

        uv_features, face_idx = self.renderer.project_uv_single_view(self.mesh.vertices,
                                                                     self.mesh.faces,
                                                                     self.face_attributes,
                                                                     elev=theta,
                                                                     azim=phi,
                                                                     radius=radius,
                                                                     look_at_height=self.dy)
        unique_face_idx = torch.unique(face_idx)
        print('')

    def create_basecolor_exclusion_mask_from_tensor(self, input_tensor):
        # 배치 차원이 있는지 확인: 만약 있다면, 첫 번째 배치만 사용
        if input_tensor.dim() == 4:
            tensor = input_tensor[0]  # shape: [C, H, W]
        else:
            tensor = input_tensor  # shape: [H, W, C] 혹은 이미 [C, H, W]
        
        # 채널 차원 확인: 보통 처리하기 쉬운 형태는 [H, W, C]
        if tensor.dim() == 3:
            if tensor.shape[0] == 3:  # [3, H, W] → 전치
                img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
            elif tensor.shape[-1] == 3:  # 이미 [H, W, 3]
                img_np = tensor.detach().cpu().numpy()
            else:
                raise ValueError("입력 텐서의 채널 수가 3이 아닙니다.")
        else:
            raise ValueError("입력 텐서는 3차원이어야 합니다.")
        
        # np.array 값은 [H, W, 3] 형태, 값 범위가 [0,1]이므로 0~255로 변환 후 정수형으로 변환
        img_np = (img_np * 255).astype(np.uint8)
        H, W, C = img_np.shape

        # 평면화하여 각 픽셀의 색상을 (H*W, 3)로 만듦
        pixels = img_np.reshape(-1, C)
        
        # 고유 색상과 등장 횟수를 계산
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        max_index = np.argmax(counts)
        base_color = unique_colors[max_index]  # 가장 많이 등장하는 색상 (예: (R, G, B))
        
        # 이진 마스크 생성: base_color와 정확히 일치하면 0, 나머지는 1
        mask = np.ones((H, W), dtype=np.uint8)
        # 각 픽셀의 색상이 base_color와 일치하는지 확인 (모든 채널이 동일)
        equal_mask = np.all(img_np == base_color, axis=-1)  # shape: [H, W], boolean
        mask[equal_mask] = 0  # base_color 픽셀은 0
        
        # 최종 마스크를 [1, 1, H, W] 형태로 변환
        binary_mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        return binary_mask

    def project_uv_mask_to_view(self, uv_features, overlap_mask):
        """
        uv_features: 렌더링 시 얻은 UV 그리드, 보통 [B, H_uv, W_uv, 2] 형태.
        overlap_mask: UV 도메인에서 생성한 이진 마스크, 보통 [B, 1, H_tex, W_tex] (예: texture 해상도)
        
        목표: overlap_mask를 현재 뷰(2D 이미지 공간)로 투영(프로젝션)하여 [B, 1, H_img, W_img] 형태의 마스크 생성.
        
        절차:
        1. overlap_mask의 공간 해상도가 텍스처 해상도라면, 이를 uv_features 해상도로 보간(interpolate)합니다.
        2. texture_mapping 함수는 두 번째 인자로 입력받은 텐서의 채널 수가 uv_features의 마지막 차원(즉, 2)와 맞아야 하므로,
            overlap_mask를 2채널([B, 2, H_uv, W_uv])로 확장합니다.
        3. uv_features를 그리드로 사용하여 texture_mapping을 호출하면, 2D 이미지 공간상의 투영된 마스크가 나옵니다.
        4. 결과가 여러 채널이면 평균하여 단일 채널 마스크로 만듭니다.
        """
        # (1) uv_features를 grid로 사용하기 위해, 만약 uv_features가 [B, H_uv, W_uv, 2]가 아니라면 변환.
        if uv_features.dim() == 4 and uv_features.shape[-1] != 2:
            uv_grid = uv_features.permute(0, 2, 3, 1)
        else:
            uv_grid = uv_features  # [B, H_uv, W_uv, 2]

        # (2) overlap_mask가 [B,1,H_tex,W_tex] 형태라고 가정하면, 먼저 텍스처 해상도 H_tex, W_tex
        #     이 값을 uv_grid 해상도 (예: H_uv, W_uv)로 보간합니다.
        overlap_mask_resized = F.interpolate(overlap_mask, size=(uv_grid.shape[1], uv_grid.shape[2]),
                                            mode='bilinear', align_corners=False)
        # (3) texture_mapping 함수는 두 번째 입력의 채널 수가 grid의 마지막 차원(2)와 맞아야 하므로, 
        #     overlap_mask_resized [B,1,H_uv,W_uv]를 2채널로 확장.
        overlap_mask_2ch = overlap_mask_resized.repeat(1, 2, 1, 1)
        
        # (4) texture_mapping: uv_grid ( [B, H_uv, W_uv, 2] )를 사용하여 overlap_mask_2ch를 2D 이미지 공간으로 투영.
        projected_mask = kal.render.mesh.texture_mapping(uv_grid, overlap_mask_2ch,
                                                        mode=self.renderer.interpolation_mode)
        # projected_mask 결과가 [B, 2, H_img, W_img]일 가능성이 있으므로, 평균하여 단일 채널로 만듭니다.
        # 먼저, 프로젝트 결과가 PyTorch 표준 형식인 [B, C, H, W]가 되도록 확인하고,
        projected_mask = torch.mean(projected_mask, dim=3, keepdim=True)
        projected_mask = projected_mask.permute(0, 3, 1, 2)  
        
        return projected_mask