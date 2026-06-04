import torch
from .utils.misc import logger, synchronize_timer
import inspect
from dataclasses import dataclass
from typing import List, Optional
import trimesh
import numpy as np
from tqdm import tqdm
import copy
from typing import List, Optional, Union
import os
from safetensors.torch import load_file
from .utils.mesh_utils import (
    SampleMesh,
    load_surface_points,
    sample_bbox_points_from_trimesh,
    explode_mesh,
    fix_mesh,
)
from .utils.misc import (
    init_from_ckpt,
    instantiate_from_config,
    get_config_from_file,
    smart_load_model,
)
from easydict import EasyDict

import json
from diffusers.utils.torch_utils import randn_tensor
from pathlib import Path


# Text2Part v2 patch: split PartFormer execution into observable stages while
# preserving the original __call__ API for existing callers.
@dataclass
class PreparedInputs:
    obj_surface: torch.Tensor
    obj_surface_raw: Optional[dict]
    mesh: Optional[trimesh.Trimesh]
    center: np.ndarray
    scale: float
    seed: int


@dataclass
class BBoxInputs:
    prepared: PreparedInputs
    aabb: torch.Tensor

    @property
    def n_bbox(self) -> int:
        return int(self.aabb.shape[0])


@dataclass
class PipelineInputs:
    obj_surface: torch.Tensor
    aabb: torch.Tensor
    part_surface_inbbox: torch.Tensor
    mesh: Optional[trimesh.Trimesh]
    center: np.ndarray
    scale: float
    raw_bbox_count: int
    valid_bbox_count: int


@synchronize_timer("Export to trimesh")
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                mesh_output = fix_mesh(mesh_output)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        mesh_output = fix_mesh(mesh_output)
        return mesh_output


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[Union[List[float], np.ndarray]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to"
            " set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps`"
                " does not support custom timestep schedules. Please check whether you"
                " are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps`"
                " does not support custom sigmas schedules. Please check whether you"
                " are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TokenAllocMixin:

    def allocate_tokens(self, bboxes, num_latents=512):
        return np.array([num_latents] * bboxes.shape[0])


class PartFormerPipeline(TokenAllocMixin):

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        bbox_predictor=None,
        verbose=False,
        **kwargs,
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.kwargs = kwargs
        self.bbox_predictor = bbox_predictor
        self.verbose = verbose
        self.kwargs = kwargs

    @classmethod
    @synchronize_timer("Hunyuan3D PartGen Pipeline Model Loading")
    def from_single_file(
        cls,
        ckpt_path=None,
        config=None,
        device="cuda",
        dtype=torch.float32,
        use_safetensors=None,
        ignore_keys=(),
        **kwargs,
    ):
        # prepare config
        if config is None:
            config = get_config_from_file(
                str(
                    Path(__file__).parent.parent
                    / "config"
                    / "partformer_full_pipeline_512_with_sonata.yaml"
                )
            )
        # TODO:
        if ckpt_path is None:
            ckpt_path = str(
                Path(__file__).parent
                / "ckpts"
                / "partformer_full_pipeline_512_with_sonata.ckpt"
            )
        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace(".ckpt", ".safetensors")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            # parse safetensors
            import safetensors.torch

            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split(".")[0]
                new_key = key[len(model_name) + 1 :]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            # ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt = torch.load(ckpt_path, map_location="cpu")
        # load model
        model = instantiate_from_config(config["model"])
        # model.load_state_dict(ckpt["model"])
        init_from_ckpt(model, ckpt, prefix="model", ignore_keys=ignore_keys)
        vae = instantiate_from_config(config["shapevae"])
        # vae.load_state_dict(ckpt["shapevae"], strict=False)
        init_from_ckpt(vae, ckpt, prefix="shapevae", ignore_keys=ignore_keys)
        if config.get("conditioner", None) is not None:
            conditioner = instantiate_from_config(config["conditioner"])
            init_from_ckpt(
                conditioner, ckpt, prefix="conditioner", ignore_keys=ignore_keys
            )
        else:
            conditioner = vae
        scheduler = instantiate_from_config(config["scheduler"])
        bbox_predictor = instantiate_from_config(config.get("bbox_predictor", None))
        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            bbox_predictor=bbox_predictor,  # TODO: add bbox predictor
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)
        return cls(**model_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_path="tencent/Hunyuan3D-Part",
        dtype=torch.float32,
        device="cuda",
        **kwargs,
    ):
        model_dir = smart_load_model(
            model_path=model_path,
        )
        model_ckpt = load_file(os.path.join(model_dir, "model/model.safetensors"))
        conditioner_ckpt = load_file(
            os.path.join(model_dir, "conditioner/conditioner.safetensors")
        )
        shapevae_ckpt = load_file(
            os.path.join(model_dir, "shapevae/shapevae.safetensors")
        )
        p3sam_path = os.path.join(model_dir, "p3sam/p3sam.safetensors")
        with open(os.path.join(model_dir, "model/config.json"), "r") as f:
            model_config = EasyDict(json.load(f))
        with open(os.path.join(model_dir, "conditioner/config.json"), "r") as f:
            conditioner_config = EasyDict(json.load(f))
        with open(os.path.join(model_dir, "shapevae/config.json"), "r") as f:
            shapevae_config = EasyDict(json.load(f))
        with open(os.path.join(model_dir, "scheduler/config.json"), "r") as f:
            scheduler_config = EasyDict(json.load(f))
        with open(os.path.join(model_dir, "p3sam/config.json"), "r") as f:
            bbox_predictor_config = EasyDict(json.load(f))
            bbox_predictor_config["params"]["ckpt_path"] = p3sam_path
        # load model
        model = instantiate_from_config(model_config)
        model.load_state_dict(model_ckpt)
        vae = instantiate_from_config(shapevae_config)
        vae.load_state_dict(shapevae_ckpt)
        conditioner = instantiate_from_config(conditioner_config)
        conditioner.load_state_dict(conditioner_ckpt)
        scheduler = instantiate_from_config(scheduler_config)
        bbox_predictor = instantiate_from_config(bbox_predictor_config)
        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            bbox_predictor=bbox_predictor,  # TODO: add bbox predictor
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)
        return cls(**model_kwargs)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)
        if self.bbox_predictor is not None:
            self.bbox_predictor.to(device=device, dtype=dtype)

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def predict_bbox(
        self, mesh: trimesh.Trimesh, scale_box=1.0, drop_normal=True, seed=42
    ):
        """
        Predict the bounding box of the object surface.
        Args:
            obj_surface (`torch.Tensor`): [B, N, 3]
        Returns:
            `torch.Tensor`: [B, K, 2, 3] where K is the number of bounding boxes
        """
        if self.bbox_predictor is None:
            raise ValueError("bbox_predictor is not set.")
        aabb, face_ids, mesh = self.bbox_predictor.predict_aabb(
            mesh, post_process=True, seed=seed
        )
        # aabb, face_ids, mesh = self.bbox_predictor.predict_aabb(mesh, post_process=False)
        aabb = torch.from_numpy(aabb)
        return aabb

    def prepare_latents(
        self, batch_size, latent_shape, dtype, device, generator, latents=None
    ):
        # prepare latents for different parts
        shape = (batch_size, *latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but"
                f" requested an effective batch size of {batch_size}. Make sure the"
                " batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, "init_noise_sigma", 1.0)
        return latents

    @synchronize_timer("Encode cond")
    def encode_cond(
        self,
        part_surface_inbbox,
        object_surface,
        do_classifier_free_guidance,
    ):
        bsz = object_surface.shape[0]
        cond = self.conditioner(part_surface_inbbox, object_surface)

        if do_classifier_free_guidance:
            # TODO: do_classifier_free_guidance, un_cond
            un_cond = {k: torch.zeros_like(v) for k, v in cond.items()}

            def cat_recursive(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.cat([a, b], dim=0).to(self.dtype)
                out = {}
                for k in a.keys():
                    out[k] = cat_recursive(a[k], b[k])
                return out

            cond = cat_recursive(cond, un_cond)
        return cond

    def normalize_mesh(self, mesh):
        vertices = mesh.vertices
        min_xyz = np.min(vertices, axis=0)
        max_xyz = np.max(vertices, axis=0)
        center = (min_xyz + max_xyz) / 2.0
        # scale = np.max(np.linalg.norm(vertices - center, axis=1))
        scale = np.max(max_xyz - min_xyz) / 2 / 0.8
        vertices = (vertices - center) / scale
        mesh.vertices = vertices
        return mesh, center, scale

    # Text2Part v2 patch: split the original check_inputs path into CPU prep,
    # bbox prediction, and CPU bbox sampling stages.
    def prepare_inputs_cpu(
        self,
        obj_surface=None,
        obj_surface_raw=None,
        mesh_path=None,
        mesh=None,
        seed=42,
    ) -> PreparedInputs:
        """CPU-only mesh load, normalization, and object-surface sampling."""
        if obj_surface is None:
            if mesh_path is None and (mesh is None and obj_surface_raw is None):
                raise ValueError(
                    "obj_surface or mesh_path/mesh/obj_surface_raw must be provided."
                )
        center = np.zeros(3)
        scale = 1.0

        if obj_surface is None:
            if obj_surface_raw is None:
                if mesh is not None:
                    obj_surface_raw = SampleMesh(
                        mesh.vertices, mesh.faces, -1, seed=seed
                    )
                elif mesh_path is not None:
                    mesh = trimesh.load(mesh_path, force="mesh")
                    mesh, center, scale = self.normalize_mesh(mesh)
                    print(f"Normalize mesh: {center}, {scale}")
                    obj_surface_raw = SampleMesh(
                        mesh.vertices, mesh.faces, -1, seed=seed
                    )
                else:
                    raise ValueError("obj_surface or mesh_path/mesh must be provided.")
            rng = np.random.default_rng(seed=seed)
            obj_surface, _ = load_surface_points(
                rng,
                obj_surface_raw["random_surface"],
                obj_surface_raw["sharp_surface"],
                pc_size=81920,
                pc_sharpedge_size=0,
                return_sharpedge_label=True,
                return_normal=True,
            )
            obj_surface = obj_surface.unsqueeze(0)
        return PreparedInputs(
            obj_surface=obj_surface,
            obj_surface_raw=obj_surface_raw,
            mesh=mesh,
            center=center,
            scale=scale,
            seed=seed,
        )

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def predict_bboxes_gpu(
        self,
        prepared: PreparedInputs,
        *,
        aabb=None,
        seed=42,
    ) -> BBoxInputs:
        """Run or normalize bbox prediction and return unbatched CPU bboxes."""
        if aabb is None:
            if prepared.mesh is None:
                raise ValueError("mesh is required when aabb is not provided.")
            aabb = self.predict_bbox(prepared.mesh, seed=seed)
            print(f"Get bbox from bbox_predictor: {aabb.shape}")
        else:
            if isinstance(aabb, np.ndarray):
                aabb = torch.from_numpy(aabb)
            else:
                aabb = aabb.detach().cpu()
            # normalize aabb by mesh scale and center
            aabb = aabb.float()
            aabb = (aabb - torch.from_numpy(prepared.center).float()) / prepared.scale

        if aabb.ndim == 4:
            if aabb.shape[0] != 1:
                raise AssertionError("Batch size > 1 is not supported yet.")
            aabb = aabb[0]
        return BBoxInputs(prepared=prepared, aabb=aabb.float().cpu())

    @staticmethod
    def _valid_bbox_mask(mesh: trimesh.Trimesh, aabb: torch.Tensor) -> torch.Tensor:
        _faces = mesh.faces
        _vertices = mesh.vertices
        _faces = np.reshape(_faces, (-1))
        num_parts = aabb.shape[0]
        if num_parts == 0:
            return torch.zeros((0,), dtype=torch.bool)
        _points = torch.from_numpy(_vertices[_faces])
        _part_mask = torch.all(
            (_points[None, :, :3] >= aabb[:, :1])
            & (_points[None, :, :3] <= aabb[:, 1:]),
            dim=-1,
        )
        _part_mask = torch.any(torch.reshape(_part_mask, (num_parts, -1, 3)), dim=-1)
        return torch.tensor(
            [len(torch.nonzero(x).squeeze(-1)) > 0 for x in _part_mask],
            dtype=torch.bool,
            device=_points.device,
        )

    def finish_inputs_cpu(
        self,
        bbox_inputs: BBoxInputs,
        *,
        part_surface_inbbox=None,
        seed=42,
    ) -> PipelineInputs:
        """CPU-only sampling of object points inside each bbox."""
        prepared = bbox_inputs.prepared
        aabb = bbox_inputs.aabb
        if part_surface_inbbox is None:
            if prepared.mesh is None:
                raise ValueError(
                    "mesh is required when part_surface_inbbox is not provided."
                )
            valid_parts_mask = self._valid_bbox_mask(prepared.mesh, aabb)
            aabb = aabb[valid_parts_mask]
            if aabb.shape[0] == 0:
                part_surface_inbbox = torch.empty((1, 0, 81920, 7))
                aabb = aabb.unsqueeze(0)
            else:
                part_surface_inbbox, valid_parts_mask = sample_bbox_points_from_trimesh(
                    prepared.mesh, aabb, num_points=81920, seed=seed
                )
                aabb = aabb[valid_parts_mask]
                aabb = aabb.unsqueeze(0)
                part_surface_inbbox = part_surface_inbbox.unsqueeze(0)
        else:
            if isinstance(part_surface_inbbox, np.ndarray):
                part_surface_inbbox = torch.from_numpy(part_surface_inbbox)
            if aabb.ndim == 3:
                aabb = aabb.unsqueeze(0)
            if part_surface_inbbox.ndim == 3:
                part_surface_inbbox = part_surface_inbbox.unsqueeze(0)

        return PipelineInputs(
            obj_surface=prepared.obj_surface,
            aabb=aabb,
            part_surface_inbbox=part_surface_inbbox,
            mesh=prepared.mesh,
            center=prepared.center,
            scale=prepared.scale,
            raw_bbox_count=bbox_inputs.n_bbox,
            valid_bbox_count=int(aabb.shape[1]),
        )

    def check_inputs(
        self,
        obj_surface=None,
        obj_surface_raw=None,
        mesh_path=None,
        mesh=None,
        aabb=None,
        part_surface_inbbox=None,
        seed=42,
    ):
        """
        Check the inputs of the pipeline.
        Args:
            obj_surface (`torch.Tensor`): [B, N, 3+3+1]
            mesh_path (`str`): path to the mesh file
            mesh (`trimesh.Trimesh`): mesh object
            aabb (`torch.Tensor`): [B, K, 2, 3]
            part_surface_inbbox (`torch.Tensor`): [B, K,N, 3+3+1]
        """
        if obj_surface is not None:
            if aabb is None or part_surface_inbbox is None:
                raise ValueError(
                    "aabb and part_surface_inbbox must be provided if obj_surface is"
                    " provided."
                )
            assert aabb.shape[0] == part_surface_inbbox.shape[0], "Batch size mismatch."

        prepared = self.prepare_inputs_cpu(
            obj_surface=obj_surface,
            obj_surface_raw=obj_surface_raw,
            mesh_path=mesh_path,
            mesh=mesh,
            seed=seed,
        )
        bbox_inputs = self.predict_bboxes_gpu(prepared, aabb=aabb, seed=seed)
        inputs = self.finish_inputs_cpu(
            bbox_inputs,
            part_surface_inbbox=part_surface_inbbox,
            seed=seed,
        )
        return (
            inputs.obj_surface,
            inputs.aabb,
            inputs.part_surface_inbbox,
            inputs.mesh,
            inputs.center,
            inputs.scale,
        )

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def _export(
        self,
        latents,
        output_type="trimesh",
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo="mc",
        enable_pbar=True,
        **kwargs,
    ):
        if not output_type == "latent":
            latents = 1.0 / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latent2mesh_2(
                # outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                octree_depth=8,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_mode=mc_algo,
                # enable_pbar=enable_pbar,
                **kwargs,
            )
        else:
            outputs = latents

        if output_type == "trimesh":
            outputs = export_to_trimesh(outputs)

        return outputs

    # Text2Part v2 patch: keep the GPU diffusion block callable without mesh
    # preparation/export so the scheduler can profile and short-circuit passes.
    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def run_diffusion_gpu(
        self,
        inputs: PipelineInputs,
        *,
        generator,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        eta: float = 0.0,
        guidance_scale: float = -1.0,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        callback=None,
        callback_steps=None,
        enable_pbar=True,
    ):
        """Move prepared inputs to GPU and run condition encoding + diffusion."""
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True
        )
        device = self.device
        dtype = self.dtype

        obj_surface = inputs.obj_surface.to(device=device, dtype=dtype)
        aabb = inputs.aabb.to(device=device, dtype=dtype)
        part_surface_inbbox = inputs.part_surface_inbbox.to(device=device, dtype=dtype)
        batch_size, num_parts, N, dim = part_surface_inbbox.shape
        # TODO: support batch size > 1
        assert batch_size == 1, "Batch size > 1 is not supported yet."

        num_tokens = torch.tensor(
            [self.allocate_tokens(x, self.vae.latent_shape[0]) for x in aabb],
            device=device,
        )
        latent_shape = self.vae.latent_shape
        latents = self.prepare_latents(
            num_parts, latent_shape, dtype, device, generator
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        cond = self.encode_cond(
            part_surface_inbbox.reshape(batch_size * num_parts, N, dim),
            obj_surface.expand(batch_size * num_parts, -1, -1),
            do_classifier_free_guidance,
        )

        guidance_cond = None
        if getattr(self.model, "guidance_cond_proj_dim", None) is not None:
            logger.info("Using lcm guidance scale")
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        if sigmas is None and timesteps is None:
            sigmas = np.linspace(0, 1, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        torch.cuda.empty_cache()

        with synchronize_timer("Diffusion Sampling"):
            for i, t in enumerate(
                tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")
            ):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    aabb_model_input = torch.repeat_interleave(aabb, 2, dim=0)
                else:
                    latent_model_input = latents
                    aabb_model_input = aabb

                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(
                    latent_model_input,
                    timestep,
                    cond,
                    aabb=aabb_model_input,
                    num_tokens=num_tokens,
                    guidance_cond=guidance_cond,
                )

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return latents, {
            "num_parts": num_parts,
            "num_inference_steps": num_inference_steps,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "raw_bbox_count": inputs.raw_bbox_count,
            "valid_bbox_count": inputs.valid_bbox_count,
        }

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def export_latents(
        self,
        latents,
        *,
        center: np.ndarray,
        scale: float,
        output_type: Optional[str] = "trimesh",
        box_v=1.01,
        octree_resolution=512,
        mc_level=-1 / 512,
        num_chunks=400000,
        mc_algo="mc",
        enable_pbar=True,
    ):
        """Export part latents and denormalize them into a trimesh.Scene."""
        out = trimesh.Scene()
        for i, part_latent in enumerate(latents):
            try:
                part_mesh = self._export(
                    latents=part_latent.unsqueeze(0),
                    output_type=output_type,
                    box_v=box_v,
                    mc_level=mc_level,
                    num_chunks=num_chunks,
                    octree_resolution=octree_resolution,
                    mc_algo=mc_algo,
                    enable_pbar=enable_pbar,
                )[0]
                out.add_geometry(part_mesh)
                random_color = np.random.randint(0, 255, size=3)
                part_mesh.visual.face_colors = random_color
            except Exception as e:
                logger.error(f"Failed to export part {i} with error {e}")
        print(f"Denormalize mesh: {center}, {scale}")
        for key in out.geometry.keys():
            _v = out.geometry[key].vertices
            _v = _v * scale + center
            out.geometry[key].vertices = _v
        return out

    def _bbox_debug_scene(self, inputs: PipelineInputs):
        mesh_bbox = trimesh.Scene()
        if inputs.mesh is not None:
            mesh_bbox.add_geometry(inputs.mesh)
        else:
            mesh = trimesh.points.PointCloud(
                inputs.obj_surface[0, :, :3].float().cpu().numpy()
            )
            mesh_bbox.add_geometry(mesh)
        for bbox in inputs.aabb[0]:
            box = trimesh.path.creation.box_outline()
            box.vertices *= (bbox[1] - bbox[0]).float().cpu().numpy()
            box.vertices += (bbox[0] + bbox[1]).float().cpu().numpy() / 2
            mesh_bbox.add_geometry(box)
        return mesh_bbox

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def __call__(
        self,
        obj_surface=None,
        obj_surface_raw=None,
        mesh_path=None,
        mesh=None,
        aabb=None,
        part_surface_inbbox=None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        eta: float = 0.0,
        # guidance_scale: float = 7.5,
        guidance_scale: float = -1.0,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        seed=42,
        # marching cubes
        box_v=1.01,
        octree_resolution=512,
        mc_level=-1 / 512,
        num_chunks=400000,
        mc_algo="mc",
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ):
        """
        Args:
            obj_surface (`torch.Tensor`): [B, N, 3+3+1]
            aabb (`torch.Tensor`): [B, K, 2, 3]
            part_surface_inbbox (`torch.Tensor`): [B, K,N, 3+3+1]
        Returns:
            `trimesh.Scene` : single object composed of multiple parts
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if obj_surface is not None:
            if aabb is None or part_surface_inbbox is None:
                raise ValueError(
                    "aabb and part_surface_inbbox must be provided if obj_surface is"
                    " provided."
                )
            assert aabb.shape[0] == part_surface_inbbox.shape[0], "Batch size mismatch."

        prepared = self.prepare_inputs_cpu(
            obj_surface=obj_surface,
            obj_surface_raw=obj_surface_raw,
            mesh_path=mesh_path,
            mesh=mesh,
            seed=seed,
        )
        bbox_inputs = self.predict_bboxes_gpu(prepared, aabb=aabb, seed=seed)
        inputs = self.finish_inputs_cpu(
            bbox_inputs,
            part_surface_inbbox=part_surface_inbbox,
            seed=seed,
        )

        if self.verbose:
            mesh_bbox = self._bbox_debug_scene(inputs)

        latents, _meta = self.run_diffusion_gpu(
            inputs,
            generator=generator,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            eta=eta,
            guidance_scale=guidance_scale,
            dual_guidance_scale=dual_guidance_scale,
            dual_guidance=dual_guidance,
            callback=callback,
            callback_steps=callback_steps,
            enable_pbar=enable_pbar,
        )
        out = self.export_latents(
            latents,
            center=inputs.center,
            scale=inputs.scale,
            output_type=output_type,
            box_v=box_v,
            octree_resolution=octree_resolution,
            mc_level=mc_level,
            num_chunks=num_chunks,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )

        if self.verbose:
            explode_object = explode_mesh(copy.deepcopy(out), explosion_scale=0.2)
            out_bbox = trimesh.Scene()
            out_bbox.add_geometry(out)
            for bbox in inputs.aabb[0]:
                box = trimesh.path.creation.box_outline()
                box.vertices *= (bbox[1] - bbox[0]).float().cpu().numpy()
                box.vertices += (bbox[0] + bbox[1]).float().cpu().numpy() / 2
                box.vertices = box.vertices * inputs.scale + inputs.center
                out_bbox.add_geometry(box)
            return out, (out_bbox, mesh_bbox, explode_object)
        else:
            # return only the generated mesh
            return out, None
