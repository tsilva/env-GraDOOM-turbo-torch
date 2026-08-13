"""CUDA kernels for branch-heavy engine primitives."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_FAST_NATIVE_PORTAL_LAYERS = 16
_FAST_NATIVE_SPRITE_LAYERS = 1


@triton.jit
def _bounded_observation_augment_kernel(
    observations_ptr,
    randoms_ptr,
    output_ptr,
    observation_stride_n: tl.constexpr,
    observation_stride_c: tl.constexpr,
    observation_stride_y: tl.constexpr,
    observation_stride_x: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    total_pixels: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Apply one bounded spatial/photometric transform per frame stack."""

    output_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixels_per_env = channels * height * width
    env = output_offset // pixels_per_env
    within_env = output_offset - env * pixels_per_env
    channel = within_env // (height * width)
    within_channel = within_env - channel * height * width
    output_y = within_channel // width
    output_x = within_channel - output_y * width
    output_valid = output_offset < total_pixels

    random_base = env * 4
    random_x = tl.load(randoms_ptr + random_base, mask=output_valid, other=0.5)
    random_y = tl.load(randoms_ptr + random_base + 1, mask=output_valid, other=0.5)
    random_gain = tl.load(randoms_ptr + random_base + 2, mask=output_valid, other=0.5)
    random_bias = tl.load(randoms_ptr + random_base + 3, mask=output_valid, other=0.5)
    shift_x = (random_x * 5.0).to(tl.int32) - 2
    shift_y = (random_y * 5.0).to(tl.int32) - 2
    gain = 0.9 + random_gain * 0.2
    bias = -8.0 + random_bias * 16.0

    source_x = output_x - shift_x
    source_y = output_y - shift_y
    source_valid = (
        output_valid
        & (source_x >= 0)
        & (source_x < width)
        & (source_y >= 0)
        & (source_y < height)
    )
    source_offset = (
        env * observation_stride_n
        + channel * observation_stride_c
        + source_y * observation_stride_y
        + source_x * observation_stride_x
    )
    pixel = tl.load(observations_ptr + source_offset, mask=source_valid, other=0.0).to(
        tl.float32
    )
    adjusted = tl.maximum(0.0, tl.minimum(255.0, pixel * gain + bias))
    adjusted = tl.where(source_valid, tl.floor(adjusted + 0.5), 0.0)
    tl.store(output_ptr + output_offset, adjusted, mask=output_valid)


@torch.library.custom_op("gradoom::bounded_observation_augment", mutates_args=())
def bounded_observation_augment(
    observations: torch.Tensor,
    randoms: torch.Tensor,
) -> torch.Tensor:
    """Apply bounded stack-consistent jitter to CUDA uint8 policy observations.

    ``randoms`` has four uniform values per environment. They select integer x/y
    shifts in ``[-2, 2]``, grayscale gain in ``[0.9, 1.1)``, and bias in
    ``[-8, 8)``. The same transform is applied to all frames in each stack.
    """

    if observations.ndim != 4:
        raise ValueError("observations must have shape (N, C, H, W)")
    if observations.dtype != torch.uint8:
        raise TypeError("observations must be uint8")
    if randoms.shape != (observations.shape[0], 4):
        raise ValueError("randoms must have shape (N, 4)")
    if randoms.dtype != torch.float32:
        raise TypeError("randoms must be float32")
    if observations.device != randoms.device:
        raise ValueError("observations and randoms must be on the same device")
    if observations.device.type != "cuda":
        raise ValueError("bounded observation augmentation requires CUDA tensors")
    output = torch.empty_like(observations, memory_format=torch.contiguous_format)
    total_pixels = observations.numel()
    block = 256
    torch.library.wrap_triton(_bounded_observation_augment_kernel)[
        (triton.cdiv(total_pixels, block),)
    ](
        observations,
        randoms,
        output,
        *observations.stride(),
        observations.shape[1],
        observations.shape[2],
        observations.shape[3],
        total_pixels,
        block,
        num_warps=4,
    )
    return output


@bounded_observation_augment.register_fake
def _bounded_observation_augment_fake(
    observations: torch.Tensor,
    randoms: torch.Tensor,
) -> torch.Tensor:
    del randoms
    return torch.empty_like(observations, memory_format=torch.contiguous_format)


@triton.jit
def _policy_area_grayscale_kernel(
    indexed_ptr,
    palette_ptr,
    output_ptr,
    source_height: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
    SAMPLE_BLOCK: tl.constexpr,
):
    """Exact 320x240 -> 84x84 ViZDoom-turbo indexed-area conversion."""

    output_pixels = 84 * 84
    program = tl.program_id(0)
    env = program // tl.cdiv(output_pixels, OUTPUT_BLOCK)
    output_block = program % tl.cdiv(output_pixels, OUTPUT_BLOCK)
    output_offset = output_block * OUTPUT_BLOCK + tl.arange(0, OUTPUT_BLOCK)
    output_valid = output_offset < output_pixels
    output_y = output_offset // 84
    output_x = output_offset % 84

    sample = tl.arange(0, SAMPLE_BLOCK)
    sample_y = sample // 5
    sample_x = sample % 5
    y_start = output_y[:, None] * 240
    y_end = (output_y[:, None] + 1) * 240
    x_start = output_x[:, None] * 320
    x_end = (output_x[:, None] + 1) * 320
    source_y = y_start // 84 + sample_y[None, :]
    source_x = x_start // 84 + sample_x[None, :]
    y_overlap = tl.minimum(y_end, (source_y + 1) * 84) - tl.maximum(
        y_start, source_y * 84
    )
    x_overlap = tl.minimum(x_end, (source_x + 1) * 84) - tl.maximum(
        x_start, source_x * 84
    )
    sample_valid = (
        output_valid[:, None]
        & (source_y >= 0)
        & (source_y < source_height)
        & (source_x >= 0)
        & (source_x < 320)
        & (y_overlap > 0)
        & (x_overlap > 0)
    )
    weight = (y_overlap * x_overlap) // 48
    source_offset = env * source_height * 320 + source_y * 320 + source_x
    palette_index = tl.load(indexed_ptr + source_offset, mask=sample_valid, other=0).to(tl.int32)
    red = tl.load(palette_ptr + palette_index * 3, mask=sample_valid, other=0).to(tl.int32)
    green = tl.load(palette_ptr + palette_index * 3 + 1, mask=sample_valid, other=0).to(
        tl.int32
    )
    blue = tl.load(palette_ptr + palette_index * 3 + 2, mask=sample_valid, other=0).to(
        tl.int32
    )
    red_sum = tl.sum(red * weight, axis=1)
    green_sum = tl.sum(green * weight, axis=1)
    blue_sum = tl.sum(blue * weight, axis=1)
    red_pooled = (red_sum + 800) // 1600
    green_pooled = (green_sum + 800) // 1600
    blue_pooled = (blue_sum + 800) // 1600
    grayscale = (red_pooled * 77 + green_pooled * 150 + blue_pooled * 29 + 128) // 256
    tl.store(
        output_ptr + env * output_pixels + output_offset,
        grayscale,
        mask=output_valid,
    )


@torch.library.custom_op("gradoom::policy_area_grayscale", mutates_args=())
def policy_area_grayscale(indexed: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Apply the pinned ViZDoom-turbo area/grayscale transform on CUDA."""

    if indexed.ndim != 3 or indexed.shape[1:] not in ((208, 320), (240, 320)):
        raise ValueError("indexed must have shape (N, 208, 320) or (N, 240, 320)")
    if indexed.dtype != torch.uint8 or palette.dtype != torch.uint8:
        raise TypeError("indexed and palette must be uint8 tensors")
    if palette.shape != (256, 3):
        raise ValueError("palette must have shape (256, 3)")
    output = torch.empty((indexed.shape[0], 84, 84), device=indexed.device, dtype=torch.uint8)
    output_block = 32
    sample_block = 32
    grid = (indexed.shape[0] * triton.cdiv(84 * 84, output_block),)
    torch.library.wrap_triton(_policy_area_grayscale_kernel)[grid](
        indexed,
        palette,
        output,
        indexed.shape[1],
        output_block,
        sample_block,
        num_warps=4,
    )
    return output


@policy_area_grayscale.register_fake
def _policy_area_grayscale_fake(
    indexed: torch.Tensor,
    palette: torch.Tensor,
) -> torch.Tensor:
    del palette
    return torch.empty((indexed.shape[0], 84, 84), device=indexed.device, dtype=torch.uint8)


@triton.jit
def _render_fast_native_flats_kernel(
    player_x,
    player_y,
    player_angle,
    view_z,
    center,
    ray_offsets,
    floor_plane_heights,
    ceiling_plane_heights,
    sector_lookup,
    lookup_metadata,
    sector_heights,
    sector_floor_texture_ids,
    sector_ceiling_texture_ids,
    texture_widths,
    texture_heights,
    texture_index_atlas,
    sector_lights,
    flash_light,
    colormap,
    frame,
    surface_depth,
    env_count: tl.constexpr,
    view_height: tl.constexpr,
    view_width: tl.constexpr,
    floor_plane_count: tl.constexpr,
    ceiling_plane_count: tl.constexpr,
    lookup_height: tl.constexpr,
    lookup_width: tl.constexpr,
    atlas_stride_texture: tl.constexpr,
    atlas_stride_y: tl.constexpr,
    atlas_stride_x: tl.constexpr,
    block: tl.constexpr,
):
    """Resolve deathmatch visplanes with a compact world-space sector LUT."""

    offset = tl.program_id(0) * block + tl.arange(0, block)
    pixels_per_env = view_height * view_width
    total_pixels = env_count * pixels_per_env
    valid = offset < total_pixels
    env = offset // pixels_per_env
    pixel = offset - env * pixels_per_env
    pixel_y = pixel // view_width
    pixel_x = pixel - pixel_y * view_width

    current_x = tl.load(player_x + env, mask=valid, other=0.0)
    current_y = tl.load(player_y + env, mask=valid, other=0.0)
    current_angle = tl.load(player_angle + env, mask=valid, other=0.0)
    current_view_z = tl.load(view_z + env, mask=valid, other=41.0)
    current_center = tl.load(center + env, mask=valid, other=103.5)
    ray_offset = tl.load(ray_offsets + pixel_x, mask=valid, other=0.0)
    ray_angle = current_angle + ray_offset
    ray_cos = tl.cos(ray_angle)
    ray_sin = tl.sin(ray_angle)
    cosine_correction = tl.maximum(tl.cos(ray_offset), 1.0e-4)
    row_delta = pixel_y.to(tl.float32) - current_center
    floor_pixel = row_delta > 0.0
    denominator = tl.maximum(tl.abs(row_delta), 0.5)
    focal_y = 192.0
    origin_x = tl.load(lookup_metadata)
    origin_y = tl.load(lookup_metadata + 1)
    cell_size = tl.load(lookup_metadata + 2)
    player_lookup_x = tl.floor((current_x - origin_x) / cell_size).to(tl.int64)
    player_lookup_y = tl.floor((current_y - origin_y) / cell_size).to(tl.int64)
    player_in_lookup = (
        (player_lookup_x >= 0)
        & (player_lookup_x < lookup_width)
        & (player_lookup_y >= 0)
        & (player_lookup_y < lookup_height)
    )
    fallback_sector = tl.load(
        sector_lookup + player_lookup_y * lookup_width + player_lookup_x,
        mask=valid & player_in_lookup,
        other=0,
    ).to(tl.int64)
    fallback_sector = tl.maximum(fallback_sector, 0)

    best_distance = tl.full((block,), float("inf"), tl.float32)
    selected_sector = fallback_sector

    for plane_index in tl.static_range(floor_plane_count):
        plane_z = tl.load(floor_plane_heights + plane_index)
        plane_height = current_view_z - plane_z
        perpendicular_depth = plane_height * focal_y / denominator
        ray_distance = perpendicular_depth / cosine_correction
        world_x = current_x + ray_cos * ray_distance
        world_y = current_y + ray_sin * ray_distance
        lookup_x = tl.floor((world_x - origin_x) / cell_size).to(tl.int64)
        lookup_y = tl.floor((world_y - origin_y) / cell_size).to(tl.int64)
        in_lookup = (
            (lookup_x >= 0)
            & (lookup_x < lookup_width)
            & (lookup_y >= 0)
            & (lookup_y < lookup_height)
        )
        sector = tl.load(
            sector_lookup + lookup_y * lookup_width + lookup_x,
            mask=valid & in_lookup,
            other=-1,
        ).to(tl.int64)
        safe_sector = tl.maximum(sector, 0)
        sector_floor = tl.load(sector_heights + safe_sector * 2)
        candidate = (
            valid
            & floor_pixel
            & in_lookup
            & (sector >= 0)
            & (tl.abs(sector_floor - plane_z) < 0.01)
            & (plane_height > 0.0)
            & (ray_distance > 0.0)
            & (ray_distance < best_distance)
        )
        best_distance = tl.where(candidate, ray_distance, best_distance)
        selected_sector = tl.where(candidate, sector, selected_sector)

    for plane_index in tl.static_range(ceiling_plane_count):
        plane_z = tl.load(ceiling_plane_heights + plane_index)
        plane_height = plane_z - current_view_z
        perpendicular_depth = plane_height * focal_y / denominator
        ray_distance = perpendicular_depth / cosine_correction
        world_x = current_x + ray_cos * ray_distance
        world_y = current_y + ray_sin * ray_distance
        lookup_x = tl.floor((world_x - origin_x) / cell_size).to(tl.int64)
        lookup_y = tl.floor((world_y - origin_y) / cell_size).to(tl.int64)
        in_lookup = (
            (lookup_x >= 0)
            & (lookup_x < lookup_width)
            & (lookup_y >= 0)
            & (lookup_y < lookup_height)
        )
        sector = tl.load(
            sector_lookup + lookup_y * lookup_width + lookup_x,
            mask=valid & in_lookup,
            other=-1,
        ).to(tl.int64)
        safe_sector = tl.maximum(sector, 0)
        sector_ceiling = tl.load(sector_heights + safe_sector * 2 + 1)
        candidate = (
            valid
            & ~floor_pixel
            & in_lookup
            & (sector >= 0)
            & (tl.abs(sector_ceiling - plane_z) < 0.01)
            & (plane_height > 0.0)
            & (ray_distance > 0.0)
            & (ray_distance < best_distance)
        )
        best_distance = tl.where(candidate, ray_distance, best_distance)
        selected_sector = tl.where(candidate, sector, selected_sector)

    selected_floor = tl.load(sector_heights + selected_sector * 2)
    selected_ceiling = tl.load(sector_heights + selected_sector * 2 + 1)
    selected_plane_height = tl.where(
        floor_pixel,
        current_view_z - selected_floor,
        selected_ceiling - current_view_z,
    )
    fallback_distance = (
        selected_plane_height * focal_y / denominator / cosine_correction
    )
    ray_distance = tl.where(best_distance == float("inf"), fallback_distance, best_distance)
    world_x = current_x + ray_cos * ray_distance
    world_y = current_y + ray_sin * ray_distance
    floor_texture = tl.load(sector_floor_texture_ids + selected_sector)
    ceiling_texture = tl.load(sector_ceiling_texture_ids + selected_sector)
    texture_id = tl.where(floor_pixel, floor_texture, ceiling_texture).to(tl.int64)
    texture_width = tl.load(texture_widths + texture_id).to(tl.int64)
    texture_height = tl.load(texture_heights + texture_id).to(tl.int64)
    texture_u = tl.floor(world_x).to(tl.int64) % texture_width
    texture_v = tl.floor(-world_y).to(tl.int64) % texture_height
    texture_u = tl.where(texture_u < 0, texture_u + texture_width, texture_u)
    texture_v = tl.where(texture_v < 0, texture_v + texture_height, texture_v)
    palette_index = tl.load(
        texture_index_atlas
        + texture_id * atlas_stride_texture
        + texture_v * atlas_stride_y
        + texture_u * atlas_stride_x,
        mask=valid,
        other=0,
    ).to(tl.int64)
    light = tl.load(sector_lights + selected_sector).to(tl.float32)
    light += tl.load(flash_light + env, mask=valid, other=0).to(tl.float32) * 16.0
    base_shade = 61.0 - light / 4.0
    row_distance = tl.abs(current_center + 0.5 - pixel_y.to(tl.float32))
    visibility = tl.minimum(
        24.0,
        6.6666565 * row_distance / tl.maximum(selected_plane_height, 1.0e-4),
    )
    shade = tl.maximum(0, tl.minimum(31, tl.floor(base_shade - visibility))).to(tl.int64)
    lit_index = tl.load(colormap + shade * 256 + palette_index).to(tl.uint8)
    tl.store(frame + offset, lit_index, mask=valid)
    resolved_depth = tl.where(
        best_distance == float("inf"),
        float("inf"),
        best_distance * cosine_correction,
    )
    tl.store(surface_depth + offset, resolved_depth, mask=valid)


@torch.library.custom_op(
    "gradoom::render_fast_native_flats",
    mutates_args=(),
    device_types="cuda",
)
def render_fast_native_flats(
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    ray_offsets: torch.Tensor,
    floor_plane_heights: torch.Tensor,
    ceiling_plane_heights: torch.Tensor,
    sector_lookup: torch.Tensor,
    lookup_metadata: torch.Tensor,
    sector_heights: torch.Tensor,
    sector_floor_texture_ids: torch.Tensor,
    sector_ceiling_texture_ids: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    sector_lights: torch.Tensor,
    flash_light: torch.Tensor,
    colormap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a native-resolution indexed visplane approximation in one launch."""

    view_height = 208
    view_width = ray_offsets.shape[0]
    frame = torch.empty(
        (player_x.shape[0], view_height, view_width),
        device=player_x.device,
        dtype=torch.uint8,
    )
    surface_depth = torch.empty(
        frame.shape,
        device=player_x.device,
        dtype=torch.float32,
    )
    block = 1024
    total_pixels = frame.numel()
    torch.library.wrap_triton(_render_fast_native_flats_kernel)[
        (triton.cdiv(total_pixels, block),)
    ](
        player_x,
        player_y,
        player_angle,
        view_z,
        center,
        ray_offsets,
        floor_plane_heights,
        ceiling_plane_heights,
        sector_lookup,
        lookup_metadata,
        sector_heights,
        sector_floor_texture_ids,
        sector_ceiling_texture_ids,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        sector_lights,
        flash_light,
        colormap,
        frame,
        surface_depth,
        player_x.shape[0],
        view_height,
        view_width,
        floor_plane_heights.shape[0],
        ceiling_plane_heights.shape[0],
        sector_lookup.shape[0],
        sector_lookup.shape[1],
        texture_index_atlas.stride(0),
        texture_index_atlas.stride(1),
        texture_index_atlas.stride(2),
        block,
        num_warps=8,
    )
    return frame, surface_depth


@render_fast_native_flats.register_fake
def _render_fast_native_flats_fake(
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    ray_offsets: torch.Tensor,
    floor_plane_heights: torch.Tensor,
    ceiling_plane_heights: torch.Tensor,
    sector_lookup: torch.Tensor,
    lookup_metadata: torch.Tensor,
    sector_heights: torch.Tensor,
    sector_floor_texture_ids: torch.Tensor,
    sector_ceiling_texture_ids: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    sector_lights: torch.Tensor,
    flash_light: torch.Tensor,
    colormap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        player_y,
        player_angle,
        view_z,
        center,
        floor_plane_heights,
        ceiling_plane_heights,
        sector_lookup,
        lookup_metadata,
        sector_heights,
        sector_floor_texture_ids,
        sector_ceiling_texture_ids,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        sector_lights,
        flash_light,
        colormap,
    )
    shape = (player_x.shape[0], 208, ray_offsets.shape[0])
    return (
        torch.empty(shape, device=player_x.device, dtype=torch.uint8),
        torch.empty(shape, device=player_x.device, dtype=torch.float32),
    )


@triton.jit
def _frozen_nature_conv1_kernel(
    observations,
    weight,
    bias,
    output,
    observation_stride_n: tl.constexpr,
    observation_stride_c: tl.constexpr,
    observation_stride_y: tl.constexpr,
    observation_stride_x: tl.constexpr,
    weight_stride_n: tl.constexpr,
    weight_stride_c: tl.constexpr,
    weight_stride_y: tl.constexpr,
    weight_stride_x: tl.constexpr,
    output_stride_n: tl.constexpr,
    output_stride_c: tl.constexpr,
    output_stride_y: tl.constexpr,
    output_stride_x: tl.constexpr,
    batch_size: tl.constexpr,
    output_channels: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Fused uint8 normalization and fixed NatureCNN 8x8/4 convolution."""

    output_width: tl.constexpr = 20
    output_pixels: tl.constexpr = output_width * output_width
    reduction_size: tl.constexpr = 4 * 8 * 8
    output_row = tl.program_id(0) * block_m + tl.arange(0, block_m)
    output_channel = tl.program_id(1) * block_n + tl.arange(0, block_n)
    valid_row = output_row < batch_size * output_pixels
    valid_channel = output_channel < output_channels
    env = output_row // output_pixels
    pixel = output_row - env * output_pixels
    output_y = pixel // output_width
    output_x = pixel - output_y * output_width
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_base in tl.static_range(0, reduction_size, block_k):
        reduction = k_base + tl.arange(0, block_k)
        input_channel = reduction // 64
        kernel_pixel = reduction - input_channel * 64
        kernel_y = kernel_pixel // 8
        kernel_x = kernel_pixel - kernel_y * 8
        observation_index = (
            env[:, None] * observation_stride_n
            + input_channel[None, :] * observation_stride_c
            + (output_y[:, None] * 4 + kernel_y[None, :]) * observation_stride_y
            + (output_x[:, None] * 4 + kernel_x[None, :]) * observation_stride_x
        )
        weight_index = (
            output_channel[None, :] * weight_stride_n
            + input_channel[:, None] * weight_stride_c
            + kernel_y[:, None] * weight_stride_y
            + kernel_x[:, None] * weight_stride_x
        )
        image = tl.load(
            observations + observation_index,
            mask=valid_row[:, None],
            other=0,
        ).to(tl.float32)
        image *= 1.0 / 255.0
        kernel = tl.load(
            weight + weight_index,
            mask=valid_channel[None, :],
            other=0.0,
        )
        accumulator = tl.dot(image, kernel, accumulator, input_precision="tf32")

    values = tl.maximum(
        accumulator
        + tl.load(
            bias + output_channel[None, :],
            mask=valid_channel[None, :],
            other=0.0,
        ),
        0.0,
    )
    output_index = (
        env[:, None] * output_stride_n
        + output_channel[None, :] * output_stride_c
        + output_y[:, None] * output_stride_y
        + output_x[:, None] * output_stride_x
    )
    tl.store(
        output + output_index,
        values,
        mask=valid_row[:, None] & valid_channel[None, :],
    )


@torch.library.custom_op(
    "gradoom::frozen_nature_conv1",
    mutates_args=(),
    device_types="cuda",
)
def frozen_nature_conv1(
    observations: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Run NatureCNN's frozen first convolution and ReLU directly from uint8 input."""

    batch_size = observations.shape[0]
    output_channels = weight.shape[0]
    output = torch.empty(
        (batch_size, output_channels, 20, 20),
        dtype=weight.dtype,
        device=observations.device,
        memory_format=torch.channels_last,
    )
    block_m = 64
    block_n = 16
    block_k = 64
    grid = (
        triton.cdiv(batch_size * 20 * 20, block_m),
        triton.cdiv(output_channels, block_n),
    )
    torch.library.wrap_triton(_frozen_nature_conv1_kernel)[grid](
        observations,
        weight,
        bias,
        output,
        *observations.stride(),
        *weight.stride(),
        *output.stride(),
        batch_size,
        output_channels,
        block_m,
        block_n,
        block_k,
        num_warps=2,
    )
    return output


@frozen_nature_conv1.register_fake
def _frozen_nature_conv1_fake(
    observations: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del bias
    return torch.empty(
        (observations.shape[0], weight.shape[0], 20, 20),
        dtype=weight.dtype,
        device=observations.device,
        memory_format=torch.channels_last,
    )


@triton.jit
def _portal_intersections_kernel(
    player_x,
    player_y,
    player_angle,
    ray_offsets,
    walls,
    wall_blocks_sight,
    active,
    nearest_distances,
    nearest_walls,
    nearest_along,
    blocking_distances,
    wall_count: tl.constexpr,
    ray_count: tl.constexpr,
    layer_count: tl.constexpr,
    block_walls: tl.constexpr,
    use_active_mask: tl.constexpr,
):
    ray_index = tl.program_id(0)
    env_index = ray_index // ray_count
    column = ray_index - env_index * ray_count
    if use_active_mask and not tl.load(active + env_index):
        return
    ray_offset = tl.load(ray_offsets + column)
    ray_angle = tl.load(player_angle + env_index) + ray_offset
    ray_x = tl.cos(ray_angle)
    ray_y = tl.sin(ray_angle)
    origin_x = tl.load(player_x + env_index)
    origin_y = tl.load(player_y + env_index)
    wall_index = tl.arange(0, block_walls)
    valid_wall = wall_index < wall_count
    wall_base = wall_index * 4
    start_x = tl.load(walls + wall_base, mask=valid_wall, other=0.0)
    start_y = tl.load(walls + wall_base + 1, mask=valid_wall, other=0.0)
    end_x = tl.load(walls + wall_base + 2, mask=valid_wall, other=0.0)
    end_y = tl.load(walls + wall_base + 3, mask=valid_wall, other=0.0)
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    offset_x = start_x - origin_x
    offset_y = start_y - origin_y
    denominator = ray_x * segment_y - ray_y * segment_x
    safe = tl.where(tl.abs(denominator) < 1.0e-6, 1.0, denominator)
    distance = (offset_x * segment_y - offset_y * segment_x) / safe
    along = (offset_x * ray_y - offset_y * ray_x) / safe
    intersects = (
        valid_wall
        & (tl.abs(denominator) >= 1.0e-6)
        & (distance > 0.0)
        & (along >= 0.0)
        & (along <= 1.0)
    )
    corrected = tl.where(intersects, distance * tl.cos(ray_offset), float("inf"))
    blocks_sight = tl.load(
        wall_blocks_sight + wall_index,
        mask=valid_wall,
        other=0,
    ).to(tl.int1)
    blocking = tl.min(tl.where(blocks_sight, corrected, float("inf")), axis=0)
    tl.store(
        blocking_distances + ray_index,
        tl.maximum(1.0, tl.minimum(4096.0, blocking)),
    )
    output_base = ray_index * layer_count
    remaining = corrected
    for layer in tl.static_range(layer_count):
        selected_distance = tl.min(remaining, axis=0)
        selected_wall = tl.argmin(remaining, axis=0, tie_break_left=True)
        selected_along = tl.sum(
            tl.where(wall_index == selected_wall, along, 0.0),
            axis=0,
        )
        tl.store(
            nearest_distances + output_base + layer,
            tl.where(
                selected_distance == float("inf"),
                selected_distance,
                tl.maximum(1.0, tl.minimum(4096.0, selected_distance)),
            ),
        )
        tl.store(nearest_walls + output_base + layer, selected_wall)
        tl.store(
            nearest_along + output_base + layer,
            tl.maximum(0.0, tl.minimum(1.0, selected_along)),
        )
        remaining = tl.where(
            wall_index == selected_wall,
            float("inf"),
            remaining,
        )


@torch.library.custom_op(
    "gradoom::portal_intersections",
    mutates_args=(),
    device_types="cuda",
)
def portal_intersections(
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    ray_offsets: torch.Tensor,
    walls: torch.Tensor,
    wall_blocks_sight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the closest wall crossings and closest blocking wall per ray."""

    env_count = player_x.shape[0]
    ray_count = ray_offsets.shape[0]
    wall_count = walls.shape[0]
    layer_count = min(_FAST_NATIVE_PORTAL_LAYERS, wall_count)
    distances = torch.empty(
        (env_count, ray_count, layer_count),
        device=player_x.device,
        dtype=torch.float32,
    )
    wall_indices = torch.empty(
        (env_count, ray_count, layer_count),
        device=player_x.device,
        dtype=torch.int64,
    )
    along = torch.empty_like(distances)
    blocking_distances = torch.empty(
        (env_count, ray_count),
        device=player_x.device,
        dtype=torch.float32,
    )
    grid = (env_count * ray_count,)
    torch.library.wrap_triton(_portal_intersections_kernel)[grid](
        player_x,
        player_y,
        player_angle,
        ray_offsets,
        walls,
        wall_blocks_sight,
        player_x,
        distances,
        wall_indices,
        along,
        blocking_distances,
        wall_count,
        ray_count,
        layer_count,
        triton.next_power_of_2(wall_count),
        False,
        num_warps=1,
    )
    return distances, wall_indices, along, blocking_distances


@torch.library.custom_op(
    "gradoom::masked_portal_intersections",
    mutates_args=(),
    device_types="cuda",
)
def masked_portal_intersections(
    active: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    ray_offsets: torch.Tensor,
    walls: torch.Tensor,
    wall_blocks_sight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Portal intersections that skip inactive environment lanes."""

    env_count = player_x.shape[0]
    ray_count = ray_offsets.shape[0]
    wall_count = walls.shape[0]
    layer_count = min(_FAST_NATIVE_PORTAL_LAYERS, wall_count)
    shape = (env_count, ray_count, layer_count)
    distances = torch.empty(shape, device=player_x.device, dtype=torch.float32)
    wall_indices = torch.empty(shape, device=player_x.device, dtype=torch.int64)
    along = torch.empty_like(distances)
    blocking_distances = torch.empty(
        (env_count, ray_count),
        device=player_x.device,
        dtype=torch.float32,
    )
    torch.library.wrap_triton(_portal_intersections_kernel)[(env_count * ray_count,)](
        player_x,
        player_y,
        player_angle,
        ray_offsets,
        walls,
        wall_blocks_sight,
        active,
        distances,
        wall_indices,
        along,
        blocking_distances,
        wall_count,
        ray_count,
        layer_count,
        triton.next_power_of_2(wall_count),
        True,
        num_warps=1,
    )
    return distances, wall_indices, along, blocking_distances


@masked_portal_intersections.register_fake
def _masked_portal_intersections_fake(
    active: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    ray_offsets: torch.Tensor,
    walls: torch.Tensor,
    wall_blocks_sight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del active, player_y, player_angle, wall_blocks_sight
    shape = (
        player_x.shape[0],
        ray_offsets.shape[0],
        min(_FAST_NATIVE_PORTAL_LAYERS, walls.shape[0]),
    )
    distances = player_x.new_empty(shape)
    wall_indices = torch.empty(shape, device=player_x.device, dtype=torch.int64)
    along = player_x.new_empty(shape)
    blocking = player_x.new_empty((player_x.shape[0], ray_offsets.shape[0]))
    return distances, wall_indices, along, blocking


@portal_intersections.register_fake
def _portal_intersections_fake(
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    ray_offsets: torch.Tensor,
    walls: torch.Tensor,
    wall_blocks_sight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del wall_blocks_sight
    shape = (
        player_x.shape[0],
        ray_offsets.shape[0],
        min(_FAST_NATIVE_PORTAL_LAYERS, walls.shape[0]),
    )
    distances = player_x.new_empty(shape)
    wall_indices = torch.empty(shape, device=player_x.device, dtype=torch.int64)
    along = player_x.new_empty(shape)
    blocking = player_x.new_empty((player_x.shape[0], ray_offsets.shape[0]))
    del player_y, player_angle
    return distances, wall_indices, along, blocking


@triton.jit
def _sector_from_crossings(
    ray_crossing,
    sector_edge_mask,
    wall_index,
    valid_wall,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
):
    """Return the first sector with odd ray-crossing parity.

    The certified deathmatch map has fewer than 31 sectors, so encode every
    wall's sector memberships in one int32 and reduce all parities together.
    Keep the scalar fallback so other user-supplied maps remain supported.
    """

    if sector_count < 31:
        wall_sector_bits = wall_index * 0
        for sector in tl.static_range(sector_count):
            sector_edge = tl.load(
                sector_edge_mask + sector * portal_wall_count + wall_index,
                mask=valid_wall,
                other=0,
            ).to(tl.int1)
            wall_sector_bits = wall_sector_bits | tl.where(
                sector_edge,
                1 << sector,
                0,
            )
        crossing_sector_bits = tl.xor_sum(
            tl.where(ray_crossing, wall_sector_bits, 0),
            axis=0,
        )
        result = 0
        found = False
        for sector in tl.static_range(sector_count):
            inside = (crossing_sector_bits & (1 << sector)) != 0
            select = inside & ~found
            result = tl.where(select, sector, result)
            found = found | inside
        return result

    result = 0
    found = False
    for sector in tl.static_range(sector_count):
        sector_edge = tl.load(
            sector_edge_mask + sector * portal_wall_count + wall_index,
            mask=valid_wall,
            other=0,
        ).to(tl.int1)
        parity = (tl.sum((ray_crossing & sector_edge).to(tl.int32), axis=0) & 1) != 0
        select = parity & ~found
        result = tl.where(select, sector, result)
        found = found | parity
    return result


@triton.jit
def _drop_opening_at(
    point_x,
    point_y,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    wall_index = tl.arange(0, block_portal_walls)
    valid_wall = wall_index < portal_wall_count
    wall_base = wall_index * 4
    x1 = tl.load(portal_walls + wall_base, mask=valid_wall, other=0.0)
    y1 = tl.load(portal_walls + wall_base + 1, mask=valid_wall, other=0.0)
    x2 = tl.load(portal_walls + wall_base + 2, mask=valid_wall, other=0.0)
    y2 = tl.load(portal_walls + wall_base + 3, mask=valid_wall, other=0.0)
    safe_dy = tl.where(tl.abs(y2 - y1) < 1.0e-6, 1.0, y2 - y1)
    crossing_x = x1 + (point_y - y1) * (x2 - x1) / safe_dy
    ray_crossing = valid_wall & ((y1 > point_y) != (y2 > point_y)) & (point_x < crossing_x)
    center_sector = _sector_from_crossings(
        ray_crossing,
        sector_edge_mask,
        wall_index,
        valid_wall,
        portal_wall_count,
        sector_count,
    )

    floor = tl.load(sector_heights + center_sector * 2)
    ceiling = tl.load(sector_heights + center_sector * 2 + 1)
    left = point_x - 20.0
    right = point_x + 20.0
    bottom = point_y - 20.0
    top = point_y + 20.0
    bounds_overlap = (
        valid_wall
        & (right > tl.minimum(x1, x2))
        & (left < tl.maximum(x1, x2))
        & (top > tl.minimum(y1, y2))
        & (bottom < tl.maximum(y1, y2))
    )
    dx = x2 - x1
    dy = y2 - y1
    side_bottom_left = dx * (bottom - y1) - dy * (left - x1)
    side_bottom_right = dx * (bottom - y1) - dy * (right - x1)
    side_top_left = dx * (top - y1) - dy * (left - x1)
    side_top_right = dx * (top - y1) - dy * (right - x1)
    minimum_side = tl.minimum(
        tl.minimum(side_bottom_left, side_bottom_right),
        tl.minimum(side_top_left, side_top_right),
    )
    maximum_side = tl.maximum(
        tl.maximum(side_bottom_left, side_bottom_right),
        tl.maximum(side_top_left, side_top_right),
    )
    touches_line = bounds_overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)
    front_sector = tl.load(
        portal_wall_sectors + wall_index * 2,
        mask=valid_wall,
        other=-1,
    )
    back_sector = tl.load(
        portal_wall_sectors + wall_index * 2 + 1,
        mask=valid_wall,
        other=-1,
    )
    valid_front = front_sector >= 0
    valid_back = back_sector >= 0
    safe_front = tl.maximum(front_sector, 0)
    safe_back = tl.maximum(back_sector, 0)
    front_floor = tl.load(sector_heights + safe_front * 2)
    front_ceiling = tl.load(sector_heights + safe_front * 2 + 1)
    back_floor = tl.load(sector_heights + safe_back * 2)
    back_ceiling = tl.load(sector_heights + safe_back * 2 + 1)
    touched_front = touches_line & valid_front
    touched_back = touches_line & valid_back
    floor = tl.maximum(
        floor,
        tl.maximum(
            tl.max(tl.where(touched_front, front_floor, -float("inf")), axis=0),
            tl.max(tl.where(touched_back, back_floor, -float("inf")), axis=0),
        ),
    )
    ceiling = tl.minimum(
        ceiling,
        tl.minimum(
            tl.min(tl.where(touched_front, front_ceiling, float("inf")), axis=0),
            tl.min(tl.where(touched_back, back_ceiling, float("inf")), axis=0),
        ),
    )
    return floor, ceiling


@triton.jit
def _move_drops_kernel(
    active_drop,
    drop_x_fixed,
    drop_y_fixed,
    drop_z_fixed,
    drop_velocity_x_fixed,
    drop_velocity_y_fixed,
    drop_velocity_z_fixed,
    drop_x,
    drop_y,
    drop_z,
    blocking_walls,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    actor_index = tl.program_id(0)
    current_x_fixed = tl.load(drop_x_fixed + actor_index)
    current_y_fixed = tl.load(drop_y_fixed + actor_index)
    current_z_fixed = tl.load(drop_z_fixed + actor_index)
    if not tl.load(active_drop + actor_index):
        tl.store(drop_velocity_x_fixed + actor_index, 0)
        tl.store(drop_velocity_y_fixed + actor_index, 0)
        tl.store(drop_velocity_z_fixed + actor_index, 0)
        tl.store(drop_x + actor_index, current_x_fixed.to(tl.float32) / 65536.0)
        tl.store(drop_y + actor_index, current_y_fixed.to(tl.float32) / 65536.0)
        tl.store(drop_z + actor_index, current_z_fixed.to(tl.float32) / 65536.0)
        return

    velocity_x_fixed = tl.load(drop_velocity_x_fixed + actor_index)
    velocity_y_fixed = tl.load(drop_velocity_y_fixed + actor_index)
    velocity_z_fixed = tl.load(drop_velocity_z_fixed + actor_index)
    current_x = current_x_fixed.to(tl.float32) / 65536.0
    current_y = current_y_fixed.to(tl.float32) / 65536.0
    current_z = current_z_fixed.to(tl.float32) / 65536.0
    old_floor, old_ceiling = _drop_opening_at(
        current_x,
        current_y,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )
    old_floor_fixed = (old_floor * 65536.0).to(tl.int64)
    grounded_before_move = current_z_fixed <= old_floor_fixed
    proposed_x_fixed = current_x_fixed + velocity_x_fixed
    proposed_y_fixed = current_y_fixed + velocity_y_fixed
    proposed_x = proposed_x_fixed.to(tl.float32) / 65536.0
    proposed_y = proposed_y_fixed.to(tl.float32) / 65536.0
    proposed_floor, proposed_ceiling = _drop_opening_at(
        proposed_x,
        proposed_y,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )

    wall_index = tl.arange(0, block_blocking_walls)
    valid_wall = wall_index < blocking_wall_count
    wall_base = wall_index * 4
    x1 = tl.load(blocking_walls + wall_base, mask=valid_wall, other=0.0)
    y1 = tl.load(blocking_walls + wall_base + 1, mask=valid_wall, other=0.0)
    x2 = tl.load(blocking_walls + wall_base + 2, mask=valid_wall, other=0.0)
    y2 = tl.load(blocking_walls + wall_base + 3, mask=valid_wall, other=0.0)
    left = proposed_x - 20.0
    right = proposed_x + 20.0
    bottom = proposed_y - 20.0
    top = proposed_y + 20.0
    overlap = (
        valid_wall
        & (right > tl.minimum(x1, x2))
        & (left < tl.maximum(x1, x2))
        & (top > tl.minimum(y1, y2))
        & (bottom < tl.maximum(y1, y2))
    )
    dx = x2 - x1
    dy = y2 - y1
    side_bottom_left = dx * (bottom - y1) - dy * (left - x1)
    side_bottom_right = dx * (bottom - y1) - dy * (right - x1)
    side_top_left = dx * (top - y1) - dy * (left - x1)
    side_top_right = dx * (top - y1) - dy * (right - x1)
    minimum_side = tl.minimum(
        tl.minimum(side_bottom_left, side_bottom_right),
        tl.minimum(side_top_left, side_top_right),
    )
    maximum_side = tl.maximum(
        tl.maximum(side_bottom_left, side_bottom_right),
        tl.maximum(side_top_left, side_top_right),
    )
    wall_collision = (
        tl.max((overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)).to(tl.int32), axis=0)
        != 0
    )
    horizontal_collision = (
        wall_collision
        | (proposed_floor > current_z + 24.0)
        | (proposed_ceiling - tl.maximum(current_z, proposed_floor) < 16.0)
    )
    moved = ~horizontal_collision
    moved_x_fixed = tl.where(moved, proposed_x_fixed, current_x_fixed)
    moved_y_fixed = tl.where(moved, proposed_y_fixed, current_y_fixed)
    moved_floor = tl.where(moved, proposed_floor, old_floor)
    moved_ceiling = tl.where(moved, proposed_ceiling, old_ceiling)
    retained_x = tl.where(moved, velocity_x_fixed, 0)
    retained_y = tl.where(moved, velocity_y_fixed, 0)
    stopped = (
        (retained_x > -4096)
        & (retained_x < 4096)
        & (retained_y > -4096)
        & (retained_y < 4096)
    )
    friction_x = tl.where(stopped, 0, (retained_x * 0xE800) >> 16)
    friction_y = tl.where(stopped, 0, (retained_y * 0xE800) >> 16)
    next_velocity_x = tl.where(grounded_before_move, friction_x, retained_x)
    next_velocity_y = tl.where(grounded_before_move, friction_y, retained_y)

    floor_fixed = (moved_floor * 65536.0).to(tl.int64)
    ceiling_fixed = (moved_ceiling * 65536.0).to(tl.int64)
    proposed_z_fixed = current_z_fixed + velocity_z_fixed
    above_floor = proposed_z_fixed > floor_fixed
    next_velocity_z = tl.where(above_floor, velocity_z_fixed - 65536, velocity_z_fixed)
    hit_floor = proposed_z_fixed <= floor_fixed
    ceiling_limit_fixed = ceiling_fixed - 16 * 65536
    hit_ceiling = proposed_z_fixed > ceiling_limit_fixed
    clipped_z_fixed = tl.minimum(tl.maximum(proposed_z_fixed, floor_fixed), ceiling_limit_fixed)
    next_velocity_z = tl.where(hit_floor & (next_velocity_z < 0), 0, next_velocity_z)
    next_velocity_z = tl.where(hit_ceiling & (next_velocity_z > 0), 0, next_velocity_z)
    tl.store(drop_x_fixed + actor_index, moved_x_fixed)
    tl.store(drop_y_fixed + actor_index, moved_y_fixed)
    tl.store(drop_z_fixed + actor_index, clipped_z_fixed)
    tl.store(drop_velocity_x_fixed + actor_index, next_velocity_x)
    tl.store(drop_velocity_y_fixed + actor_index, next_velocity_y)
    tl.store(drop_velocity_z_fixed + actor_index, next_velocity_z)
    tl.store(drop_x + actor_index, moved_x_fixed.to(tl.float32) / 65536.0)
    tl.store(drop_y + actor_index, moved_y_fixed.to(tl.float32) / 65536.0)
    tl.store(drop_z + actor_index, clipped_z_fixed.to(tl.float32) / 65536.0)


@torch.library.custom_op(
    "gradoom::move_drops_",
    mutates_args=(
        "drop_x_fixed",
        "drop_y_fixed",
        "drop_z_fixed",
        "drop_velocity_x_fixed",
        "drop_velocity_y_fixed",
        "drop_velocity_z_fixed",
        "drop_x",
        "drop_y",
        "drop_z",
    ),
    device_types="cuda",
)
def move_drops_(
    active_drop: torch.Tensor,
    drop_x_fixed: torch.Tensor,
    drop_y_fixed: torch.Tensor,
    drop_z_fixed: torch.Tensor,
    drop_velocity_x_fixed: torch.Tensor,
    drop_velocity_y_fixed: torch.Tensor,
    drop_velocity_z_fixed: torch.Tensor,
    drop_x: torch.Tensor,
    drop_y: torch.Tensor,
    drop_z: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
) -> None:
    """Advance sparse dropped pickups without dense inactive-slot geometry."""

    blocking_wall_count = blocking_walls.shape[0]
    portal_wall_count = portal_walls.shape[0]
    torch.library.wrap_triton(_move_drops_kernel)[(active_drop.numel(),)](
        active_drop,
        drop_x_fixed,
        drop_y_fixed,
        drop_z_fixed,
        drop_velocity_x_fixed,
        drop_velocity_y_fixed,
        drop_velocity_z_fixed,
        drop_x,
        drop_y,
        drop_z,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        blocking_wall_count,
        portal_wall_count,
        sector_edge_mask.shape[0],
        triton.next_power_of_2(blocking_wall_count),
        triton.next_power_of_2(portal_wall_count),
        num_warps=1,
    )


@move_drops_.register_fake
def _move_drops_fake(
    active_drop: torch.Tensor,
    drop_x_fixed: torch.Tensor,
    drop_y_fixed: torch.Tensor,
    drop_z_fixed: torch.Tensor,
    drop_velocity_x_fixed: torch.Tensor,
    drop_velocity_y_fixed: torch.Tensor,
    drop_velocity_z_fixed: torch.Tensor,
    drop_x: torch.Tensor,
    drop_y: torch.Tensor,
    drop_z: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
) -> None:
    del (
        active_drop,
        drop_x_fixed,
        drop_y_fixed,
        drop_z_fixed,
        drop_velocity_x_fixed,
        drop_velocity_y_fixed,
        drop_velocity_z_fixed,
        drop_x,
        drop_y,
        drop_z,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
    )


@triton.jit
def _render_portal_walls_kernel(
    frame,
    surface_depth,
    active,
    view_z,
    center,
    distances,
    wall_indices,
    wall_along,
    player_x,
    player_y,
    portal_walls,
    portal_wall_sectors,
    sector_heights,
    portal_side_texture_ids,
    portal_side_texture_offsets,
    portal_wall_lengths,
    texture_widths,
    texture_heights,
    texture_index_atlas,
    colormap,
    policy_grayscale_palette,
    sector_lights,
    flash_light,
    projection_focal_y,
    observation_height: tl.constexpr,
    observation_width: tl.constexpr,
    layer_count: tl.constexpr,
    atlas_stride_texture: tl.constexpr,
    atlas_stride_y: tl.constexpr,
    atlas_stride_x: tl.constexpr,
    block_height: tl.constexpr,
    use_active_mask: tl.constexpr,
    output_indexed: tl.constexpr,
    use_flash_light: tl.constexpr,
    write_surface_depth: tl.constexpr,
):
    ray_index = tl.program_id(0)
    y_block = tl.program_id(1)
    env_index = ray_index // observation_width
    column = ray_index - env_index * observation_width
    if use_active_mask and not tl.load(active + env_index):
        return
    pixel_y = y_block * block_height + tl.arange(0, block_height)
    valid_pixel = pixel_y < observation_height
    frame_index = (
        env_index * observation_height * observation_width + pixel_y * observation_width + column
    )
    value = tl.load(frame + frame_index, mask=valid_pixel, other=0.0).to(tl.float32)
    if write_surface_depth:
        depth_value = tl.load(
            surface_depth + frame_index,
            mask=valid_pixel,
            other=float("inf"),
        )
    filled = tl.zeros((block_height,), dtype=tl.int1)
    current_view_z = tl.load(view_z + env_index)
    current_center = tl.load(center + env_index)
    current_x = tl.load(player_x + env_index)
    current_y = tl.load(player_y + env_index)
    intersection_base = ray_index * layer_count

    for layer in tl.static_range(layer_count):
        distance = tl.load(distances + intersection_base + layer)
        if distance != float("inf"):
            wall_index = tl.load(wall_indices + intersection_base + layer)
            along = tl.load(wall_along + intersection_base + layer)
            front = tl.maximum(
                tl.load(portal_wall_sectors + wall_index * 2),
                0,
            )
            back_raw = tl.load(portal_wall_sectors + wall_index * 2 + 1)
            back = tl.maximum(back_raw, 0)
            front_floor = tl.load(sector_heights + front * 2)
            front_ceiling = tl.load(sector_heights + front * 2 + 1)
            back_floor = tl.load(sector_heights + back * 2)
            back_ceiling = tl.load(sector_heights + back * 2 + 1)
            one_sided = back_raw < 0
            lower_low = tl.minimum(front_floor, back_floor)
            lower_high = tl.maximum(front_floor, back_floor)
            upper_low = tl.minimum(front_ceiling, back_ceiling)
            upper_high = tl.maximum(front_ceiling, back_ceiling)

            one_top = current_center - (
                (front_ceiling - current_view_z) * projection_focal_y / distance
            )
            one_bottom = current_center - (
                (front_floor - current_view_z) * projection_focal_y / distance
            )
            lower_top = current_center - (
                (lower_high - current_view_z) * projection_focal_y / distance
            )
            lower_bottom = current_center - (
                (lower_low - current_view_z) * projection_focal_y / distance
            )
            upper_top = current_center - (
                (upper_high - current_view_z) * projection_focal_y / distance
            )
            upper_bottom = current_center - (
                (upper_low - current_view_z) * projection_focal_y / distance
            )

            wall_base = wall_index * 4
            wall_x1 = tl.load(portal_walls + wall_base)
            wall_y1 = tl.load(portal_walls + wall_base + 1)
            segment_x = tl.load(portal_walls + wall_base + 2) - wall_x1
            segment_y = tl.load(portal_walls + wall_base + 3) - wall_y1
            camera_cross = segment_x * (current_y - wall_y1) - segment_y * (current_x - wall_x1)
            side_index = (camera_cross > 0.0).to(tl.int64)
            from_front = side_index == 0
            view_floor = tl.where(from_front, front_floor, back_floor)
            other_floor = tl.where(from_front, back_floor, front_floor)
            view_ceiling = tl.where(from_front, front_ceiling, back_ceiling)
            other_ceiling = tl.where(from_front, back_ceiling, front_ceiling)
            one_span = (
                one_sided
                & from_front
                & (pixel_y.to(tl.float32) >= one_top)
                & (pixel_y.to(tl.float32) < one_bottom)
            )
            lower_span = (
                ~one_sided
                & (view_floor < other_floor)
                & (pixel_y.to(tl.float32) >= lower_top)
                & (pixel_y.to(tl.float32) < lower_bottom)
            )
            upper_span = (
                ~one_sided
                & (view_ceiling > other_ceiling)
                & (pixel_y.to(tl.float32) >= upper_top)
                & (pixel_y.to(tl.float32) < upper_bottom)
            )
            texture_base = wall_index * 6 + side_index * 3
            texture_id = tl.where(
                one_span,
                tl.load(portal_side_texture_ids + texture_base),
                tl.where(
                    lower_span,
                    tl.load(portal_side_texture_ids + texture_base + 1),
                    tl.load(portal_side_texture_ids + texture_base + 2),
                ),
            )
            span = valid_pixel & (one_span | lower_span | upper_span) & (texture_id >= 0) & ~filled
            safe_texture_id = tl.maximum(texture_id, 0)
            texture_width = tl.load(texture_widths + safe_texture_id)
            texture_height = tl.load(texture_heights + safe_texture_id)
            offset_base = wall_index * 4 + side_index * 2
            texture_offset_u = tl.load(portal_side_texture_offsets + offset_base)
            texture_offset_v = tl.load(portal_side_texture_offsets + offset_base + 1)
            wall_length = tl.load(portal_wall_lengths + wall_index)
            horizontal_offset = tl.where(
                from_front,
                along * wall_length,
                wall_length - along * wall_length,
            )
            # PrepWallRoundFix retains the last texel at the reversed endpoint
            # rather than wrapping the exact wall length back to column zero.
            horizontal_offset = tl.minimum(horizontal_offset, wall_length - 1.0e-4)
            texture_u = tl.floor(horizontal_offset + texture_offset_u).to(tl.int64)
            texture_u = texture_u % texture_width
            texture_u = tl.where(texture_u < 0, texture_u + texture_width, texture_u)
            world_z = current_view_z + (
                (current_center - pixel_y.to(tl.float32)) * distance / projection_focal_y
            )
            texture_origin_z = tl.where(
                one_span,
                view_ceiling,
                tl.where(lower_span, other_floor, other_ceiling),
            )
            texture_v = tl.floor(texture_origin_z - world_z + texture_offset_v).to(tl.int64)
            texture_v = texture_v % texture_height
            texture_v = tl.where(texture_v < 0, texture_v + texture_height, texture_v)
            palette_index = tl.load(
                texture_index_atlas
                + safe_texture_id * atlas_stride_texture
                + texture_v * atlas_stride_y
                + texture_u * atlas_stride_x,
                mask=span,
                other=0,
            ).to(tl.int64)
            view_sector = tl.where(from_front, front, back)
            light = tl.load(sector_lights + view_sector).to(tl.float32)
            if use_flash_light:
                light += tl.load(flash_light + env_index).to(tl.float32) * 16.0
            horizontal_wall = tl.abs(segment_y) < 1.0e-6
            vertical_wall = tl.abs(segment_x) < 1.0e-6
            light += tl.where(vertical_wall, 16.0, tl.where(horizontal_wall, -16.0, 0.0))
            base_shade = 61.0 - light / 4.0
            visibility = tl.minimum(24.0, 1280.0 / tl.maximum(distance, 1.0))
            shade = tl.maximum(0, tl.minimum(31, tl.floor(base_shade - visibility))).to(
                tl.int64
            )
            lit_index = tl.load(
                colormap + shade * 256 + palette_index,
                mask=span,
                other=0,
            ).to(tl.int64)
            wall_value = tl.where(
                output_indexed,
                lit_index.to(tl.float32),
                tl.load(
                    policy_grayscale_palette + lit_index,
                    mask=span,
                    other=0.0,
                ).to(tl.float32),
            )
            value = tl.where(span, wall_value, value)
            if write_surface_depth:
                depth_value = tl.where(span, distance, depth_value)
            filled = filled | span

    tl.store(frame + frame_index, value, mask=valid_pixel)
    if write_surface_depth:
        tl.store(surface_depth + frame_index, depth_value, mask=valid_pixel)


@torch.library.custom_op(
    "gradoom::render_portal_walls_",
    mutates_args=("frame",),
    device_types="cuda",
)
def render_portal_walls_(
    frame: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    distances: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_along: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_heights: torch.Tensor,
    portal_side_texture_ids: torch.Tensor,
    portal_side_texture_offsets: torch.Tensor,
    portal_wall_lengths: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    colormap: torch.Tensor,
    policy_grayscale_palette: torch.Tensor,
    sector_lights: torch.Tensor,
) -> None:
    """Composite all sorted portal-wall layers in one column/pixel launch."""

    observation_height = frame.shape[1]
    observation_width = frame.shape[2]
    layer_count = distances.shape[2]
    block_height = 128
    grid = (
        frame.shape[0] * observation_width,
        triton.cdiv(observation_height, block_height),
    )
    torch.library.wrap_triton(_render_portal_walls_kernel)[grid](
        frame,
        frame,
        view_z,
        view_z,
        center,
        distances,
        wall_indices,
        wall_along,
        player_x,
        player_y,
        portal_walls,
        portal_wall_sectors,
        sector_heights,
        portal_side_texture_ids,
        portal_side_texture_offsets,
        portal_wall_lengths,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        colormap,
        policy_grayscale_palette,
        sector_lights,
        view_z,
        67.2,
        observation_height,
        observation_width,
        layer_count,
        texture_index_atlas.stride(0),
        texture_index_atlas.stride(1),
        texture_index_atlas.stride(2),
        block_height,
        False,
        False,
        False,
        False,
        num_warps=2,
    )


@torch.library.custom_op(
    "gradoom::masked_render_portal_walls_",
    mutates_args=("frame",),
    device_types="cuda",
)
def masked_render_portal_walls_(
    frame: torch.Tensor,
    active: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    distances: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_along: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_heights: torch.Tensor,
    portal_side_texture_ids: torch.Tensor,
    portal_side_texture_offsets: torch.Tensor,
    portal_wall_lengths: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    colormap: torch.Tensor,
    policy_grayscale_palette: torch.Tensor,
    sector_lights: torch.Tensor,
) -> None:
    """Composite portal walls only for active environment lanes."""

    observation_height = frame.shape[1]
    observation_width = frame.shape[2]
    layer_count = distances.shape[2]
    block_height = 128
    grid = (
        frame.shape[0] * observation_width,
        triton.cdiv(observation_height, block_height),
    )
    torch.library.wrap_triton(_render_portal_walls_kernel)[grid](
        frame,
        frame,
        active,
        view_z,
        center,
        distances,
        wall_indices,
        wall_along,
        player_x,
        player_y,
        portal_walls,
        portal_wall_sectors,
        sector_heights,
        portal_side_texture_ids,
        portal_side_texture_offsets,
        portal_wall_lengths,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        colormap,
        policy_grayscale_palette,
        sector_lights,
        view_z,
        67.2,
        observation_height,
        observation_width,
        layer_count,
        texture_index_atlas.stride(0),
        texture_index_atlas.stride(1),
        texture_index_atlas.stride(2),
        block_height,
        True,
        False,
        False,
        False,
        num_warps=2,
    )


@torch.library.custom_op(
    "gradoom::render_fast_native_portal_walls_",
    mutates_args=("frame", "surface_depth"),
    device_types="cuda",
)
def render_fast_native_portal_walls_(
    frame: torch.Tensor,
    surface_depth: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    distances: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_along: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_heights: torch.Tensor,
    portal_side_texture_ids: torch.Tensor,
    portal_side_texture_offsets: torch.Tensor,
    portal_wall_lengths: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    colormap: torch.Tensor,
    sector_lights: torch.Tensor,
    flash_light: torch.Tensor,
) -> None:
    """Composite native-resolution indexed walls for the fused policy path."""

    observation_height = frame.shape[1]
    observation_width = frame.shape[2]
    layer_count = distances.shape[2]
    block_height = 256
    grid = (
        frame.shape[0] * observation_width,
        triton.cdiv(observation_height, block_height),
    )
    torch.library.wrap_triton(_render_portal_walls_kernel)[grid](
        frame,
        surface_depth,
        view_z,
        view_z,
        center,
        distances,
        wall_indices,
        wall_along,
        player_x,
        player_y,
        portal_walls,
        portal_wall_sectors,
        sector_heights,
        portal_side_texture_ids,
        portal_side_texture_offsets,
        portal_wall_lengths,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        colormap,
        colormap,
        sector_lights,
        flash_light,
        192.0,
        observation_height,
        observation_width,
        layer_count,
        texture_index_atlas.stride(0),
        texture_index_atlas.stride(1),
        texture_index_atlas.stride(2),
        block_height,
        False,
        True,
        True,
        True,
        num_warps=4,
    )


@render_fast_native_portal_walls_.register_fake
def _render_fast_native_portal_walls_fake(
    frame: torch.Tensor,
    surface_depth: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    distances: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_along: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_heights: torch.Tensor,
    portal_side_texture_ids: torch.Tensor,
    portal_side_texture_offsets: torch.Tensor,
    portal_wall_lengths: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    colormap: torch.Tensor,
    sector_lights: torch.Tensor,
    flash_light: torch.Tensor,
) -> None:
    del (
        frame,
        surface_depth,
        view_z,
        center,
        distances,
        wall_indices,
        wall_along,
        player_x,
        player_y,
        portal_walls,
        portal_wall_sectors,
        sector_heights,
        portal_side_texture_ids,
        portal_side_texture_offsets,
        portal_wall_lengths,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        colormap,
        sector_lights,
        flash_light,
    )


@masked_render_portal_walls_.register_fake
def _masked_render_portal_walls_fake(
    frame: torch.Tensor,
    active: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    distances: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_along: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_heights: torch.Tensor,
    portal_side_texture_ids: torch.Tensor,
    portal_side_texture_offsets: torch.Tensor,
    portal_wall_lengths: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    colormap: torch.Tensor,
    policy_grayscale_palette: torch.Tensor,
    sector_lights: torch.Tensor,
) -> None:
    del (
        frame,
        active,
        view_z,
        center,
        distances,
        wall_indices,
        wall_along,
        player_x,
        player_y,
        portal_walls,
        portal_wall_sectors,
        sector_heights,
        portal_side_texture_ids,
        portal_side_texture_offsets,
        portal_wall_lengths,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        colormap,
        policy_grayscale_palette,
        sector_lights,
    )


@render_portal_walls_.register_fake
def _render_portal_walls_fake(
    frame: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    distances: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_along: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_heights: torch.Tensor,
    portal_side_texture_ids: torch.Tensor,
    portal_side_texture_offsets: torch.Tensor,
    portal_wall_lengths: torch.Tensor,
    texture_widths: torch.Tensor,
    texture_heights: torch.Tensor,
    texture_index_atlas: torch.Tensor,
    colormap: torch.Tensor,
    policy_grayscale_palette: torch.Tensor,
    sector_lights: torch.Tensor,
) -> None:
    del (
        frame,
        view_z,
        center,
        distances,
        wall_indices,
        wall_along,
        player_x,
        player_y,
        portal_walls,
        portal_wall_sectors,
        sector_heights,
        portal_side_texture_ids,
        portal_side_texture_offsets,
        portal_wall_lengths,
        texture_widths,
        texture_heights,
        texture_index_atlas,
        colormap,
        policy_grayscale_palette,
        sector_lights,
    )


@triton.jit
def _render_sprites_kernel(
    frame,
    blocking_distance,
    actor_x,
    actor_y,
    actor_z,
    actor_alive,
    actor_type,
    player_x,
    player_y,
    player_angle,
    view_z,
    center,
    sprite_widths,
    sprite_heights,
    sprite_left_offsets,
    sprite_top_offsets,
    sprite_atlas,
    sprite_opaque,
    observation_height: tl.constexpr,
    observation_width: tl.constexpr,
    actor_count: tl.constexpr,
    atlas_stride_type: tl.constexpr,
    atlas_stride_y: tl.constexpr,
    atlas_stride_x: tl.constexpr,
    block_actors: tl.constexpr,
    block_height: tl.constexpr,
):
    ray_index = tl.program_id(0)
    env_index = ray_index // observation_width
    column = ray_index - env_index * observation_width
    actor_slot = tl.arange(0, block_actors)
    valid_actor = actor_slot < actor_count
    actor_index = env_index * actor_count + actor_slot
    current_x = tl.load(player_x + env_index)
    current_y = tl.load(player_y + env_index)
    current_angle = tl.load(player_angle + env_index)
    dx = tl.load(actor_x + actor_index, mask=valid_actor, other=0.0) - current_x
    dy = tl.load(actor_y + actor_index, mask=valid_actor, other=0.0) - current_y
    distance = tl.maximum(tl.sqrt(dx * dx + dy * dy), 1.0)
    relative = libdevice.atan2(dy, dx) - current_angle
    relative = relative + 3.141592653589793
    relative = (
        relative - tl.floor(relative / 6.283185307179586) * 6.283185307179586 - 3.141592653589793
    )
    safe_type = tl.maximum(
        tl.load(actor_type + actor_index, mask=valid_actor, other=0),
        0,
    )
    scale = 42.0 / distance
    width = tl.load(sprite_widths + safe_type).to(tl.float32)
    # Match the perspective projection used by the Torch path and Doom's
    # view transform.  Linear angle-to-column mapping increasingly displaced
    # combat sprites toward the edges of the 90-degree field of view.
    screen_center = observation_width * 0.5 - tl.sin(relative) / tl.cos(relative) * 42.0
    left = screen_center - tl.load(sprite_left_offsets + safe_type) * scale
    right = left + width * scale
    candidate = (
        valid_actor
        & tl.load(actor_alive + actor_index, mask=valid_actor, other=0).to(tl.int1)
        & (column.to(tl.float32) >= left)
        & (column.to(tl.float32) < right)
        & (tl.abs(relative) < 0.7853981633974483)
        & (distance < tl.load(blocking_distance + ray_index))
    )
    candidate_distance = tl.where(candidate, distance, float("inf"))
    nearest_distance = tl.min(candidate_distance, axis=0)
    nearest_actor = tl.argmin(candidate_distance, axis=0, tie_break_left=True)
    selected_index = env_index * actor_count + nearest_actor
    selected_type = tl.maximum(tl.load(actor_type + selected_index), 0)
    selected_width = tl.load(sprite_widths + selected_type).to(tl.int64)
    selected_height = tl.load(sprite_heights + selected_type).to(tl.int64)
    selected_scale = 42.0 / nearest_distance
    selected_vertical_scale = 67.2 / nearest_distance
    selected_dx = tl.load(actor_x + selected_index) - current_x
    selected_dy = tl.load(actor_y + selected_index) - current_y
    selected_relative = libdevice.atan2(selected_dy, selected_dx) - current_angle
    selected_relative = selected_relative + 3.141592653589793
    selected_relative = (
        selected_relative
        - tl.floor(selected_relative / 6.283185307179586) * 6.283185307179586
        - 3.141592653589793
    )
    selected_screen_center = (
        observation_width * 0.5
        - tl.sin(selected_relative) / tl.cos(selected_relative) * 42.0
    )
    selected_left = selected_screen_center - tl.load(
        sprite_left_offsets + selected_type
    ) * selected_scale
    selected_top = (
        tl.load(center + env_index)
        + (tl.load(view_z + env_index) - tl.load(actor_z + selected_index))
        * selected_vertical_scale
        - tl.load(sprite_top_offsets + selected_type) * selected_vertical_scale
    )
    sprite_u = tl.floor((column.to(tl.float32) - selected_left) / selected_scale).to(tl.int64)
    pixel_y = tl.arange(0, block_height)
    valid_pixel = pixel_y < observation_height
    sprite_v = tl.floor((pixel_y.to(tl.float32) - selected_top) / selected_vertical_scale).to(
        tl.int64
    )
    inside = (
        valid_pixel
        & (nearest_distance != float("inf"))
        & (sprite_u >= 0)
        & (sprite_u < selected_width)
        & (sprite_v >= 0)
        & (sprite_v < selected_height)
    )
    safe_u = tl.maximum(0, tl.minimum(sprite_u, selected_width - 1))
    safe_v = tl.maximum(0, tl.minimum(sprite_v, selected_height - 1))
    atlas_index = (
        selected_type * atlas_stride_type + safe_v * atlas_stride_y + safe_u * atlas_stride_x
    )
    opaque = tl.load(sprite_opaque + atlas_index, mask=valid_pixel, other=0).to(tl.int1)
    sprite_value = tl.load(sprite_atlas + atlas_index, mask=valid_pixel, other=0).to(tl.float32)
    frame_index = (
        env_index * observation_height * observation_width + pixel_y * observation_width + column
    )
    prior = tl.load(frame + frame_index, mask=valid_pixel, other=0.0)
    tl.store(
        frame + frame_index,
        tl.where(inside & opaque, sprite_value, prior),
        mask=valid_pixel,
    )


@torch.library.custom_op(
    "gradoom::render_sprites_",
    mutates_args=("frame",),
    device_types="cuda",
)
def render_sprites_(
    frame: torch.Tensor,
    blocking_distance: torch.Tensor,
    actor_x: torch.Tensor,
    actor_y: torch.Tensor,
    actor_z: torch.Tensor,
    actor_alive: torch.Tensor,
    actor_type: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    sprite_widths: torch.Tensor,
    sprite_heights: torch.Tensor,
    sprite_left_offsets: torch.Tensor,
    sprite_top_offsets: torch.Tensor,
    sprite_atlas: torch.Tensor,
    sprite_opaque: torch.Tensor,
) -> None:
    """Select and composite the nearest visible sprite for every column."""

    observation_height = frame.shape[1]
    observation_width = frame.shape[2]
    actor_count = actor_x.shape[1]
    block_height = triton.next_power_of_2(observation_height)
    torch.library.wrap_triton(_render_sprites_kernel)[(frame.shape[0] * observation_width,)](
        frame,
        blocking_distance,
        actor_x,
        actor_y,
        actor_z,
        actor_alive,
        actor_type,
        player_x,
        player_y,
        player_angle,
        view_z,
        center,
        sprite_widths,
        sprite_heights,
        sprite_left_offsets,
        sprite_top_offsets,
        sprite_atlas,
        sprite_opaque,
        observation_height,
        observation_width,
        actor_count,
        sprite_atlas.stride(0),
        sprite_atlas.stride(1),
        sprite_atlas.stride(2),
        triton.next_power_of_2(actor_count),
        block_height,
        num_warps=1,
    )


@render_sprites_.register_fake
def _render_sprites_fake(
    frame: torch.Tensor,
    blocking_distance: torch.Tensor,
    actor_x: torch.Tensor,
    actor_y: torch.Tensor,
    actor_z: torch.Tensor,
    actor_alive: torch.Tensor,
    actor_type: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    sprite_widths: torch.Tensor,
    sprite_heights: torch.Tensor,
    sprite_left_offsets: torch.Tensor,
    sprite_top_offsets: torch.Tensor,
    sprite_atlas: torch.Tensor,
    sprite_opaque: torch.Tensor,
) -> None:
    del (
        frame,
        blocking_distance,
        actor_x,
        actor_y,
        actor_z,
        actor_alive,
        actor_type,
        player_x,
        player_y,
        player_angle,
        view_z,
        center,
        sprite_widths,
        sprite_heights,
        sprite_left_offsets,
        sprite_top_offsets,
        sprite_atlas,
        sprite_opaque,
    )


@triton.jit
def _project_fast_native_sprites_kernel(
    actor_x,
    actor_y,
    actor_sprite,
    player_x,
    player_y,
    player_angle,
    sprite_widths,
    sprite_left_offsets,
    projected_depth,
    projected_left,
    projected_right,
    total_actors: tl.constexpr,
    actor_count: tl.constexpr,
    block: tl.constexpr,
):
    """Project every actor once instead of repeating trig for all 320 columns."""

    offset = tl.program_id(0) * block + tl.arange(0, block)
    valid = offset < total_actors
    env = offset // actor_count
    candidate_x = tl.load(actor_x + offset, mask=valid, other=0.0)
    candidate_y = tl.load(actor_y + offset, mask=valid, other=0.0)
    dx = candidate_x - tl.load(player_x + env, mask=valid, other=0.0)
    dy = candidate_y - tl.load(player_y + env, mask=valid, other=0.0)
    distance = tl.maximum(tl.sqrt(dx * dx + dy * dy), 1.0)
    relative = libdevice.atan2(dy, dx) - tl.load(
        player_angle + env,
        mask=valid,
        other=0.0,
    )
    relative = relative + 3.141592653589793
    relative = (
        relative
        - tl.floor(relative / 6.283185307179586) * 6.283185307179586
        - 3.141592653589793
    )
    depth = distance * tl.cos(relative)
    safe_depth = tl.maximum(depth, 1.0)
    sprite = tl.maximum(tl.load(actor_sprite + offset, mask=valid, other=0), 0)
    scale_x = 160.0 / safe_depth
    screen_center = 160.0 - tl.sin(relative) / tl.cos(relative) * 160.0
    left = screen_center - tl.load(sprite_left_offsets + sprite) * scale_x
    right = left + tl.load(sprite_widths + sprite).to(tl.float32) * scale_x
    in_view = valid & (depth > 0.125) & (tl.abs(relative) < 0.7853981633974483)
    tl.store(projected_depth + offset, tl.where(in_view, depth, float("inf")), mask=valid)
    tl.store(projected_left + offset, left, mask=valid)
    tl.store(projected_right + offset, right, mask=valid)


@triton.jit
def _render_fast_native_sprites_kernel(
    frame,
    blocking_distance,
    surface_depth,
    actor_x,
    actor_y,
    actor_z,
    actor_alive,
    actor_sprite,
    actor_fullbright,
    player_x,
    player_y,
    player_angle,
    projected_depth,
    projected_left,
    projected_right,
    view_z,
    center,
    sprite_widths,
    sprite_heights,
    sprite_left_offsets,
    sprite_top_offsets,
    sprite_atlas,
    sprite_opaque,
    sector_lookup,
    lookup_metadata,
    sector_lights,
    flash_light,
    colormap,
    observation_height: tl.constexpr,
    observation_width: tl.constexpr,
    actor_count: tl.constexpr,
    atlas_stride_type: tl.constexpr,
    atlas_stride_y: tl.constexpr,
    atlas_stride_x: tl.constexpr,
    lookup_height: tl.constexpr,
    lookup_width: tl.constexpr,
    block_actors: tl.constexpr,
    block_height: tl.constexpr,
    sprite_rank: tl.constexpr,
):
    """Composite one depth-ranked native indexed actor per screen column."""

    ray_index = tl.program_id(0)
    env_index = ray_index // observation_width
    column = ray_index - env_index * observation_width
    candidate_slot = tl.arange(0, block_actors)
    valid_actor = candidate_slot < actor_count
    candidate_actor = candidate_slot
    actor_index = env_index * actor_count + candidate_actor
    depth = tl.load(projected_depth + actor_index, mask=valid_actor, other=float("inf"))
    left = tl.load(projected_left + actor_index, mask=valid_actor, other=0.0)
    right = tl.load(projected_right + actor_index, mask=valid_actor, other=0.0)
    candidate = (
        valid_actor
        & tl.load(actor_alive + actor_index, mask=valid_actor, other=0).to(tl.int1)
        & (column.to(tl.float32) >= left)
        & (column.to(tl.float32) < right)
        & (depth < tl.load(blocking_distance + ray_index))
    )
    candidate_depth = tl.where(candidate, depth, float("inf"))
    # The wrapper invokes ranks far-to-near.  Retaining several horizontally
    # overlapping sprites is necessary because a nearer sprite can be short
    # or transparent at this row; selecting only one actor for the whole
    # column incorrectly erased every actor behind it.
    selected_depth = float("inf")
    selected_actor = 0
    for _rank in range(sprite_rank + 1):
        selected_depth = tl.min(candidate_depth, axis=0)
        selected_actor = tl.argmin(candidate_depth, axis=0, tie_break_left=True)
        candidate_depth = tl.where(
            candidate_slot == selected_actor,
            float("inf"),
            candidate_depth,
        )
    selected_index = env_index * actor_count + selected_actor
    selected_sprite = tl.maximum(tl.load(actor_sprite + selected_index), 0)
    selected_width = tl.load(sprite_widths + selected_sprite).to(tl.int64)
    selected_height = tl.load(sprite_heights + selected_sprite).to(tl.int64)
    selected_scale_x = 160.0 / selected_depth
    selected_scale_y = 192.0 / selected_depth
    selected_x = tl.load(actor_x + selected_index)
    selected_y = tl.load(actor_y + selected_index)
    selected_left = tl.load(projected_left + selected_index)
    selected_top = (
        tl.load(center + env_index)
        + (tl.load(view_z + env_index) - tl.load(actor_z + selected_index))
        * selected_scale_y
        - tl.load(sprite_top_offsets + selected_sprite) * selected_scale_y
    )
    sprite_u = tl.floor(
        (column.to(tl.float32) - selected_left) / selected_scale_x
    ).to(tl.int64)
    pixel_y = tl.arange(0, block_height)
    valid_pixel = pixel_y < observation_height
    sprite_v = tl.floor(
        (pixel_y.to(tl.float32) - selected_top) / selected_scale_y
    ).to(tl.int64)
    inside = (
        valid_pixel
        & (selected_depth != float("inf"))
        & (sprite_u >= 0)
        & (sprite_u < selected_width)
        & (sprite_v >= 0)
        & (sprite_v < selected_height)
    )
    safe_u = tl.maximum(0, tl.minimum(sprite_u, selected_width - 1))
    safe_v = tl.maximum(0, tl.minimum(sprite_v, selected_height - 1))
    atlas_index = (
        selected_sprite * atlas_stride_type
        + safe_v * atlas_stride_y
        + safe_u * atlas_stride_x
    )
    opaque = tl.load(sprite_opaque + atlas_index, mask=inside, other=0).to(tl.int1)
    palette_index = tl.load(sprite_atlas + atlas_index, mask=inside, other=0).to(tl.int64)
    origin_x = tl.load(lookup_metadata)
    origin_y = tl.load(lookup_metadata + 1)
    cell_size = tl.load(lookup_metadata + 2)
    lookup_x = tl.floor((selected_x - origin_x) / cell_size).to(tl.int64)
    lookup_y = tl.floor((selected_y - origin_y) / cell_size).to(tl.int64)
    in_lookup = (
        (lookup_x >= 0)
        & (lookup_x < lookup_width)
        & (lookup_y >= 0)
        & (lookup_y < lookup_height)
    )
    sector = tl.load(
        sector_lookup + lookup_y * lookup_width + lookup_x,
        mask=in_lookup,
        other=0,
    ).to(tl.int64)
    sector = tl.maximum(sector, 0)
    light = tl.load(sector_lights + sector).to(tl.float32)
    light += tl.load(flash_light + env_index).to(tl.float32) * 16.0
    light = tl.where(tl.load(actor_fullbright + selected_index), 255.0, light)
    base_shade = 61.0 - light / 4.0
    visibility = tl.minimum(24.0, 1280.0 / tl.maximum(selected_depth, 1.0))
    shade = tl.maximum(0, tl.minimum(31, tl.floor(base_shade - visibility))).to(tl.int64)
    lit_index = tl.load(
        colormap + shade * 256 + palette_index,
        mask=inside & opaque,
        other=0,
    ).to(tl.uint8)
    frame_index = (
        env_index * observation_height * observation_width
        + pixel_y * observation_width
        + column
    )
    visible_against_scene = selected_depth < tl.load(
        surface_depth + frame_index,
        mask=valid_pixel,
        other=float("inf"),
    )
    prior = tl.load(frame + frame_index, mask=valid_pixel, other=0)
    tl.store(
        frame + frame_index,
        tl.where(inside & opaque & visible_against_scene, lit_index, prior),
        mask=valid_pixel,
    )


@torch.library.custom_op(
    "gradoom::render_fast_native_sprites_",
    mutates_args=("frame",),
    device_types="cuda",
)
def render_fast_native_sprites_(
    frame: torch.Tensor,
    blocking_distance: torch.Tensor,
    surface_depth: torch.Tensor,
    actor_x: torch.Tensor,
    actor_y: torch.Tensor,
    actor_z: torch.Tensor,
    actor_alive: torch.Tensor,
    actor_sprite: torch.Tensor,
    actor_fullbright: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    sprite_widths: torch.Tensor,
    sprite_heights: torch.Tensor,
    sprite_left_offsets: torch.Tensor,
    sprite_top_offsets: torch.Tensor,
    sprite_atlas: torch.Tensor,
    sprite_opaque: torch.Tensor,
    sector_lookup: torch.Tensor,
    lookup_metadata: torch.Tensor,
    sector_lights: torch.Tensor,
    flash_light: torch.Tensor,
    colormap: torch.Tensor,
) -> None:
    """Composite fixed-shape native indexed actors in one kernel launch."""

    observation_height = frame.shape[1]
    observation_width = frame.shape[2]
    actor_count = actor_x.shape[1]
    block_height = triton.next_power_of_2(observation_height)
    projected_depth = torch.empty_like(actor_x)
    projected_left = torch.empty_like(actor_x)
    projected_right = torch.empty_like(actor_x)
    projection_block = 256
    torch.library.wrap_triton(_project_fast_native_sprites_kernel)[
        (triton.cdiv(actor_x.numel(), projection_block),)
    ](
        actor_x,
        actor_y,
        actor_sprite,
        player_x,
        player_y,
        player_angle,
        sprite_widths,
        sprite_left_offsets,
        projected_depth,
        projected_left,
        projected_right,
        actor_x.numel(),
        actor_count,
        projection_block,
        num_warps=4,
    )
    # Paint far-to-near so transparent foreground texels preserve an already
    # rendered farther actor, matching Doom's masked-sprite composition.
    for sprite_rank in reversed(range(_FAST_NATIVE_SPRITE_LAYERS)):
        torch.library.wrap_triton(_render_fast_native_sprites_kernel)[
            (frame.shape[0] * observation_width,)
        ](
            frame,
            blocking_distance,
            surface_depth,
            actor_x,
            actor_y,
            actor_z,
            actor_alive,
            actor_sprite,
            actor_fullbright,
            player_x,
            player_y,
            player_angle,
            projected_depth,
            projected_left,
            projected_right,
            view_z,
            center,
            sprite_widths,
            sprite_heights,
            sprite_left_offsets,
            sprite_top_offsets,
            sprite_atlas,
            sprite_opaque,
            sector_lookup,
            lookup_metadata,
            sector_lights,
            flash_light,
            colormap,
            observation_height,
            observation_width,
            actor_count,
            sprite_atlas.stride(0),
            sprite_atlas.stride(1),
            sprite_atlas.stride(2),
            sector_lookup.shape[0],
            sector_lookup.shape[1],
            triton.next_power_of_2(actor_count),
            block_height,
            sprite_rank,
            num_warps=4,
        )


@render_fast_native_sprites_.register_fake
def _render_fast_native_sprites_fake(
    frame: torch.Tensor,
    blocking_distance: torch.Tensor,
    surface_depth: torch.Tensor,
    actor_x: torch.Tensor,
    actor_y: torch.Tensor,
    actor_z: torch.Tensor,
    actor_alive: torch.Tensor,
    actor_sprite: torch.Tensor,
    actor_fullbright: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_angle: torch.Tensor,
    view_z: torch.Tensor,
    center: torch.Tensor,
    sprite_widths: torch.Tensor,
    sprite_heights: torch.Tensor,
    sprite_left_offsets: torch.Tensor,
    sprite_top_offsets: torch.Tensor,
    sprite_atlas: torch.Tensor,
    sprite_opaque: torch.Tensor,
    sector_lookup: torch.Tensor,
    lookup_metadata: torch.Tensor,
    sector_lights: torch.Tensor,
    flash_light: torch.Tensor,
    colormap: torch.Tensor,
) -> None:
    del (
        frame,
        blocking_distance,
        surface_depth,
        actor_x,
        actor_y,
        actor_z,
        actor_alive,
        actor_sprite,
        actor_fullbright,
        player_x,
        player_y,
        player_angle,
        view_z,
        center,
        sprite_widths,
        sprite_heights,
        sprite_left_offsets,
        sprite_top_offsets,
        sprite_atlas,
        sprite_opaque,
        sector_lookup,
        lookup_metadata,
        sector_lights,
        flash_light,
        colormap,
    )


@triton.jit
def _render_native_weapon_kernel(
    frame,
    frame_ids,
    flash_ids,
    horizontal_offsets_fixed,
    vertical_offsets_fixed,
    visible,
    patch_atlas,
    patch_opaque,
    patch_widths,
    patch_heights,
    patch_left_offsets,
    patch_top_offsets,
    output,
    total_pixels: tl.constexpr,
    view_height: tl.constexpr,
    view_width: tl.constexpr,
    atlas_stride_type: tl.constexpr,
    atlas_stride_y: tl.constexpr,
    atlas_stride_x: tl.constexpr,
    block: tl.constexpr,
):
    """Composite the weapon and muzzle-flash patches in one native-pixel pass."""

    offset = tl.program_id(0) * block + tl.arange(0, block)
    valid = offset < total_pixels
    pixels_per_env = view_height * view_width
    env = offset // pixels_per_env
    pixel = offset - env * pixels_per_env
    pixel_y = pixel // view_width
    pixel_x = pixel - pixel_y * view_width
    prior = tl.load(frame + offset, mask=valid, other=0)
    lane_visible = tl.load(visible + env, mask=valid, other=0).to(tl.int1)
    horizontal_offset = tl.load(horizontal_offsets_fixed + env, mask=valid, other=0)
    vertical_offset = tl.load(vertical_offsets_fixed + env, mask=valid, other=0)

    frame_id = tl.load(frame_ids + env, mask=valid, other=0).to(tl.int64)
    frame_width = tl.load(patch_widths + frame_id).to(tl.int64)
    frame_height = tl.load(patch_heights + frame_id).to(tl.int64)
    frame_left_offset = tl.load(patch_left_offsets + frame_id).to(tl.int64)
    frame_top_offset = tl.load(patch_top_offsets + frame_id).to(tl.int64)
    frame_screen_left = (horizontal_offset - frame_left_offset * 65536) >> 16
    frame_source_x = pixel_x - frame_screen_left
    # R_DrawPSprite uses BASEYCENTER=100, weapon top 32+0x6000/FRACUNIT,
    # and the reciprocal scale of the full 320x240 target.
    frame_texture_mid = 4431872 - vertical_offset + frame_top_offset * 65536
    frame_source_y = (frame_texture_mid + (pixel_y - 103) * 54613) >> 16
    frame_inside = (
        valid
        & lane_visible
        & (frame_source_x >= 0)
        & (frame_source_x < frame_width)
        & (frame_source_y >= 0)
        & (frame_source_y < frame_height)
    )
    safe_frame_x = tl.maximum(0, tl.minimum(frame_source_x, frame_width - 1))
    safe_frame_y = tl.maximum(0, tl.minimum(frame_source_y, frame_height - 1))
    frame_atlas_index = (
        frame_id * atlas_stride_type
        + safe_frame_y * atlas_stride_y
        + safe_frame_x * atlas_stride_x
    )
    frame_value = tl.load(patch_atlas + frame_atlas_index, mask=frame_inside, other=0)
    frame_alpha = tl.load(patch_opaque + frame_atlas_index, mask=frame_inside, other=0).to(
        tl.int1
    )
    composited = tl.where(frame_inside & frame_alpha, frame_value, prior)

    raw_flash_id = tl.load(flash_ids + env, mask=valid, other=-1).to(tl.int64)
    has_flash = raw_flash_id >= 0
    flash_id = tl.maximum(raw_flash_id, 0)
    flash_width = tl.load(patch_widths + flash_id).to(tl.int64)
    flash_height = tl.load(patch_heights + flash_id).to(tl.int64)
    flash_left_offset = tl.load(patch_left_offsets + flash_id).to(tl.int64)
    flash_top_offset = tl.load(patch_top_offsets + flash_id).to(tl.int64)
    flash_screen_left = (horizontal_offset - flash_left_offset * 65536) >> 16
    flash_source_x = pixel_x - flash_screen_left
    flash_texture_mid = 4431872 - vertical_offset + flash_top_offset * 65536
    flash_source_y = (flash_texture_mid + (pixel_y - 103) * 54613) >> 16
    flash_inside = (
        valid
        & lane_visible
        & has_flash
        & (flash_source_x >= 0)
        & (flash_source_x < flash_width)
        & (flash_source_y >= 0)
        & (flash_source_y < flash_height)
    )
    safe_flash_x = tl.maximum(0, tl.minimum(flash_source_x, flash_width - 1))
    safe_flash_y = tl.maximum(0, tl.minimum(flash_source_y, flash_height - 1))
    flash_atlas_index = (
        flash_id * atlas_stride_type
        + safe_flash_y * atlas_stride_y
        + safe_flash_x * atlas_stride_x
    )
    flash_value = tl.load(patch_atlas + flash_atlas_index, mask=flash_inside, other=0)
    flash_alpha = tl.load(
        patch_opaque + flash_atlas_index,
        mask=flash_inside,
        other=0,
    ).to(tl.int1)
    tl.store(
        output + offset,
        tl.where(flash_inside & flash_alpha, flash_value, composited),
        mask=valid,
    )


@torch.library.custom_op(
    "gradoom::render_native_weapon",
    mutates_args=(),
    device_types="cuda",
)
def render_native_weapon(
    frame: torch.Tensor,
    frame_ids: torch.Tensor,
    flash_ids: torch.Tensor,
    horizontal_offsets_fixed: torch.Tensor,
    vertical_offsets_fixed: torch.Tensor,
    visible: torch.Tensor,
    patch_atlas: torch.Tensor,
    patch_opaque: torch.Tensor,
    patch_widths: torch.Tensor,
    patch_heights: torch.Tensor,
    patch_left_offsets: torch.Tensor,
    patch_top_offsets: torch.Tensor,
) -> torch.Tensor:
    """Return the exact native psprite composition without dense gather tensors."""

    output = torch.empty_like(frame)
    block = 256
    torch.library.wrap_triton(_render_native_weapon_kernel)[
        (triton.cdiv(frame.numel(), block),)
    ](
        frame,
        frame_ids,
        flash_ids,
        horizontal_offsets_fixed,
        vertical_offsets_fixed,
        visible,
        patch_atlas,
        patch_opaque,
        patch_widths,
        patch_heights,
        patch_left_offsets,
        patch_top_offsets,
        output,
        frame.numel(),
        frame.shape[1],
        frame.shape[2],
        patch_atlas.stride(0),
        patch_atlas.stride(1),
        patch_atlas.stride(2),
        block,
        num_warps=4,
    )
    return output


@render_native_weapon.register_fake
def _render_native_weapon_fake(
    frame: torch.Tensor,
    frame_ids: torch.Tensor,
    flash_ids: torch.Tensor,
    horizontal_offsets_fixed: torch.Tensor,
    vertical_offsets_fixed: torch.Tensor,
    visible: torch.Tensor,
    patch_atlas: torch.Tensor,
    patch_opaque: torch.Tensor,
    patch_widths: torch.Tensor,
    patch_heights: torch.Tensor,
    patch_left_offsets: torch.Tensor,
    patch_top_offsets: torch.Tensor,
) -> torch.Tensor:
    del (
        frame_ids,
        flash_ids,
        horizontal_offsets_fixed,
        vertical_offsets_fixed,
        visible,
        patch_atlas,
        patch_opaque,
        patch_widths,
        patch_heights,
        patch_left_offsets,
        patch_top_offsets,
    )
    return torch.empty_like(frame)


@triton.jit
def _try_enemy_chase_step_kernel(
    requested,
    direction,
    moving_type,
    enemy_x_fixed,
    enemy_y_fixed,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_alive,
    enemy_death_type,
    enemy_death_tics,
    enemy_death_elapsed,
    enemy_death_extreme,
    enemy_move_direction,
    player_x,
    player_y,
    player_z,
    player_dead,
    chase_step_x_fixed,
    chase_step_y_fixed,
    enemy_radius,
    enemy_height,
    enemy_no_block_delay,
    enemy_xdeath_no_block_delay,
    blocking_walls,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    doll_starts,
    doll_z,
    moved_output,
    enemy_slots: tl.constexpr,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    actor_index = tl.program_id(0)
    request = tl.load(requested + actor_index)
    tl.store(moved_output + actor_index, 0)
    if not request:
        return

    candidate_direction = tl.load(direction + actor_index)
    tl.store(enemy_move_direction + actor_index, candidate_direction)
    if candidate_direction >= 8:
        return

    current_x_fixed = tl.load(enemy_x_fixed + actor_index)
    current_y_fixed = tl.load(enemy_y_fixed + actor_index)
    env_index = actor_index // enemy_slots
    actor_slot = actor_index - env_index * enemy_slots
    actor_type = tl.load(moving_type + actor_index)
    delta_index = actor_type * 8 + candidate_direction
    proposed_x_fixed = current_x_fixed + tl.load(chase_step_x_fixed + delta_index)
    proposed_y_fixed = current_y_fixed + tl.load(chase_step_y_fixed + delta_index)
    proposed_x = proposed_x_fixed.to(tl.float32) / 65536.0
    proposed_y = proposed_y_fixed.to(tl.float32) / 65536.0
    radius = tl.load(enemy_radius + actor_type)
    height = tl.load(enemy_height + actor_type)
    left = proposed_x - radius
    right = proposed_x + radius
    bottom = proposed_y - radius
    top = proposed_y + radius

    # One-sided blocking-line collision.
    blocking_index = tl.arange(0, block_blocking_walls)
    valid_blocking = blocking_index < blocking_wall_count
    blocking_base = blocking_index * 4
    bx1 = tl.load(
        blocking_walls + blocking_base,
        mask=valid_blocking,
        other=0.0,
    )
    by1 = tl.load(
        blocking_walls + blocking_base + 1,
        mask=valid_blocking,
        other=0.0,
    )
    bx2 = tl.load(
        blocking_walls + blocking_base + 2,
        mask=valid_blocking,
        other=0.0,
    )
    by2 = tl.load(
        blocking_walls + blocking_base + 3,
        mask=valid_blocking,
        other=0.0,
    )
    bounds_overlap = (
        valid_blocking
        & (right > tl.minimum(bx1, bx2))
        & (left < tl.maximum(bx1, bx2))
        & (top > tl.minimum(by1, by2))
        & (bottom < tl.maximum(by1, by2))
    )
    bdx = bx2 - bx1
    bdy = by2 - by1
    side_bottom_left = bdx * (bottom - by1) - bdy * (left - bx1)
    side_bottom_right = bdx * (bottom - by1) - bdy * (right - bx1)
    side_top_left = bdx * (top - by1) - bdy * (left - bx1)
    side_top_right = bdx * (top - by1) - bdy * (right - bx1)
    minimum_side = tl.minimum(
        tl.minimum(side_bottom_left, side_bottom_right),
        tl.minimum(side_top_left, side_top_right),
    )
    maximum_side = tl.maximum(
        tl.maximum(side_bottom_left, side_bottom_right),
        tl.maximum(side_top_left, side_top_right),
    )
    collision = (
        tl.max(
            (bounds_overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)).to(tl.int32),
            axis=0,
        )
        != 0
    )

    # Sector containing the candidate center. This is the same odd/even edge
    # test as _sector_at, reduced in-place instead of materializing E x S x W.
    wall_index = tl.arange(0, block_portal_walls)
    valid_wall = wall_index < portal_wall_count
    wall_base = wall_index * 4
    x1 = tl.load(portal_walls + wall_base, mask=valid_wall, other=0.0)
    y1 = tl.load(portal_walls + wall_base + 1, mask=valid_wall, other=0.0)
    x2 = tl.load(portal_walls + wall_base + 2, mask=valid_wall, other=0.0)
    y2 = tl.load(portal_walls + wall_base + 3, mask=valid_wall, other=0.0)
    safe_dy = tl.where(tl.abs(y2 - y1) < 1.0e-6, 1.0, y2 - y1)
    crossing_x = x1 + (proposed_y - y1) * (x2 - x1) / safe_dy
    ray_crossing = valid_wall & ((y1 > proposed_y) != (y2 > proposed_y)) & (proposed_x < crossing_x)
    if sector_count < 31:
        crossing_front_sector = tl.load(
            portal_wall_sectors + wall_index * 2,
            mask=valid_wall,
            other=-1,
        )
        crossing_back_sector = tl.load(
            portal_wall_sectors + wall_index * 2 + 1,
            mask=valid_wall,
            other=-1,
        )
        crossing_front_bit = tl.where(
            crossing_front_sector >= 0,
            1 << tl.maximum(crossing_front_sector, 0),
            0,
        )
        crossing_back_bit = tl.where(
            crossing_back_sector >= 0,
            1 << tl.maximum(crossing_back_sector, 0),
            0,
        )
        crossing_sector_bits = tl.xor_sum(
            tl.where(
                ray_crossing,
                crossing_front_bit ^ crossing_back_bit,
                0,
            ),
            axis=0,
        )
        center_sector = 0
        found_sector = False
        for sector in tl.static_range(sector_count):
            inside = (crossing_sector_bits & (1 << sector)) != 0
            select_sector = inside & ~found_sector
            center_sector = tl.where(select_sector, sector, center_sector)
            found_sector = found_sector | inside
    else:
        center_sector = _sector_from_crossings(
            ray_crossing,
            sector_edge_mask,
            wall_index,
            valid_wall,
            portal_wall_count,
            sector_count,
        )

    floor = tl.load(sector_heights + center_sector * 2)
    ceiling = tl.load(sector_heights + center_sector * 2 + 1)
    dropoff = floor

    # Floors and ceilings of every sector side touched by the actor box.
    bounds_overlap = (
        valid_wall
        & (right > tl.minimum(x1, x2))
        & (left < tl.maximum(x1, x2))
        & (top > tl.minimum(y1, y2))
        & (bottom < tl.maximum(y1, y2))
    )
    dx = x2 - x1
    dy = y2 - y1
    side_bottom_left = dx * (bottom - y1) - dy * (left - x1)
    side_bottom_right = dx * (bottom - y1) - dy * (right - x1)
    side_top_left = dx * (top - y1) - dy * (left - x1)
    side_top_right = dx * (top - y1) - dy * (right - x1)
    minimum_side = tl.minimum(
        tl.minimum(side_bottom_left, side_bottom_right),
        tl.minimum(side_top_left, side_top_right),
    )
    maximum_side = tl.maximum(
        tl.maximum(side_bottom_left, side_bottom_right),
        tl.maximum(side_top_left, side_top_right),
    )
    touches_line = bounds_overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)
    front_sector = tl.load(
        portal_wall_sectors + wall_index * 2,
        mask=valid_wall,
        other=-1,
    )
    back_sector = tl.load(
        portal_wall_sectors + wall_index * 2 + 1,
        mask=valid_wall,
        other=-1,
    )
    valid_front = front_sector >= 0
    valid_back = back_sector >= 0
    safe_front = tl.maximum(front_sector, 0)
    safe_back = tl.maximum(back_sector, 0)
    front_floor = tl.load(sector_heights + safe_front * 2)
    front_ceiling = tl.load(sector_heights + safe_front * 2 + 1)
    back_floor = tl.load(sector_heights + safe_back * 2)
    back_ceiling = tl.load(sector_heights + safe_back * 2 + 1)
    touched_front = touches_line & valid_front
    touched_back = touches_line & valid_back
    touched_floor = tl.maximum(
        tl.max(tl.where(touched_front, front_floor, -float("inf")), axis=0),
        tl.max(tl.where(touched_back, back_floor, -float("inf")), axis=0),
    )
    touched_ceiling = tl.minimum(
        tl.min(tl.where(touched_front, front_ceiling, float("inf")), axis=0),
        tl.min(tl.where(touched_back, back_ceiling, float("inf")), axis=0),
    )
    touched_dropoff = tl.minimum(
        tl.min(tl.where(touched_front, front_floor, float("inf")), axis=0),
        tl.min(tl.where(touched_back, back_floor, float("inf")), axis=0),
    )
    floor = tl.maximum(floor, touched_floor)
    ceiling = tl.minimum(ceiling, touched_ceiling)
    dropoff = tl.minimum(dropoff, touched_dropoff)
    actor_z = tl.load(enemy_z + actor_index)
    collision = collision | (floor > actor_z + 24.0)
    collision = collision | (dropoff < actor_z - 24.0)
    collision = collision | (ceiling - tl.maximum(actor_z, floor) < height)

    # Other monsters, including the short-lived solid portion of death states.
    other_slot = tl.arange(0, enemy_slots)
    other_index = env_index * enemy_slots + other_slot
    other_type_raw = tl.load(enemy_type + other_index)
    other_death_type = tl.load(enemy_death_type + other_index)
    other_type = tl.maximum(
        tl.where(other_type_raw >= 0, other_type_raw, other_death_type),
        0,
    )
    other_radius = tl.load(enemy_radius + other_type)
    other_height = tl.load(enemy_height + other_type)
    other_is_alive = tl.load(enemy_alive + other_index).to(tl.int1)
    other_death_tic = tl.load(enemy_death_tics + other_index)
    other_death_elapsed_value = tl.load(enemy_death_elapsed + other_index)
    other_extreme = tl.load(enemy_death_extreme + other_index).to(tl.int1)
    normal_delay = tl.load(enemy_no_block_delay + other_type)
    extreme_delay = tl.load(enemy_xdeath_no_block_delay + other_type)
    no_block_delay = tl.where(other_extreme, extreme_delay, normal_delay)
    dying_solid = (
        (other_death_type >= 0)
        & (other_death_tic > 0)
        & (other_death_elapsed_value < no_block_delay)
    )
    other_solid = (other_is_alive | dying_solid) & (other_slot != actor_slot)
    other_height = tl.where(other_is_alive, other_height, other_height * 0.25)
    other_z = tl.load(enemy_z + other_index)
    vertical_overlap = (actor_z < other_z + other_height) & (other_z < actor_z + height)
    other_dx = proposed_x - tl.load(enemy_x + other_index)
    other_dy = proposed_y - tl.load(enemy_y + other_index)
    enemy_overlap = (
        other_solid
        & vertical_overlap
        & (tl.abs(other_dx) < radius + other_radius)
        & (tl.abs(other_dy) < radius + other_radius)
    )
    collision = collision | (tl.max(enemy_overlap.to(tl.int32), axis=0) != 0)

    # Controlled player and voodoo dolls.
    current_player_z = tl.load(player_z + env_index)
    player_overlap = (actor_z < current_player_z + 56.0) & (current_player_z < actor_z + height)
    player_dx = proposed_x - tl.load(player_x + env_index)
    player_dy = proposed_y - tl.load(player_y + env_index)
    collision = collision | (
        ~tl.load(player_dead + env_index).to(tl.int1)
        & player_overlap
        & (tl.abs(player_dx) < radius + 16.0)
        & (tl.abs(player_dy) < radius + 16.0)
    )
    for doll in tl.static_range(doll_count):
        current_doll_z = tl.load(doll_z + doll)
        doll_overlap = (actor_z < current_doll_z + 56.0) & (current_doll_z < actor_z + height)
        doll_dx = proposed_x - tl.load(doll_starts + doll * 3)
        doll_dy = proposed_y - tl.load(doll_starts + doll * 3 + 1)
        collision = collision | (
            doll_overlap & (tl.abs(doll_dx) < radius + 16.0) & (tl.abs(doll_dy) < radius + 16.0)
        )

    if collision:
        return
    tl.store(enemy_x_fixed + actor_index, proposed_x_fixed)
    tl.store(enemy_y_fixed + actor_index, proposed_y_fixed)
    tl.store(moved_output + actor_index, 1)


@triton.jit
def _normalize_enemy_xy_kernel(
    enemy_x_fixed,
    enemy_y_fixed,
    enemy_x,
    enemy_y,
    element_count: tl.constexpr,
    block_size: tl.constexpr,
):
    offset = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offset < element_count
    x_fixed = tl.load(enemy_x_fixed + offset, mask=valid)
    y_fixed = tl.load(enemy_y_fixed + offset, mask=valid)
    tl.store(
        enemy_x + offset,
        x_fixed.to(tl.float32) / 65536.0,
        mask=valid,
    )
    tl.store(
        enemy_y + offset,
        y_fixed.to(tl.float32) / 65536.0,
        mask=valid,
    )


@torch.library.custom_op(
    "gradoom::try_enemy_chase_step",
    mutates_args=(
        "enemy_x_fixed",
        "enemy_y_fixed",
        "enemy_x",
        "enemy_y",
        "enemy_move_direction",
    ),
    device_types="cuda",
)
def try_enemy_chase_step(
    requested: torch.Tensor,
    direction: torch.Tensor,
    moving_type: torch.Tensor,
    enemy_x_fixed: torch.Tensor,
    enemy_y_fixed: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_move_direction: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    player_dead: torch.Tensor,
    chase_step_x_fixed: torch.Tensor,
    chase_step_y_fixed: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> torch.Tensor:
    """Attempt sparse, branchy P_Move candidates without dense temporaries."""

    moved = torch.empty_like(requested)
    enemy_slots = requested.shape[1]
    blocking_wall_count = blocking_walls.shape[0]
    portal_wall_count = portal_walls.shape[0]
    sector_count = sector_edge_mask.shape[0]
    doll_count = doll_z.shape[0]
    grid = (requested.numel(),)
    torch.library.wrap_triton(_try_enemy_chase_step_kernel)[grid](
        requested,
        direction,
        moving_type,
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_move_direction,
        player_x,
        player_y,
        player_z,
        player_dead,
        chase_step_x_fixed,
        chase_step_y_fixed,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        doll_starts,
        doll_z,
        moved,
        enemy_slots,
        blocking_wall_count,
        portal_wall_count,
        sector_count,
        doll_count,
        triton.next_power_of_2(blocking_wall_count),
        triton.next_power_of_2(portal_wall_count),
        num_warps=2,
    )
    element_count = requested.numel()
    block_size = 256
    torch.library.wrap_triton(_normalize_enemy_xy_kernel)[
        (triton.cdiv(element_count, block_size),)
    ](
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_x,
        enemy_y,
        element_count,
        block_size,
        num_warps=4,
    )
    return moved


@try_enemy_chase_step.register_fake
def _try_enemy_chase_step_fake(
    requested: torch.Tensor,
    direction: torch.Tensor,
    moving_type: torch.Tensor,
    enemy_x_fixed: torch.Tensor,
    enemy_y_fixed: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_move_direction: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    player_dead: torch.Tensor,
    chase_step_x_fixed: torch.Tensor,
    chase_step_y_fixed: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> torch.Tensor:
    del (
        direction,
        moving_type,
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_move_direction,
        player_x,
        player_y,
        player_z,
        player_dead,
        chase_step_x_fixed,
        chase_step_y_fixed,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        doll_starts,
        doll_z,
    )
    return torch.empty_like(requested)


@triton.jit
def _enemy_hitscan_trace_kernel(
    pellet_damage,
    spread_bam,
    base_bam,
    visible,
    vertical_slope,
    maximum_horizontal_distance,
    fine_sine_fixed,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_death_type,
    enemy_alive,
    enemy_radius,
    enemy_height,
    player_x,
    player_y,
    player_z,
    player_dead,
    doll_starts,
    doll_z,
    portal_walls,
    portal_wall_sectors,
    portal_wall_blocks_sight,
    sector_heights,
    player_damage,
    actual_player_damage,
    enemy_damage,
    enemy_slots: tl.constexpr,
    pellet_count: tl.constexpr,
    wall_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_walls: tl.constexpr,
    block_actors: tl.constexpr,
):
    trace_index = tl.program_id(0)
    env_stride = enemy_slots * pellet_count
    env_index = trace_index // env_stride
    in_env = trace_index - env_index * env_stride
    attacker_slot = in_env // pellet_count
    attacker_index = env_index * enemy_slots + attacker_slot
    damage = tl.load(pellet_damage + trace_index)
    attacker_visible = tl.load(visible + attacker_index)
    if (damage <= 0.0) | ~attacker_visible:
        return

    bam = tl.load(base_bam + attacker_index) + tl.load(spread_bam + trace_index)
    fine_angle = (bam & 0xFFFFFFFF) >> 19
    sine = tl.load(fine_sine_fixed + (fine_angle & 8191)).to(tl.float32) / 65536.0
    cosine = tl.load(fine_sine_fixed + ((fine_angle + 2048) & 8191)).to(tl.float32) / 65536.0
    source_x = tl.load(enemy_x + attacker_index)
    source_y = tl.load(enemy_y + attacker_index)
    shoot_z = tl.load(enemy_z + attacker_index) + 36.0
    slope = tl.load(vertical_slope + attacker_index)
    maximum_distance = tl.load(maximum_horizontal_distance + attacker_index)

    wall_index = tl.arange(0, block_walls)
    valid_wall = wall_index < wall_count
    wall_base = wall_index * 4
    start_x = tl.load(portal_walls + wall_base, mask=valid_wall, other=0.0)
    start_y = tl.load(portal_walls + wall_base + 1, mask=valid_wall, other=0.0)
    end_x = tl.load(portal_walls + wall_base + 2, mask=valid_wall, other=0.0)
    end_y = tl.load(portal_walls + wall_base + 3, mask=valid_wall, other=0.0)
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    offset_x = start_x - source_x
    offset_y = start_y - source_y
    denominator = cosine * segment_y - sine * segment_x
    safe = tl.where(tl.abs(denominator) < 1.0e-6, 1.0, denominator)
    wall_distance = (offset_x * segment_y - offset_y * segment_x) / safe
    along_wall = (offset_x * sine - offset_y * cosine) / safe
    wall_intercept = (
        valid_wall
        & (tl.abs(denominator) >= 1.0e-6)
        & (wall_distance > 1.0e-4)
        & (along_wall >= 0.0)
        & (along_wall <= 1.0)
    )
    safe_wall_distance = tl.where(wall_intercept, wall_distance, 0.0)
    wall_hit_z = shoot_z + slope * safe_wall_distance
    front_sector = tl.load(
        portal_wall_sectors + wall_index * 2,
        mask=valid_wall,
        other=-1,
    )
    back_sector = tl.load(
        portal_wall_sectors + wall_index * 2 + 1,
        mask=valid_wall,
        other=-1,
    )
    valid_portal = (front_sector >= 0) & (back_sector >= 0)
    safe_front = tl.maximum(front_sector, 0)
    safe_back = tl.maximum(back_sector, 0)
    portal_bottom = tl.maximum(
        tl.load(sector_heights + safe_front * 2),
        tl.load(sector_heights + safe_back * 2),
    )
    portal_top = tl.minimum(
        tl.load(sector_heights + safe_front * 2 + 1),
        tl.load(sector_heights + safe_back * 2 + 1),
    )
    blocks_sight = tl.load(
        portal_wall_blocks_sight + wall_index,
        mask=valid_wall,
        other=0,
    ).to(tl.int1)
    wall_blocks_pellet = (
        wall_intercept
        & (
            blocks_sight
            | ~valid_portal
            | (wall_hit_z <= portal_bottom)
            | (wall_hit_z >= portal_top)
        )
        & (wall_distance < maximum_distance)
    )
    nearest_blocking_wall = tl.min(
        tl.where(wall_blocks_pellet, wall_distance, float("inf")),
        axis=0,
    )

    actor_count = enemy_slots + 1 + doll_count
    actor_slot = tl.arange(0, block_actors)
    valid_actor = actor_slot < actor_count
    is_enemy = actor_slot < enemy_slots
    is_player = actor_slot == enemy_slots
    is_doll = actor_slot > enemy_slots
    safe_enemy_slot = tl.minimum(actor_slot, enemy_slots - 1)
    enemy_actor_index = env_index * enemy_slots + safe_enemy_slot
    raw_type = tl.load(enemy_type + enemy_actor_index, mask=valid_actor, other=-1)
    death_type = tl.load(
        enemy_death_type + enemy_actor_index,
        mask=valid_actor,
        other=-1,
    )
    effective_type = tl.maximum(tl.where(raw_type >= 0, raw_type, death_type), 0)
    enemy_actor_x = tl.load(enemy_x + enemy_actor_index, mask=valid_actor, other=0.0)
    enemy_actor_y = tl.load(enemy_y + enemy_actor_index, mask=valid_actor, other=0.0)
    enemy_actor_z = tl.load(enemy_z + enemy_actor_index, mask=valid_actor, other=0.0)
    enemy_actor_height = tl.load(enemy_height + effective_type)
    enemy_actor_radius = tl.load(enemy_radius + effective_type)
    enemy_actor_alive = tl.load(
        enemy_alive + enemy_actor_index,
        mask=valid_actor,
        other=0,
    ).to(tl.int1)
    controlled_x = tl.load(player_x + env_index)
    controlled_y = tl.load(player_y + env_index)
    controlled_z = tl.load(player_z + env_index)
    controlled_alive = ~tl.load(player_dead + env_index).to(tl.int1)
    doll_index = tl.maximum(actor_slot - enemy_slots - 1, 0)
    doll_x = tl.load(
        doll_starts + doll_index * 3,
        mask=valid_actor & is_doll,
        other=0.0,
    )
    doll_y = tl.load(
        doll_starts + doll_index * 3 + 1,
        mask=valid_actor & is_doll,
        other=0.0,
    )
    current_doll_z = tl.load(
        doll_z + doll_index,
        mask=valid_actor & is_doll,
        other=0.0,
    )
    actor_x = tl.where(is_enemy, enemy_actor_x, tl.where(is_player, controlled_x, doll_x))
    actor_y = tl.where(is_enemy, enemy_actor_y, tl.where(is_player, controlled_y, doll_y))
    actor_z = tl.where(
        is_enemy,
        enemy_actor_z,
        tl.where(is_player, controlled_z, current_doll_z),
    )
    actor_height = tl.where(is_enemy, enemy_actor_height, 56.0)
    actor_radius = tl.where(is_enemy, enemy_actor_radius, 16.0)
    actor_alive_value = valid_actor & tl.where(
        is_enemy,
        enemy_actor_alive,
        controlled_alive,
    )

    same_sign = (cosine >= 0.0) == (sine >= 0.0)
    diagonal_x = actor_x - actor_radius
    diagonal_y = actor_y + tl.where(same_sign, actor_radius, -actor_radius)
    diagonal_dx = actor_radius * 2.0
    diagonal_dy = tl.where(same_sign, -actor_radius * 2.0, actor_radius * 2.0)
    actor_offset_x = diagonal_x - source_x
    actor_offset_y = diagonal_y - source_y
    actor_denominator = cosine * diagonal_dy - sine * diagonal_dx
    actor_safe = tl.where(
        tl.abs(actor_denominator) < 1.0e-6,
        1.0,
        actor_denominator,
    )
    actor_distance = (actor_offset_x * diagonal_dy - actor_offset_y * diagonal_dx) / actor_safe
    along_diagonal = (actor_offset_x * sine - actor_offset_y * cosine) / actor_safe
    actor_intercept = (
        valid_actor
        & (tl.abs(actor_denominator) >= 1.0e-6)
        & (actor_distance >= 0.0)
        & (along_diagonal >= 0.0)
        & (along_diagonal <= 1.0)
    )
    intercept_z = shoot_z + slope * tl.where(actor_intercept, actor_distance, 0.0)
    actor_hit = (
        actor_alive_value
        & ((actor_slot >= enemy_slots) | (actor_slot != attacker_slot))
        & actor_intercept
        & (actor_distance <= maximum_distance)
        & (actor_distance < nearest_blocking_wall)
        & (intercept_z >= actor_z)
        & (intercept_z <= actor_z + actor_height)
    )
    candidate_distance = tl.where(actor_hit, actor_distance, float("inf"))
    target = tl.argmin(candidate_distance, axis=0, tie_break_left=True)
    target_distance = tl.min(candidate_distance, axis=0)
    if target_distance == float("inf"):
        return
    if target < enemy_slots:
        enemy_output_index = (env_index * enemy_slots + attacker_slot) * enemy_slots + target
        tl.atomic_add(enemy_damage + enemy_output_index, damage)
    else:
        tl.store(player_damage + trace_index, damage)
        if target == enemy_slots:
            tl.store(actual_player_damage + trace_index, damage)


@torch.library.custom_op(
    "gradoom::enemy_hitscan_trace",
    mutates_args=(),
    device_types="cuda",
)
def enemy_hitscan_trace(
    pellet_damage: torch.Tensor,
    spread_bam: torch.Tensor,
    base_bam: torch.Tensor,
    visible: torch.Tensor,
    vertical_slope: torch.Tensor,
    maximum_horizontal_distance: torch.Tensor,
    fine_sine_fixed: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    player_dead: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    portal_wall_blocks_sight: torch.Tensor,
    sector_heights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace only active monster pellets and return the dense compatibility result."""

    player_damage = torch.zeros_like(pellet_damage)
    actual_player_damage = torch.zeros_like(pellet_damage)
    enemy_slots = enemy_x.shape[1]
    pellet_count = pellet_damage.shape[2]
    enemy_damage = torch.zeros(
        (*pellet_damage.shape[:2], enemy_slots),
        device=pellet_damage.device,
        dtype=pellet_damage.dtype,
    )
    wall_count = portal_walls.shape[0]
    doll_count = doll_z.shape[0]
    grid = (pellet_damage.numel(),)
    torch.library.wrap_triton(_enemy_hitscan_trace_kernel)[grid](
        pellet_damage,
        spread_bam,
        base_bam,
        visible,
        vertical_slope,
        maximum_horizontal_distance,
        fine_sine_fixed,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_death_type,
        enemy_alive,
        enemy_radius,
        enemy_height,
        player_x,
        player_y,
        player_z,
        player_dead,
        doll_starts,
        doll_z,
        portal_walls,
        portal_wall_sectors,
        portal_wall_blocks_sight,
        sector_heights,
        player_damage,
        actual_player_damage,
        enemy_damage,
        enemy_slots,
        pellet_count,
        wall_count,
        doll_count,
        triton.next_power_of_2(wall_count),
        triton.next_power_of_2(enemy_slots + 1 + doll_count),
        num_warps=8,
    )
    return player_damage, actual_player_damage, enemy_damage


@enemy_hitscan_trace.register_fake
def _enemy_hitscan_trace_fake(
    pellet_damage: torch.Tensor,
    spread_bam: torch.Tensor,
    base_bam: torch.Tensor,
    visible: torch.Tensor,
    vertical_slope: torch.Tensor,
    maximum_horizontal_distance: torch.Tensor,
    fine_sine_fixed: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    player_dead: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    portal_wall_blocks_sight: torch.Tensor,
    sector_heights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        spread_bam,
        base_bam,
        visible,
        vertical_slope,
        maximum_horizontal_distance,
        fine_sine_fixed,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_death_type,
        enemy_alive,
        enemy_radius,
        enemy_height,
        player_x,
        player_y,
        player_z,
        player_dead,
        doll_starts,
        doll_z,
        portal_walls,
        portal_wall_sectors,
        portal_wall_blocks_sight,
        sector_heights,
    )
    enemy_damage = pellet_damage.new_empty((*pellet_damage.shape[:2], enemy_x.shape[1]))
    return torch.empty_like(pellet_damage), torch.empty_like(pellet_damage), enemy_damage


@triton.jit
def _select_enemy_spawn_position_kernel(
    requested,
    candidate_x,
    candidate_y,
    radius_value,
    actor_z_value,
    actor_height_value,
    blocking_walls,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_alive,
    enemy_death_type,
    enemy_death_tics,
    enemy_death_elapsed,
    enemy_death_extreme,
    enemy_radius,
    enemy_height,
    enemy_no_block_delay,
    enemy_xdeath_no_block_delay,
    player_x,
    player_y,
    player_z,
    doll_starts,
    doll_z,
    fallback,
    output_x,
    output_y,
    output_valid,
    enemy_slots: tl.constexpr,
    candidate_count: tl.constexpr,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    env_index = tl.program_id(0)
    tl.store(output_x + env_index, tl.load(fallback))
    tl.store(output_y + env_index, tl.load(fallback + 1))
    tl.store(output_valid + env_index, 0)
    if not tl.load(requested + env_index):
        return
    radius = tl.load(radius_value)
    actor_z = tl.load(actor_z_value)
    actor_height = tl.load(actor_height_value)
    found = False

    for candidate in tl.static_range(candidate_count):
        x = tl.load(candidate_x + env_index * candidate_count + candidate)
        y = tl.load(candidate_y + env_index * candidate_count + candidate)
        left = x - radius
        right = x + radius
        bottom = y - radius
        top = y + radius

        blocking_index = tl.arange(0, block_blocking_walls)
        valid_blocking = blocking_index < blocking_wall_count
        blocking_base = blocking_index * 4
        bx1 = tl.load(
            blocking_walls + blocking_base,
            mask=valid_blocking,
            other=0.0,
        )
        by1 = tl.load(
            blocking_walls + blocking_base + 1,
            mask=valid_blocking,
            other=0.0,
        )
        bx2 = tl.load(
            blocking_walls + blocking_base + 2,
            mask=valid_blocking,
            other=0.0,
        )
        by2 = tl.load(
            blocking_walls + blocking_base + 3,
            mask=valid_blocking,
            other=0.0,
        )
        bounds_overlap = (
            valid_blocking
            & (right > tl.minimum(bx1, bx2))
            & (left < tl.maximum(bx1, bx2))
            & (top > tl.minimum(by1, by2))
            & (bottom < tl.maximum(by1, by2))
        )
        bdx = bx2 - bx1
        bdy = by2 - by1
        side_bottom_left = bdx * (bottom - by1) - bdy * (left - bx1)
        side_bottom_right = bdx * (bottom - by1) - bdy * (right - bx1)
        side_top_left = bdx * (top - by1) - bdy * (left - bx1)
        side_top_right = bdx * (top - by1) - bdy * (right - bx1)
        minimum_side = tl.minimum(
            tl.minimum(side_bottom_left, side_bottom_right),
            tl.minimum(side_top_left, side_top_right),
        )
        maximum_side = tl.maximum(
            tl.maximum(side_bottom_left, side_bottom_right),
            tl.maximum(side_top_left, side_top_right),
        )
        valid = (
            tl.max(
                (bounds_overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)).to(tl.int32),
                axis=0,
            )
            == 0
        )

        wall_index = tl.arange(0, block_portal_walls)
        valid_wall = wall_index < portal_wall_count
        wall_base = wall_index * 4
        x1 = tl.load(portal_walls + wall_base, mask=valid_wall, other=0.0)
        y1 = tl.load(portal_walls + wall_base + 1, mask=valid_wall, other=0.0)
        x2 = tl.load(portal_walls + wall_base + 2, mask=valid_wall, other=0.0)
        y2 = tl.load(portal_walls + wall_base + 3, mask=valid_wall, other=0.0)
        safe_dy = tl.where(tl.abs(y2 - y1) < 1.0e-6, 1.0, y2 - y1)
        crossing_x = x1 + (y - y1) * (x2 - x1) / safe_dy
        ray_crossing = valid_wall & ((y1 > y) != (y2 > y)) & (x < crossing_x)
        center_sector = _sector_from_crossings(
            ray_crossing,
            sector_edge_mask,
            wall_index,
            valid_wall,
            portal_wall_count,
            sector_count,
        )
        floor = tl.load(sector_heights + center_sector * 2)
        ceiling = tl.load(sector_heights + center_sector * 2 + 1)
        bounds_overlap = (
            valid_wall
            & (right > tl.minimum(x1, x2))
            & (left < tl.maximum(x1, x2))
            & (top > tl.minimum(y1, y2))
            & (bottom < tl.maximum(y1, y2))
        )
        dx = x2 - x1
        dy = y2 - y1
        side_bottom_left = dx * (bottom - y1) - dy * (left - x1)
        side_bottom_right = dx * (bottom - y1) - dy * (right - x1)
        side_top_left = dx * (top - y1) - dy * (left - x1)
        side_top_right = dx * (top - y1) - dy * (right - x1)
        minimum_side = tl.minimum(
            tl.minimum(side_bottom_left, side_bottom_right),
            tl.minimum(side_top_left, side_top_right),
        )
        maximum_side = tl.maximum(
            tl.maximum(side_bottom_left, side_bottom_right),
            tl.maximum(side_top_left, side_top_right),
        )
        touches_line = bounds_overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)
        front_sector = tl.load(
            portal_wall_sectors + wall_index * 2,
            mask=valid_wall,
            other=-1,
        )
        back_sector = tl.load(
            portal_wall_sectors + wall_index * 2 + 1,
            mask=valid_wall,
            other=-1,
        )
        valid_front = front_sector >= 0
        valid_back = back_sector >= 0
        safe_front = tl.maximum(front_sector, 0)
        safe_back = tl.maximum(back_sector, 0)
        front_floor = tl.load(sector_heights + safe_front * 2)
        front_ceiling = tl.load(sector_heights + safe_front * 2 + 1)
        back_floor = tl.load(sector_heights + safe_back * 2)
        back_ceiling = tl.load(sector_heights + safe_back * 2 + 1)
        touched_front = touches_line & valid_front
        touched_back = touches_line & valid_back
        touched_floor = tl.maximum(
            tl.max(
                tl.where(touched_front, front_floor, -float("inf")),
                axis=0,
            ),
            tl.max(
                tl.where(touched_back, back_floor, -float("inf")),
                axis=0,
            ),
        )
        touched_ceiling = tl.minimum(
            tl.min(
                tl.where(touched_front, front_ceiling, float("inf")),
                axis=0,
            ),
            tl.min(
                tl.where(touched_back, back_ceiling, float("inf")),
                axis=0,
            ),
        )
        floor = tl.maximum(floor, touched_floor)
        ceiling = tl.minimum(ceiling, touched_ceiling)
        valid = valid & (actor_z >= floor) & (actor_z + actor_height <= ceiling)

        other_slot = tl.arange(0, enemy_slots)
        other_index = env_index * enemy_slots + other_slot
        raw_type = tl.load(enemy_type + other_index)
        death_type = tl.load(enemy_death_type + other_index)
        effective_type = tl.maximum(
            tl.where(raw_type >= 0, raw_type, death_type),
            0,
        )
        other_radius = tl.load(enemy_radius + effective_type)
        other_height = tl.load(enemy_height + effective_type)
        other_alive = tl.load(enemy_alive + other_index).to(tl.int1)
        death_tic = tl.load(enemy_death_tics + other_index)
        death_elapsed = tl.load(enemy_death_elapsed + other_index)
        death_extreme = tl.load(enemy_death_extreme + other_index).to(tl.int1)
        normal_delay = tl.load(enemy_no_block_delay + effective_type)
        extreme_delay = tl.load(enemy_xdeath_no_block_delay + effective_type)
        no_block_delay = tl.where(death_extreme, extreme_delay, normal_delay)
        dying_solid = (death_type >= 0) & (death_tic > 0) & (death_elapsed < no_block_delay)
        other_solid = other_alive | dying_solid
        other_height = tl.where(other_alive, other_height, other_height * 0.25)
        other_z = tl.load(enemy_z + other_index)
        vertical_overlap = (actor_z < other_z + other_height) & (other_z < actor_z + actor_height)
        enemy_dx = x - tl.load(enemy_x + other_index)
        enemy_dy = y - tl.load(enemy_y + other_index)
        enemy_overlap = (
            other_solid
            & vertical_overlap
            & (tl.abs(enemy_dx) < radius + other_radius)
            & (tl.abs(enemy_dy) < radius + other_radius)
        )
        valid = valid & (tl.max(enemy_overlap.to(tl.int32), axis=0) == 0)

        current_player_z = tl.load(player_z + env_index)
        player_overlap = (actor_z < current_player_z + 56.0) & (
            current_player_z < actor_z + actor_height
        )
        player_dx = x - tl.load(player_x + env_index)
        player_dy = y - tl.load(player_y + env_index)
        player_collision = (
            player_overlap
            & (tl.abs(player_dx) < radius + 16.0)
            & (tl.abs(player_dy) < radius + 16.0)
        )
        valid = valid & ~player_collision
        for doll in tl.static_range(doll_count):
            current_doll_z = tl.load(doll_z + doll)
            doll_overlap = (actor_z < current_doll_z + 56.0) & (
                current_doll_z < actor_z + actor_height
            )
            doll_dx = x - tl.load(doll_starts + doll * 3)
            doll_dy = y - tl.load(doll_starts + doll * 3 + 1)
            valid = valid & ~(
                doll_overlap & (tl.abs(doll_dx) < radius + 16.0) & (tl.abs(doll_dy) < radius + 16.0)
            )
        select = valid & ~found
        tl.store(output_x + env_index, x, mask=select)
        tl.store(output_y + env_index, y, mask=select)
        found = found | valid
    tl.store(output_valid + env_index, found)


@torch.library.custom_op(
    "gradoom::select_enemy_spawn_position",
    mutates_args=(),
    device_types="cuda",
)
def select_enemy_spawn_position(
    requested: torch.Tensor,
    candidate_x: torch.Tensor,
    candidate_y: torch.Tensor,
    radius: torch.Tensor,
    actor_z: torch.Tensor,
    actor_height: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the first valid monster spawn candidate on requested lanes."""

    output_x = torch.empty_like(requested, dtype=torch.float32)
    output_y = torch.empty_like(requested, dtype=torch.float32)
    output_valid = torch.empty_like(requested)
    enemy_slots = enemy_x.shape[1]
    candidate_count = candidate_x.shape[1]
    blocking_wall_count = blocking_walls.shape[0]
    portal_wall_count = portal_walls.shape[0]
    sector_count = sector_edge_mask.shape[0]
    doll_count = doll_z.shape[0]
    grid = (requested.numel(),)
    torch.library.wrap_triton(_select_enemy_spawn_position_kernel)[grid](
        requested,
        candidate_x,
        candidate_y,
        radius,
        actor_z,
        actor_height,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        doll_starts,
        doll_z,
        fallback,
        output_x,
        output_y,
        output_valid,
        enemy_slots,
        candidate_count,
        blocking_wall_count,
        portal_wall_count,
        sector_count,
        doll_count,
        triton.next_power_of_2(blocking_wall_count),
        triton.next_power_of_2(portal_wall_count),
        num_warps=8,
    )
    return output_x, output_y, output_valid


@select_enemy_spawn_position.register_fake
def _select_enemy_spawn_position_fake(
    requested: torch.Tensor,
    candidate_x: torch.Tensor,
    candidate_y: torch.Tensor,
    radius: torch.Tensor,
    actor_z: torch.Tensor,
    actor_height: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        candidate_x,
        candidate_y,
        radius,
        actor_z,
        actor_height,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        doll_starts,
        doll_z,
        fallback,
    )
    return (
        torch.empty_like(requested, dtype=torch.float32),
        torch.empty_like(requested, dtype=torch.float32),
        torch.empty_like(requested),
    )


@triton.jit
def _box_collides_blocking_walls(
    x,
    y,
    radius,
    blocking_walls,
    blocking_wall_count: tl.constexpr,
    block_blocking_walls: tl.constexpr,
):
    wall = tl.arange(0, block_blocking_walls)
    valid_wall = wall < blocking_wall_count
    base = wall * 4
    x1 = tl.load(blocking_walls + base, mask=valid_wall, other=0.0)
    y1 = tl.load(blocking_walls + base + 1, mask=valid_wall, other=0.0)
    x2 = tl.load(blocking_walls + base + 2, mask=valid_wall, other=0.0)
    y2 = tl.load(blocking_walls + base + 3, mask=valid_wall, other=0.0)
    left = x - radius
    right = x + radius
    bottom = y - radius
    top = y + radius
    bounds_overlap = (
        valid_wall
        & (right > tl.minimum(x1, x2))
        & (left < tl.maximum(x1, x2))
        & (top > tl.minimum(y1, y2))
        & (bottom < tl.maximum(y1, y2))
    )
    dx = x2 - x1
    dy = y2 - y1
    side_bottom_left = dx * (bottom - y1) - dy * (left - x1)
    side_bottom_right = dx * (bottom - y1) - dy * (right - x1)
    side_top_left = dx * (top - y1) - dy * (left - x1)
    side_top_right = dx * (top - y1) - dy * (right - x1)
    minimum_side = tl.minimum(
        tl.minimum(side_bottom_left, side_bottom_right),
        tl.minimum(side_top_left, side_top_right),
    )
    maximum_side = tl.maximum(
        tl.maximum(side_bottom_left, side_bottom_right),
        tl.maximum(side_top_left, side_top_right),
    )
    return (
        tl.max(
            (bounds_overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)).to(tl.int32),
            axis=0,
        )
        != 0
    )


@triton.jit
def _sector_at_point(
    x,
    y,
    portal_walls,
    sector_edge_mask,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    wall = tl.arange(0, block_portal_walls)
    valid_wall = wall < portal_wall_count
    base = wall * 4
    x1 = tl.load(portal_walls + base, mask=valid_wall, other=0.0)
    y1 = tl.load(portal_walls + base + 1, mask=valid_wall, other=0.0)
    x2 = tl.load(portal_walls + base + 2, mask=valid_wall, other=0.0)
    y2 = tl.load(portal_walls + base + 3, mask=valid_wall, other=0.0)
    safe_dy = tl.where(tl.abs(y2 - y1) < 1.0e-6, 1.0, y2 - y1)
    crossing_x = x1 + (y - y1) * (x2 - x1) / safe_dy
    ray_crossing = valid_wall & ((y1 > y) != (y2 > y)) & (x < crossing_x)
    return _sector_from_crossings(
        ray_crossing,
        sector_edge_mask,
        wall,
        valid_wall,
        portal_wall_count,
        sector_count,
    )


@triton.jit
def _enemy_projectile_move_kernel(
    projectile_active,
    projectile_x,
    projectile_y,
    projectile_z,
    projectile_velocity_x,
    projectile_velocity_y,
    projectile_velocity_z,
    projectile_source_slot,
    blocking_walls,
    portal_walls,
    sector_edge_mask,
    sector_heights,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_alive,
    enemy_death_type,
    enemy_death_tics,
    enemy_death_elapsed,
    enemy_death_extreme,
    enemy_radius,
    enemy_height,
    enemy_no_block_delay,
    enemy_xdeath_no_block_delay,
    player_x,
    player_y,
    player_z,
    doll_starts,
    doll_z,
    output_x,
    output_y,
    output_z,
    output_impact,
    output_player_impact,
    output_doll_impact,
    output_enemy_impact,
    output_nearest_enemy,
    projectile_slots: tl.constexpr,
    enemy_slots: tl.constexpr,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
    block_enemies: tl.constexpr,
    block_dolls: tl.constexpr,
):
    projectile_index = tl.program_id(0)
    env_index = projectile_index // projectile_slots
    projectile_slot = projectile_index - env_index * projectile_slots
    start_x = tl.load(projectile_x + projectile_index)
    start_y = tl.load(projectile_y + projectile_index)
    start_z = tl.load(projectile_z + projectile_index)
    tl.store(output_x + projectile_index, start_x)
    tl.store(output_y + projectile_index, start_y)
    tl.store(output_z + projectile_index, start_z)
    tl.store(output_impact + projectile_index, 0)
    tl.store(output_player_impact + projectile_index, 0)
    tl.store(output_doll_impact + projectile_index, 0)
    tl.store(output_enemy_impact + projectile_index, 0)
    tl.store(output_nearest_enemy + projectile_index, 0)
    if not tl.load(projectile_active + projectile_index):
        return

    velocity_x = tl.load(projectile_velocity_x + projectile_index)
    velocity_y = tl.load(projectile_velocity_y + projectile_index)
    velocity_z = tl.load(projectile_velocity_z + projectile_index)
    dominant_speed = tl.maximum(tl.abs(velocity_x), tl.abs(velocity_y))
    movement_steps = tl.where(
        dominant_speed > 5.0,
        1 + tl.floor(dominant_speed / 5.0).to(tl.int32),
        1,
    )
    source_slot = tl.load(projectile_source_slot + projectile_index)
    source_slot = tl.where(source_slot >= 0, source_slot, projectile_slot)
    source_slot = tl.minimum(source_slot, enemy_slots - 1)
    current_x = start_x
    current_y = start_y
    current_z = start_z
    moving = True
    impact = False
    player_impact = False
    doll_impact = False
    enemy_impact = False
    nearest_enemy = 0

    for step in tl.static_range(1, 5):
        enabled = moving & (movement_steps >= step)
        fraction = step / movement_steps.to(tl.float32)
        candidate_x = start_x + velocity_x * fraction
        candidate_y = start_y + velocity_y * fraction
        wall_impact = enabled & _box_collides_blocking_walls(
            candidate_x,
            candidate_y,
            6.0,
            blocking_walls,
            blocking_wall_count,
            block_blocking_walls,
        )
        sector = _sector_at_point(
            candidate_x,
            candidate_y,
            portal_walls,
            sector_edge_mask,
            portal_wall_count,
            sector_count,
            block_portal_walls,
        )
        floor = tl.load(sector_heights + sector * 2)
        ceiling = tl.load(sector_heights + sector * 2 + 1)
        opening_impact = enabled & ((current_z < floor) | (current_z + 16.0 > ceiling))

        controlled_x = tl.load(player_x + env_index)
        controlled_y = tl.load(player_y + env_index)
        controlled_z = tl.load(player_z + env_index)
        player_dx = candidate_x - controlled_x
        player_dy = candidate_y - controlled_y
        player_distance = tl.sqrt(player_dx * player_dx + player_dy * player_dy)
        player_vertical_overlap = (current_z < controlled_z + 56.0) & (
            controlled_z < current_z + 16.0
        )
        step_player_impact = (
            enabled
            & player_vertical_overlap
            & (tl.abs(player_dx) < 22.0)
            & (tl.abs(player_dy) < 22.0)
        )

        doll_slot = tl.arange(0, block_dolls)
        valid_doll = doll_slot < doll_count
        doll_x = tl.load(doll_starts + doll_slot * 3, mask=valid_doll, other=0.0)
        doll_y = tl.load(
            doll_starts + doll_slot * 3 + 1,
            mask=valid_doll,
            other=0.0,
        )
        current_doll_z = tl.load(doll_z + doll_slot, mask=valid_doll, other=0.0)
        doll_dx = candidate_x - doll_x
        doll_dy = candidate_y - doll_y
        doll_distance = tl.sqrt(doll_dx * doll_dx + doll_dy * doll_dy)
        doll_vertical_overlap = (current_z < current_doll_z + 56.0) & (
            current_doll_z < current_z + 16.0
        )
        doll_candidate = (
            valid_doll
            & enabled
            & doll_vertical_overlap
            & (tl.abs(doll_dx) < 22.0)
            & (tl.abs(doll_dy) < 22.0)
        )
        nearest_doll_distance = tl.min(
            tl.where(doll_candidate, doll_distance, float("inf")),
            axis=0,
        )
        step_doll_impact = (nearest_doll_distance != float("inf")) & (
            ~step_player_impact | (nearest_doll_distance < player_distance)
        )
        step_player_impact = step_player_impact & ~step_doll_impact

        other_slot = tl.arange(0, block_enemies)
        valid_enemy = other_slot < enemy_slots
        other_index = env_index * enemy_slots + other_slot
        raw_type = tl.load(enemy_type + other_index, mask=valid_enemy, other=-1)
        death_type = tl.load(
            enemy_death_type + other_index,
            mask=valid_enemy,
            other=-1,
        )
        effective_type = tl.maximum(
            tl.where(raw_type >= 0, raw_type, death_type),
            0,
        )
        other_alive = tl.load(
            enemy_alive + other_index,
            mask=valid_enemy,
            other=0,
        ).to(tl.int1)
        death_tic = tl.load(
            enemy_death_tics + other_index,
            mask=valid_enemy,
            other=0,
        )
        death_elapsed = tl.load(
            enemy_death_elapsed + other_index,
            mask=valid_enemy,
            other=0,
        )
        death_extreme = tl.load(
            enemy_death_extreme + other_index,
            mask=valid_enemy,
            other=0,
        ).to(tl.int1)
        normal_delay = tl.load(enemy_no_block_delay + effective_type)
        extreme_delay = tl.load(enemy_xdeath_no_block_delay + effective_type)
        no_block_delay = tl.where(death_extreme, extreme_delay, normal_delay)
        dying_solid = (death_type >= 0) & (death_tic > 0) & (death_elapsed < no_block_delay)
        other_solid = valid_enemy & (other_alive | dying_solid)
        other_height = tl.load(enemy_height + effective_type)
        other_height = tl.where(other_alive, other_height, other_height * 0.25)
        other_radius = tl.load(enemy_radius + effective_type)
        other_x = tl.load(enemy_x + other_index, mask=valid_enemy, other=0.0)
        other_y = tl.load(enemy_y + other_index, mask=valid_enemy, other=0.0)
        other_z = tl.load(enemy_z + other_index, mask=valid_enemy, other=0.0)
        enemy_dx = candidate_x - other_x
        enemy_dy = candidate_y - other_y
        enemy_distance = tl.sqrt(enemy_dx * enemy_dx + enemy_dy * enemy_dy)
        enemy_vertical_overlap = (current_z < other_z + other_height) & (other_z < current_z + 16.0)
        enemy_candidate = (
            other_solid
            & enabled
            & (other_slot != source_slot)
            & enemy_vertical_overlap
            & (tl.abs(enemy_dx) < 6.0 + other_radius)
            & (tl.abs(enemy_dy) < 6.0 + other_radius)
        )
        candidate_distance = tl.where(
            enemy_candidate,
            enemy_distance,
            float("inf"),
        )
        nearest_enemy_distance = tl.min(candidate_distance, axis=0)
        step_nearest_enemy = tl.argmin(
            candidate_distance,
            axis=0,
            tie_break_left=True,
        )
        nearest_player_actor_distance = tl.where(
            step_player_impact,
            player_distance,
            tl.where(step_doll_impact, nearest_doll_distance, float("inf")),
        )
        step_enemy_impact = (nearest_enemy_distance != float("inf")) & (
            nearest_enemy_distance < nearest_player_actor_distance
        )
        step_player_impact = step_player_impact & ~step_enemy_impact
        step_doll_impact = step_doll_impact & ~step_enemy_impact
        step_actor_impact = step_player_impact | step_doll_impact | step_enemy_impact
        step_impact = enabled & (wall_impact | opening_impact | step_actor_impact)
        successful = enabled & ~step_impact
        current_x = tl.where(successful, candidate_x, current_x)
        current_y = tl.where(successful, candidate_y, current_y)
        player_impact = player_impact | (step_impact & step_player_impact)
        doll_impact = doll_impact | (step_impact & step_doll_impact)
        nearest_enemy = tl.where(
            step_impact & step_enemy_impact,
            step_nearest_enemy,
            nearest_enemy,
        )
        enemy_impact = enemy_impact | (step_impact & step_enemy_impact)
        impact = impact | step_impact
        moving = moving & ~step_impact

    next_z = current_z + velocity_z
    sector = _sector_at_point(
        current_x,
        current_y,
        portal_walls,
        sector_edge_mask,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )
    floor = tl.load(sector_heights + sector * 2)
    ceiling = tl.load(sector_heights + sector * 2 + 1)
    plane_impact = moving & ((next_z < floor) | (next_z + 16.0 > ceiling))
    clipped_z = tl.where(
        next_z < floor,
        floor,
        tl.where(next_z + 16.0 > ceiling, ceiling - 16.0, next_z),
    )
    current_z = tl.where(moving, clipped_z, current_z)
    impact = impact | plane_impact
    tl.store(output_x + projectile_index, current_x)
    tl.store(output_y + projectile_index, current_y)
    tl.store(output_z + projectile_index, current_z)
    tl.store(output_impact + projectile_index, impact)
    tl.store(output_player_impact + projectile_index, player_impact)
    tl.store(output_doll_impact + projectile_index, doll_impact)
    tl.store(output_enemy_impact + projectile_index, enemy_impact)
    tl.store(output_nearest_enemy + projectile_index, nearest_enemy)


@torch.library.custom_op(
    "gradoom::enemy_projectile_move",
    mutates_args=(),
    device_types="cuda",
)
def enemy_projectile_move(
    projectile_active: torch.Tensor,
    projectile_x: torch.Tensor,
    projectile_y: torch.Tensor,
    projectile_z: torch.Tensor,
    projectile_velocity_x: torch.Tensor,
    projectile_velocity_y: torch.Tensor,
    projectile_velocity_z: torch.Tensor,
    projectile_source_slot: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Move only live enemy projectiles and return dense compatibility outputs."""

    output_x = torch.empty_like(projectile_x)
    output_y = torch.empty_like(projectile_y)
    output_z = torch.empty_like(projectile_z)
    output_impact = torch.empty_like(projectile_active)
    output_player_impact = torch.empty_like(projectile_active)
    output_doll_impact = torch.empty_like(projectile_active)
    output_enemy_impact = torch.empty_like(projectile_active)
    output_nearest_enemy = torch.empty_like(projectile_source_slot)
    projectile_slots = projectile_x.shape[1]
    enemy_slots = enemy_x.shape[1]
    blocking_wall_count = blocking_walls.shape[0]
    portal_wall_count = portal_walls.shape[0]
    sector_count = sector_edge_mask.shape[0]
    doll_count = doll_z.shape[0]
    grid = (projectile_active.numel(),)
    torch.library.wrap_triton(_enemy_projectile_move_kernel)[grid](
        projectile_active,
        projectile_x,
        projectile_y,
        projectile_z,
        projectile_velocity_x,
        projectile_velocity_y,
        projectile_velocity_z,
        projectile_source_slot,
        blocking_walls,
        portal_walls,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        doll_starts,
        doll_z,
        output_x,
        output_y,
        output_z,
        output_impact,
        output_player_impact,
        output_doll_impact,
        output_enemy_impact,
        output_nearest_enemy,
        projectile_slots,
        enemy_slots,
        blocking_wall_count,
        portal_wall_count,
        sector_count,
        doll_count,
        triton.next_power_of_2(blocking_wall_count),
        triton.next_power_of_2(portal_wall_count),
        triton.next_power_of_2(enemy_slots),
        triton.next_power_of_2(max(doll_count, 1)),
        num_warps=8,
    )
    return (
        output_x,
        output_y,
        output_z,
        output_impact,
        output_player_impact,
        output_doll_impact,
        output_enemy_impact,
        output_nearest_enemy,
    )


@enemy_projectile_move.register_fake
def _enemy_projectile_move_fake(
    projectile_active: torch.Tensor,
    projectile_x: torch.Tensor,
    projectile_y: torch.Tensor,
    projectile_z: torch.Tensor,
    projectile_velocity_x: torch.Tensor,
    projectile_velocity_y: torch.Tensor,
    projectile_velocity_z: torch.Tensor,
    projectile_source_slot: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del (
        projectile_velocity_x,
        projectile_velocity_y,
        projectile_velocity_z,
        blocking_walls,
        portal_walls,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        doll_starts,
        doll_z,
    )
    return (
        torch.empty_like(projectile_x),
        torch.empty_like(projectile_y),
        torch.empty_like(projectile_z),
        torch.empty_like(projectile_active),
        torch.empty_like(projectile_active),
        torch.empty_like(projectile_active),
        torch.empty_like(projectile_active),
        torch.empty_like(projectile_source_slot),
    )


@triton.jit
def _player_projectile_move_kernel(
    projectile_active,
    projectile_type,
    projectile_x,
    projectile_y,
    projectile_z,
    projectile_velocity_x,
    projectile_velocity_y,
    projectile_velocity_z,
    blocking_walls,
    portal_walls,
    sector_edge_mask,
    sector_heights,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_alive,
    enemy_radius,
    enemy_height,
    player_dead,
    doll_starts,
    doll_z,
    output_x,
    output_y,
    output_z,
    output_impact,
    output_enemy_impact,
    output_doll_impact,
    output_nearest_enemy,
    projectile_slots: tl.constexpr,
    enemy_slots: tl.constexpr,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
    block_enemies: tl.constexpr,
    block_dolls: tl.constexpr,
):
    projectile_index = tl.program_id(0)
    env_index = projectile_index // projectile_slots
    start_x = tl.load(projectile_x + projectile_index)
    start_y = tl.load(projectile_y + projectile_index)
    start_z = tl.load(projectile_z + projectile_index)
    tl.store(output_x + projectile_index, start_x)
    tl.store(output_y + projectile_index, start_y)
    tl.store(output_z + projectile_index, start_z)
    tl.store(output_impact + projectile_index, 0)
    tl.store(output_enemy_impact + projectile_index, 0)
    tl.store(output_doll_impact + projectile_index, 0)
    tl.store(output_nearest_enemy + projectile_index, 0)
    if not tl.load(projectile_active + projectile_index):
        return

    kind = tl.load(projectile_type + projectile_index)
    radius = tl.where(kind == 0, 11.0, 13.0)
    max_step = radius - 1.0
    velocity_x = tl.load(projectile_velocity_x + projectile_index)
    velocity_y = tl.load(projectile_velocity_y + projectile_index)
    velocity_z = tl.load(projectile_velocity_z + projectile_index)
    dominant_speed = tl.maximum(tl.abs(velocity_x), tl.abs(velocity_y))
    movement_steps = tl.where(
        dominant_speed > max_step,
        1 + tl.floor(dominant_speed / max_step).to(tl.int32),
        1,
    )
    current_x = start_x
    current_y = start_y
    current_z = start_z
    moving = True
    impact = False
    enemy_impact = False
    doll_impact = False
    nearest_enemy = 0

    for step in tl.static_range(1, 4):
        enabled = moving & (movement_steps >= step)
        step_value = tl.full((), step, tl.float32)
        fraction = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [step_value, movement_steps.to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        delta_x = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [velocity_x, fraction],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        delta_y = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [velocity_y, fraction],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        candidate_x = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [start_x, delta_x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        candidate_y = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [start_y, delta_y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        wall_impact = enabled & _box_collides_blocking_walls(
            candidate_x,
            candidate_y,
            radius,
            blocking_walls,
            blocking_wall_count,
            block_blocking_walls,
        )
        sector = _sector_at_point(
            candidate_x,
            candidate_y,
            portal_walls,
            sector_edge_mask,
            portal_wall_count,
            sector_count,
            block_portal_walls,
        )
        floor = tl.load(sector_heights + sector * 2)
        ceiling = tl.load(sector_heights + sector * 2 + 1)
        opening_impact = enabled & ((current_z < floor) | (current_z + 8.0 > ceiling))

        other_slot = tl.arange(0, block_enemies)
        valid_enemy = other_slot < enemy_slots
        other_index = env_index * enemy_slots + other_slot
        other_type = tl.maximum(
            tl.load(enemy_type + other_index, mask=valid_enemy, other=-1),
            0,
        )
        other_alive = valid_enemy & tl.load(
            enemy_alive + other_index,
            mask=valid_enemy,
            other=0,
        ).to(tl.int1)
        other_radius = tl.load(enemy_radius + other_type)
        other_height = tl.load(enemy_height + other_type)
        other_x = tl.load(enemy_x + other_index, mask=valid_enemy, other=0.0)
        other_y = tl.load(enemy_y + other_index, mask=valid_enemy, other=0.0)
        other_z = tl.load(enemy_z + other_index, mask=valid_enemy, other=0.0)
        enemy_dx = candidate_x - other_x
        enemy_dy = candidate_y - other_y
        enemy_distance = tl.sqrt(enemy_dx * enemy_dx + enemy_dy * enemy_dy)
        enemy_vertical_overlap = (current_z < other_z + other_height) & (other_z < current_z + 8.0)
        enemy_candidate = (
            other_alive
            & enabled
            & enemy_vertical_overlap
            & (tl.abs(enemy_dx) < radius + other_radius)
            & (tl.abs(enemy_dy) < radius + other_radius)
        )
        candidate_distance = tl.where(
            enemy_candidate,
            enemy_distance,
            float("inf"),
        )
        nearest_enemy_distance = tl.min(candidate_distance, axis=0)
        step_nearest_enemy = tl.argmin(
            candidate_distance,
            axis=0,
            tie_break_left=True,
        )
        step_enemy_impact = nearest_enemy_distance != float("inf")

        doll_slot = tl.arange(0, block_dolls)
        valid_doll = doll_slot < doll_count
        doll_x = tl.load(doll_starts + doll_slot * 3, mask=valid_doll, other=0.0)
        doll_y = tl.load(
            doll_starts + doll_slot * 3 + 1,
            mask=valid_doll,
            other=0.0,
        )
        current_doll_z = tl.load(doll_z + doll_slot, mask=valid_doll, other=0.0)
        doll_dx = candidate_x - doll_x
        doll_dy = candidate_y - doll_y
        doll_distance = tl.sqrt(doll_dx * doll_dx + doll_dy * doll_dy)
        doll_vertical_overlap = (current_z < current_doll_z + 56.0) & (
            current_doll_z < current_z + 8.0
        )
        doll_candidate = (
            valid_doll
            & enabled
            & ~tl.load(player_dead + env_index).to(tl.int1)
            & doll_vertical_overlap
            & (tl.abs(doll_dx) < radius + 16.0)
            & (tl.abs(doll_dy) < radius + 16.0)
        )
        nearest_doll_distance = tl.min(
            tl.where(doll_candidate, doll_distance, float("inf")),
            axis=0,
        )
        step_doll_impact = (nearest_doll_distance != float("inf")) & (
            nearest_doll_distance < nearest_enemy_distance
        )
        step_enemy_impact = step_enemy_impact & ~step_doll_impact
        step_actor_impact = step_enemy_impact | step_doll_impact
        step_impact = enabled & (wall_impact | opening_impact | step_actor_impact)
        successful = enabled & ~step_impact
        current_x = tl.where(successful, candidate_x, current_x)
        current_y = tl.where(successful, candidate_y, current_y)
        nearest_enemy = tl.where(
            step_impact & step_enemy_impact,
            step_nearest_enemy,
            nearest_enemy,
        )
        enemy_impact = enemy_impact | (step_impact & step_enemy_impact)
        doll_impact = doll_impact | (step_impact & step_doll_impact)
        impact = impact | step_impact
        moving = moving & ~step_impact

    next_z = current_z + velocity_z
    sector = _sector_at_point(
        current_x,
        current_y,
        portal_walls,
        sector_edge_mask,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )
    floor = tl.load(sector_heights + sector * 2)
    ceiling = tl.load(sector_heights + sector * 2 + 1)
    plane_impact = moving & ((next_z < floor) | (next_z + 8.0 > ceiling))
    clipped_z = tl.where(
        next_z < floor,
        floor,
        tl.where(next_z + 8.0 > ceiling, ceiling - 8.0, next_z),
    )
    current_z = tl.where(moving, clipped_z, current_z)
    impact = impact | plane_impact
    tl.store(output_x + projectile_index, current_x)
    tl.store(output_y + projectile_index, current_y)
    tl.store(output_z + projectile_index, current_z)
    tl.store(output_impact + projectile_index, impact)
    tl.store(output_enemy_impact + projectile_index, enemy_impact)
    tl.store(output_doll_impact + projectile_index, doll_impact)
    tl.store(output_nearest_enemy + projectile_index, nearest_enemy)


@torch.library.custom_op(
    "gradoom::player_projectile_move",
    mutates_args=(),
    device_types="cuda",
)
def player_projectile_move(
    projectile_active: torch.Tensor,
    projectile_type: torch.Tensor,
    projectile_x: torch.Tensor,
    projectile_y: torch.Tensor,
    projectile_z: torch.Tensor,
    projectile_velocity_x: torch.Tensor,
    projectile_velocity_y: torch.Tensor,
    projectile_velocity_z: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    player_dead: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Move only live player projectiles and return dense compatibility outputs."""

    output_x = torch.empty_like(projectile_x)
    output_y = torch.empty_like(projectile_y)
    output_z = torch.empty_like(projectile_z)
    output_impact = torch.empty_like(projectile_active)
    output_enemy_impact = torch.empty_like(projectile_active)
    output_doll_impact = torch.empty_like(projectile_active)
    output_nearest_enemy = torch.empty_like(projectile_type)
    projectile_slots = projectile_x.shape[1]
    enemy_slots = enemy_x.shape[1]
    blocking_wall_count = blocking_walls.shape[0]
    portal_wall_count = portal_walls.shape[0]
    sector_count = sector_edge_mask.shape[0]
    doll_count = doll_z.shape[0]
    grid = (projectile_active.numel(),)
    torch.library.wrap_triton(_player_projectile_move_kernel)[grid](
        projectile_active,
        projectile_type,
        projectile_x,
        projectile_y,
        projectile_z,
        projectile_velocity_x,
        projectile_velocity_y,
        projectile_velocity_z,
        blocking_walls,
        portal_walls,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_radius,
        enemy_height,
        player_dead,
        doll_starts,
        doll_z,
        output_x,
        output_y,
        output_z,
        output_impact,
        output_enemy_impact,
        output_doll_impact,
        output_nearest_enemy,
        projectile_slots,
        enemy_slots,
        blocking_wall_count,
        portal_wall_count,
        sector_count,
        doll_count,
        triton.next_power_of_2(blocking_wall_count),
        triton.next_power_of_2(portal_wall_count),
        triton.next_power_of_2(enemy_slots),
        triton.next_power_of_2(max(doll_count, 1)),
        num_warps=8,
    )
    return (
        output_x,
        output_y,
        output_z,
        output_impact,
        output_enemy_impact,
        output_doll_impact,
        output_nearest_enemy,
    )


@player_projectile_move.register_fake
def _player_projectile_move_fake(
    projectile_active: torch.Tensor,
    projectile_type: torch.Tensor,
    projectile_x: torch.Tensor,
    projectile_y: torch.Tensor,
    projectile_z: torch.Tensor,
    projectile_velocity_x: torch.Tensor,
    projectile_velocity_y: torch.Tensor,
    projectile_velocity_z: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    player_dead: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del (
        projectile_velocity_x,
        projectile_velocity_y,
        projectile_velocity_z,
        blocking_walls,
        portal_walls,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_radius,
        enemy_height,
        player_dead,
        doll_starts,
        doll_z,
    )
    return (
        torch.empty_like(projectile_x),
        torch.empty_like(projectile_y),
        torch.empty_like(projectile_z),
        torch.empty_like(projectile_active),
        torch.empty_like(projectile_active),
        torch.empty_like(projectile_active),
        torch.empty_like(projectile_type),
    )


@triton.jit
def _xorshift32_state(value):
    value = value ^ ((value << 13) & 0xFFFFFFFF)
    value = value ^ (value >> 17)
    value = value ^ ((value << 5) & 0xFFFFFFFF)
    return value & 0xFFFFFFFF


@triton.jit
def _random_spawn_candidates_kernel(
    requested,
    rng_state,
    spawn_bounds,
    candidate_x,
    candidate_y,
    angle,
    candidate_count: tl.constexpr,
):
    env_index = tl.program_id(0)
    candidate = tl.arange(0, candidate_count)
    output_base = env_index * candidate_count + candidate
    tl.store(candidate_x + output_base, 0.0)
    tl.store(candidate_y + output_base, 0.0)
    tl.store(angle + env_index, 0.0)
    if not tl.load(requested + env_index):
        return

    low_x = tl.load(spawn_bounds)
    high_x = tl.load(spawn_bounds + 1)
    low_y = tl.load(spawn_bounds + 2)
    high_y = tl.load(spawn_bounds + 3)
    state = tl.load(rng_state + env_index)
    for index in tl.static_range(candidate_count):
        state = _xorshift32_state(state)
        unit = state.to(tl.float32) * (1.0 / 4294967296.0)
        scaled = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [unit, high_x - low_x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        value = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [low_x, scaled],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(candidate_x + env_index * candidate_count + index, value)
    for index in tl.static_range(candidate_count):
        state = _xorshift32_state(state)
        unit = state.to(tl.float32) * (1.0 / 4294967296.0)
        scaled = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [unit, high_y - low_y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        value = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [low_y, scaled],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(candidate_y + env_index * candidate_count + index, value)
    state = _xorshift32_state(state)
    unit = state.to(tl.float32) * (1.0 / 4294967296.0)
    angle_value = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [unit, 2.0 * 3.141592653589793],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(angle + env_index, angle_value)
    tl.store(rng_state + env_index, state)


@torch.library.custom_op(
    "gradoom::random_spawn_candidates",
    mutates_args=("rng_state",),
    device_types="cuda",
)
def random_spawn_candidates(
    requested: torch.Tensor,
    rng_state: torch.Tensor,
    spawn_bounds: torch.Tensor,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all masked enemy-spawn coordinates and angle in one kernel."""

    shape = (requested.numel(), candidate_count)
    candidate_x = torch.empty(shape, device=requested.device, dtype=torch.float32)
    candidate_y = torch.empty_like(candidate_x)
    angle = torch.empty(requested.shape, device=requested.device, dtype=torch.float32)
    grid = (requested.numel(),)
    torch.library.wrap_triton(_random_spawn_candidates_kernel)[grid](
        requested,
        rng_state,
        spawn_bounds,
        candidate_x,
        candidate_y,
        angle,
        candidate_count,
        num_warps=1,
    )
    return candidate_x, candidate_y, angle


@random_spawn_candidates.register_fake
def _random_spawn_candidates_fake(
    requested: torch.Tensor,
    rng_state: torch.Tensor,
    spawn_bounds: torch.Tensor,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del rng_state, spawn_bounds
    shape = (requested.numel(), candidate_count)
    candidate_x = torch.empty(shape, device=requested.device, dtype=torch.float32)
    return (
        candidate_x,
        torch.empty_like(candidate_x),
        torch.empty(requested.shape, device=requested.device, dtype=torch.float32),
    )


@triton.jit
def _actor_opening_bounds_point(
    x,
    y,
    radius,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    center_sector = _sector_at_point(
        x,
        y,
        portal_walls,
        sector_edge_mask,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )
    floor = tl.load(sector_heights + center_sector * 2)
    ceiling = tl.load(sector_heights + center_sector * 2 + 1)
    dropoff = floor
    wall = tl.arange(0, block_portal_walls)
    valid_wall = wall < portal_wall_count
    base = wall * 4
    x1 = tl.load(portal_walls + base, mask=valid_wall, other=0.0)
    y1 = tl.load(portal_walls + base + 1, mask=valid_wall, other=0.0)
    x2 = tl.load(portal_walls + base + 2, mask=valid_wall, other=0.0)
    y2 = tl.load(portal_walls + base + 3, mask=valid_wall, other=0.0)
    left = x - radius
    right = x + radius
    bottom = y - radius
    top = y + radius
    bounds_overlap = (
        valid_wall
        & (right > tl.minimum(x1, x2))
        & (left < tl.maximum(x1, x2))
        & (top > tl.minimum(y1, y2))
        & (bottom < tl.maximum(y1, y2))
    )
    dx = x2 - x1
    dy = y2 - y1
    side_bottom_left = dx * (bottom - y1) - dy * (left - x1)
    side_bottom_right = dx * (bottom - y1) - dy * (right - x1)
    side_top_left = dx * (top - y1) - dy * (left - x1)
    side_top_right = dx * (top - y1) - dy * (right - x1)
    minimum_side = tl.minimum(
        tl.minimum(side_bottom_left, side_bottom_right),
        tl.minimum(side_top_left, side_top_right),
    )
    maximum_side = tl.maximum(
        tl.maximum(side_bottom_left, side_bottom_right),
        tl.maximum(side_top_left, side_top_right),
    )
    touches = bounds_overlap & (minimum_side <= 0.0) & (maximum_side >= 0.0)
    front_sector = tl.load(
        portal_wall_sectors + wall * 2,
        mask=valid_wall,
        other=-1,
    )
    back_sector = tl.load(
        portal_wall_sectors + wall * 2 + 1,
        mask=valid_wall,
        other=-1,
    )
    valid_front = front_sector >= 0
    valid_back = back_sector >= 0
    safe_front = tl.maximum(front_sector, 0)
    safe_back = tl.maximum(back_sector, 0)
    front_floor = tl.load(sector_heights + safe_front * 2)
    front_ceiling = tl.load(sector_heights + safe_front * 2 + 1)
    back_floor = tl.load(sector_heights + safe_back * 2)
    back_ceiling = tl.load(sector_heights + safe_back * 2 + 1)
    touched_front = touches & valid_front
    touched_back = touches & valid_back
    touched_floor = tl.maximum(
        tl.max(tl.where(touched_front, front_floor, -float("inf")), axis=0),
        tl.max(tl.where(touched_back, back_floor, -float("inf")), axis=0),
    )
    touched_ceiling = tl.minimum(
        tl.min(tl.where(touched_front, front_ceiling, float("inf")), axis=0),
        tl.min(tl.where(touched_back, back_ceiling, float("inf")), axis=0),
    )
    touched_dropoff = tl.minimum(
        tl.min(tl.where(touched_front, front_floor, float("inf")), axis=0),
        tl.min(tl.where(touched_back, back_floor, float("inf")), axis=0),
    )
    return (
        tl.maximum(floor, touched_floor),
        tl.minimum(ceiling, touched_ceiling),
        tl.minimum(dropoff, touched_dropoff),
    )


@triton.jit
def _move_enemy_thrust_kernel(
    horizontal_motion,
    moving_type,
    moving_height,
    enemy_x_fixed,
    enemy_y_fixed,
    enemy_momentum_x_fixed,
    enemy_momentum_y_fixed,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_alive,
    enemy_death_type,
    enemy_death_tics,
    enemy_death_elapsed,
    enemy_death_extreme,
    enemy_radius,
    enemy_height,
    enemy_no_block_delay,
    enemy_xdeath_no_block_delay,
    player_x,
    player_y,
    player_z,
    player_dead,
    blocking_walls,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    doll_starts,
    doll_z,
    moved_output,
    floor_output,
    ceiling_output,
    enemy_slots: tl.constexpr,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
    block_enemies: tl.constexpr,
):
    actor_index = tl.program_id(0)
    tl.store(moved_output + actor_index, 0)
    tl.store(floor_output + actor_index, 0.0)
    tl.store(ceiling_output + actor_index, 0.0)
    if not tl.load(horizontal_motion + actor_index):
        return
    env_index = actor_index // enemy_slots
    actor_slot = actor_index - env_index * enemy_slots
    actor_type = tl.load(moving_type + actor_index)
    height = tl.load(moving_height + actor_index)
    current_x_fixed = tl.load(enemy_x_fixed + actor_index)
    current_y_fixed = tl.load(enemy_y_fixed + actor_index)
    proposed_x_fixed = current_x_fixed + tl.load(enemy_momentum_x_fixed + actor_index)
    proposed_y_fixed = current_y_fixed + tl.load(enemy_momentum_y_fixed + actor_index)
    proposed_x = proposed_x_fixed.to(tl.float32) / 65536.0
    proposed_y = proposed_y_fixed.to(tl.float32) / 65536.0
    radius = tl.load(enemy_radius + actor_type)
    collision = _box_collides_blocking_walls(
        proposed_x,
        proposed_y,
        radius,
        blocking_walls,
        blocking_wall_count,
        block_blocking_walls,
    )
    floor, ceiling, _dropoff = _actor_opening_bounds_point(
        proposed_x,
        proposed_y,
        radius,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )
    actor_z = tl.load(enemy_z + actor_index)
    collision = collision | (floor > actor_z + 24.0)
    collision = collision | (ceiling - tl.maximum(actor_z, floor) < height)

    other_slot = tl.arange(0, block_enemies)
    valid_enemy = other_slot < enemy_slots
    other_index = env_index * enemy_slots + other_slot
    other_type_raw = tl.load(enemy_type + other_index, mask=valid_enemy, other=-1)
    other_death_type = tl.load(
        enemy_death_type + other_index,
        mask=valid_enemy,
        other=-1,
    )
    other_type = tl.maximum(
        tl.where(other_type_raw >= 0, other_type_raw, other_death_type),
        0,
    )
    other_radius = tl.load(enemy_radius + other_type)
    other_height = tl.load(enemy_height + other_type)
    other_alive = tl.load(
        enemy_alive + other_index,
        mask=valid_enemy,
        other=0,
    ).to(tl.int1)
    other_death_tic = tl.load(
        enemy_death_tics + other_index,
        mask=valid_enemy,
        other=0,
    )
    other_death_elapsed = tl.load(
        enemy_death_elapsed + other_index,
        mask=valid_enemy,
        other=0,
    )
    other_extreme = tl.load(
        enemy_death_extreme + other_index,
        mask=valid_enemy,
        other=0,
    ).to(tl.int1)
    normal_delay = tl.load(enemy_no_block_delay + other_type)
    extreme_delay = tl.load(enemy_xdeath_no_block_delay + other_type)
    no_block_delay = tl.where(other_extreme, extreme_delay, normal_delay)
    dying_solid = (
        (other_death_type >= 0) & (other_death_tic > 0) & (other_death_elapsed < no_block_delay)
    )
    other_solid = valid_enemy & (other_alive | dying_solid) & (other_slot != actor_slot)
    other_height = tl.where(other_alive, other_height, other_height * 0.25)
    other_z = tl.load(enemy_z + other_index, mask=valid_enemy, other=0.0)
    vertical_overlap = (actor_z < other_z + other_height) & (other_z < actor_z + height)
    other_dx = proposed_x - tl.load(
        enemy_x + other_index,
        mask=valid_enemy,
        other=0.0,
    )
    other_dy = proposed_y - tl.load(
        enemy_y + other_index,
        mask=valid_enemy,
        other=0.0,
    )
    enemy_overlap = (
        other_solid
        & vertical_overlap
        & (tl.abs(other_dx) < radius + other_radius)
        & (tl.abs(other_dy) < radius + other_radius)
    )
    collision = collision | (tl.max(enemy_overlap.to(tl.int32), axis=0) != 0)

    current_player_z = tl.load(player_z + env_index)
    player_overlap = (actor_z < current_player_z + 56.0) & (current_player_z < actor_z + height)
    player_dx = proposed_x - tl.load(player_x + env_index)
    player_dy = proposed_y - tl.load(player_y + env_index)
    collision = collision | (
        ~tl.load(player_dead + env_index).to(tl.int1)
        & player_overlap
        & (tl.abs(player_dx) < radius + 16.0)
        & (tl.abs(player_dy) < radius + 16.0)
    )
    for doll in tl.static_range(doll_count):
        current_doll_z = tl.load(doll_z + doll)
        doll_overlap = (actor_z < current_doll_z + 56.0) & (current_doll_z < actor_z + height)
        doll_dx = proposed_x - tl.load(doll_starts + doll * 3)
        doll_dy = proposed_y - tl.load(doll_starts + doll * 3 + 1)
        collision = collision | (
            doll_overlap & (tl.abs(doll_dx) < radius + 16.0) & (tl.abs(doll_dy) < radius + 16.0)
        )
    if collision:
        return
    tl.store(enemy_x_fixed + actor_index, proposed_x_fixed)
    tl.store(enemy_y_fixed + actor_index, proposed_y_fixed)
    tl.store(moved_output + actor_index, 1)
    tl.store(floor_output + actor_index, floor)
    tl.store(ceiling_output + actor_index, ceiling)


@torch.library.custom_op(
    "gradoom::move_enemy_thrust",
    mutates_args=("enemy_x_fixed", "enemy_y_fixed", "enemy_x", "enemy_y"),
    device_types="cuda",
)
def move_enemy_thrust(
    horizontal_motion: torch.Tensor,
    moving_type: torch.Tensor,
    moving_height: torch.Tensor,
    enemy_x_fixed: torch.Tensor,
    enemy_y_fixed: torch.Tensor,
    enemy_momentum_x_fixed: torch.Tensor,
    enemy_momentum_y_fixed: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    player_dead: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move only monsters with nonzero damage thrust and update public XY."""

    moved = torch.empty_like(horizontal_motion)
    floor = torch.empty_like(enemy_x)
    ceiling = torch.empty_like(enemy_x)
    enemy_slots = enemy_x.shape[1]
    blocking_wall_count = blocking_walls.shape[0]
    portal_wall_count = portal_walls.shape[0]
    sector_count = sector_edge_mask.shape[0]
    doll_count = doll_z.shape[0]
    grid = (horizontal_motion.numel(),)
    torch.library.wrap_triton(_move_enemy_thrust_kernel)[grid](
        horizontal_motion,
        moving_type,
        moving_height,
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_momentum_x_fixed,
        enemy_momentum_y_fixed,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        player_dead,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        doll_starts,
        doll_z,
        moved,
        floor,
        ceiling,
        enemy_slots,
        blocking_wall_count,
        portal_wall_count,
        sector_count,
        doll_count,
        triton.next_power_of_2(blocking_wall_count),
        triton.next_power_of_2(portal_wall_count),
        triton.next_power_of_2(enemy_slots),
        num_warps=8,
    )
    element_count = enemy_x.numel()
    block_size = 256
    normalize_grid = (triton.cdiv(element_count, block_size),)
    torch.library.wrap_triton(_normalize_enemy_xy_kernel)[normalize_grid](
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_x,
        enemy_y,
        element_count,
        block_size,
        num_warps=4,
    )
    return moved, floor, ceiling


@move_enemy_thrust.register_fake
def _move_enemy_thrust_fake(
    horizontal_motion: torch.Tensor,
    moving_type: torch.Tensor,
    moving_height: torch.Tensor,
    enemy_x_fixed: torch.Tensor,
    enemy_y_fixed: torch.Tensor,
    enemy_momentum_x_fixed: torch.Tensor,
    enemy_momentum_y_fixed: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    player_dead: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        moving_type,
        moving_height,
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_momentum_x_fixed,
        enemy_momentum_y_fixed,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        player_dead,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        doll_starts,
        doll_z,
    )
    return (
        torch.empty_like(horizontal_motion),
        torch.empty_like(enemy_x),
        torch.empty_like(enemy_x),
    )


@triton.jit
def _rocket_splash_blocked_kernel(
    requested,
    origin_x,
    origin_y,
    origin_z,
    target_x,
    target_y,
    target_z,
    target_height,
    portal_walls,
    portal_wall_sectors,
    portal_wall_blocks_sight,
    sector_heights,
    wall_indices,
    wall_valid,
    blocked_output,
    grid_minimum_x: tl.constexpr,
    grid_minimum_y: tl.constexpr,
    grid_width: tl.constexpr,
    grid_height: tl.constexpr,
    grid_cell: tl.constexpr,
    projectile_slots: tl.constexpr,
    target_slots: tl.constexpr,
    request_env_stride: tl.constexpr,
    request_projectile_stride: tl.constexpr,
    request_target_stride: tl.constexpr,
    origin_env_stride: tl.constexpr,
    origin_slot_stride: tl.constexpr,
    target_x_env_stride: tl.constexpr,
    target_x_slot_stride: tl.constexpr,
    target_y_env_stride: tl.constexpr,
    target_y_slot_stride: tl.constexpr,
    target_z_env_stride: tl.constexpr,
    target_z_slot_stride: tl.constexpr,
    target_height_env_stride: tl.constexpr,
    target_height_slot_stride: tl.constexpr,
    wall_candidate_count: tl.constexpr,
    target_block: tl.constexpr,
    block_walls: tl.constexpr,
):
    target_groups = tl.cdiv(target_slots, target_block)
    program_index = tl.program_id(0)
    env_projectile = program_index // target_groups
    target_group = program_index - env_projectile * target_groups
    env_index = env_projectile // projectile_slots
    projectile_slot = env_projectile - env_index * projectile_slots
    target_slot = target_group * target_block + tl.arange(0, target_block)
    valid_target = target_slot < target_slots
    trace_index = (env_index * projectile_slots + projectile_slot) * target_slots + target_slot
    tl.store(blocked_output + trace_index, 0, mask=valid_target)
    request_index = (
        env_index * request_env_stride
        + projectile_slot * request_projectile_stride
        + target_slot * request_target_stride
    )
    request = valid_target & tl.load(
        requested + request_index,
        mask=valid_target,
        other=0,
    ).to(tl.int1)
    if tl.max(request.to(tl.int32), axis=0) == 0:
        return

    origin_index = env_index * origin_env_stride + projectile_slot * origin_slot_stride
    target_x_index = env_index * target_x_env_stride + target_slot * target_x_slot_stride
    target_y_index = env_index * target_y_env_stride + target_slot * target_y_slot_stride
    target_z_index = env_index * target_z_env_stride + target_slot * target_z_slot_stride
    target_height_index = (
        env_index * target_height_env_stride + target_slot * target_height_slot_stride
    )
    bomb_x = tl.load(origin_x + origin_index)
    bomb_y = tl.load(origin_y + origin_index)
    bomb_z = tl.load(origin_z + origin_index)
    current_target_x_value = tl.load(
        target_x + target_x_index,
        mask=valid_target,
        other=0.0,
    )
    current_target_y_value = tl.load(
        target_y + target_y_index,
        mask=valid_target,
        other=0.0,
    )
    current_target_z_value = tl.load(
        target_z + target_z_index,
        mask=valid_target,
        other=0.0,
    )
    current_target_height_value = tl.load(
        target_height + target_height_index,
        mask=valid_target,
        other=0.0,
    )
    current_target_x = current_target_x_value[:, None]
    current_target_y = current_target_y_value[:, None]

    grid_x = tl.maximum(
        0,
        tl.minimum(
            grid_width - 1,
            tl.floor((bomb_x - grid_minimum_x) / grid_cell).to(tl.int32),
        ),
    )
    grid_y = tl.maximum(
        0,
        tl.minimum(
            grid_height - 1,
            tl.floor((bomb_y - grid_minimum_y) / grid_cell).to(tl.int32),
        ),
    )
    grid_index = grid_y * grid_width + grid_x
    candidate = tl.arange(0, block_walls)[None, :]
    valid_candidate = candidate < wall_candidate_count
    lookup_index = grid_index * wall_candidate_count + candidate
    valid_wall = valid_candidate & tl.load(
        wall_valid + lookup_index,
        mask=valid_candidate,
        other=0,
    ).to(tl.int1)
    wall_index = tl.load(
        wall_indices + lookup_index,
        mask=valid_candidate,
        other=0,
    )
    wall_base = wall_index * 4
    start_x = tl.load(portal_walls + wall_base, mask=valid_wall, other=0.0)
    start_y = tl.load(portal_walls + wall_base + 1, mask=valid_wall, other=0.0)
    end_x = tl.load(portal_walls + wall_base + 2, mask=valid_wall, other=0.0)
    end_y = tl.load(portal_walls + wall_base + 3, mask=valid_wall, other=0.0)
    segment_x = end_x - start_x
    segment_y = end_y - start_y

    # P_RadiusAttack asks whether the damaged actor can see the bomb spot.
    direction_x = bomb_x - current_target_x
    direction_y = bomb_y - current_target_y
    offset_x = start_x - current_target_x
    offset_y = start_y - current_target_y
    denominator = direction_x * segment_y - direction_y * segment_x
    safe = tl.where(tl.abs(denominator) < 1.0e-6, 1.0, denominator)
    along_ray = (offset_x * segment_y - offset_y * segment_x) / safe
    along_wall = (offset_x * direction_y - offset_y * direction_x) / safe
    intersects = (
        valid_wall
        & (tl.abs(denominator) >= 1.0e-6)
        & (along_ray > 1.0e-4)
        & (along_ray < 1.0 - 1.0e-4)
        & (along_wall >= 0.0)
        & (along_wall <= 1.0)
    )

    front_sector = tl.load(
        portal_wall_sectors + wall_index * 2,
        mask=valid_wall,
        other=-1,
    )
    back_sector = tl.load(
        portal_wall_sectors + wall_index * 2 + 1,
        mask=valid_wall,
        other=-1,
    )
    valid_portal = (front_sector >= 0) & (back_sector >= 0)
    safe_front = tl.maximum(front_sector, 0)
    safe_back = tl.maximum(back_sector, 0)
    opening_bottom = tl.maximum(
        tl.load(sector_heights + safe_front * 2),
        tl.load(sector_heights + safe_back * 2),
    )
    opening_top = tl.minimum(
        tl.load(sector_heights + safe_front * 2 + 1),
        tl.load(sector_heights + safe_back * 2 + 1),
    )
    blocks_sight = tl.load(
        portal_wall_blocks_sight + wall_index,
        mask=valid_wall,
        other=0,
    ).to(tl.int1)
    solid = intersects & (blocks_sight | ~valid_portal)
    portal = intersects & ~blocks_sight & valid_portal
    safe_fraction = tl.where(portal, along_ray, 1.0)
    sight_z_value = current_target_z_value + current_target_height_value * 0.75
    sight_z = sight_z_value[:, None]
    bottom_clip = tl.where(
        portal,
        (opening_bottom - sight_z) / safe_fraction,
        -float("inf"),
    )
    top_clip = tl.where(
        portal,
        (opening_top - sight_z) / safe_fraction,
        float("inf"),
    )
    bottom_slope = tl.maximum(
        bomb_z - sight_z_value,
        tl.max(bottom_clip, axis=1),
    )
    top_slope = tl.minimum(
        bomb_z + 8.0 - sight_z_value,
        tl.min(top_clip, axis=1),
    )
    solid_blocked = tl.max(solid.to(tl.int32), axis=1) != 0
    blocked = solid_blocked | (top_slope <= bottom_slope)
    tl.store(
        blocked_output + trace_index,
        blocked & request,
        mask=valid_target,
    )


@torch.library.custom_op(
    "gradoom::rocket_splash_blocked",
    mutates_args=(),
    device_types="cuda",
)
def rocket_splash_blocked(
    requested: torch.Tensor,
    origin_x: torch.Tensor,
    origin_y: torch.Tensor,
    origin_z: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_z: torch.Tensor,
    target_height: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    portal_wall_blocks_sight: torch.Tensor,
    sector_heights: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_valid: torch.Tensor,
    grid_minimum_x: float,
    grid_minimum_y: float,
    grid_width: int,
    grid_height: int,
    grid_cell: float,
) -> torch.Tensor:
    """Evaluate only requested rocket-radius sight traces."""

    env_count, projectile_slots, target_slots = requested.shape
    blocked = torch.empty_like(requested)
    wall_candidate_count = wall_indices.shape[1]
    target_block = min(16, triton.next_power_of_2(target_slots))
    grid = (env_count * projectile_slots * triton.cdiv(target_slots, target_block),)
    torch.library.wrap_triton(_rocket_splash_blocked_kernel)[grid](
        requested,
        origin_x,
        origin_y,
        origin_z,
        target_x,
        target_y,
        target_z,
        target_height,
        portal_walls,
        portal_wall_sectors,
        portal_wall_blocks_sight,
        sector_heights,
        wall_indices,
        wall_valid,
        blocked,
        grid_minimum_x,
        grid_minimum_y,
        grid_width,
        grid_height,
        grid_cell,
        projectile_slots,
        target_slots,
        requested.stride(0),
        requested.stride(1),
        requested.stride(2),
        origin_x.stride(0),
        origin_x.stride(1),
        target_x.stride(0),
        target_x.stride(1),
        target_y.stride(0),
        target_y.stride(1),
        target_z.stride(0),
        target_z.stride(1),
        target_height.stride(0),
        target_height.stride(1),
        wall_candidate_count,
        target_block,
        triton.next_power_of_2(wall_candidate_count),
        num_warps=2,
    )
    return blocked


@rocket_splash_blocked.register_fake
def _rocket_splash_blocked_fake(
    requested: torch.Tensor,
    origin_x: torch.Tensor,
    origin_y: torch.Tensor,
    origin_z: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_z: torch.Tensor,
    target_height: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    portal_wall_blocks_sight: torch.Tensor,
    sector_heights: torch.Tensor,
    wall_indices: torch.Tensor,
    wall_valid: torch.Tensor,
    grid_minimum_x: float,
    grid_minimum_y: float,
    grid_width: int,
    grid_height: int,
    grid_cell: float,
) -> torch.Tensor:
    del (
        origin_x,
        origin_y,
        origin_z,
        target_x,
        target_y,
        target_z,
        target_height,
        portal_walls,
        portal_wall_sectors,
        portal_wall_blocks_sight,
        sector_heights,
        wall_indices,
        wall_valid,
        grid_minimum_x,
        grid_minimum_y,
        grid_width,
        grid_height,
        grid_cell,
    )
    return torch.empty_like(requested)


@triton.jit
def _sight_blocked_point(
    origin_x,
    origin_y,
    sight_z,
    target_x,
    target_y,
    target_z,
    target_height,
    portal_walls,
    portal_wall_sectors,
    portal_wall_blocks_sight,
    sector_heights,
    wall_count: tl.constexpr,
    block_walls: tl.constexpr,
):
    """Reduce the Doom portal sight-cone test for one actor/target pair."""

    direction_x = target_x - origin_x
    direction_y = target_y - origin_y
    wall_index = tl.arange(0, block_walls)
    valid_wall = wall_index < wall_count
    wall_base = wall_index * 4
    start_x = tl.load(portal_walls + wall_base, mask=valid_wall, other=0.0)
    start_y = tl.load(portal_walls + wall_base + 1, mask=valid_wall, other=0.0)
    end_x = tl.load(portal_walls + wall_base + 2, mask=valid_wall, other=0.0)
    end_y = tl.load(portal_walls + wall_base + 3, mask=valid_wall, other=0.0)
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    offset_x = start_x - origin_x
    offset_y = start_y - origin_y
    denominator = direction_x * segment_y - direction_y * segment_x
    safe = tl.where(tl.abs(denominator) < 1.0e-6, 1.0, denominator)
    along_ray = (offset_x * segment_y - offset_y * segment_x) / safe
    along_wall = (offset_x * direction_y - offset_y * direction_x) / safe
    intersects = (
        valid_wall
        & (tl.abs(denominator) >= 1.0e-6)
        & (along_ray > 1.0e-4)
        & (along_ray < 1.0 - 1.0e-4)
        & (along_wall >= 0.0)
        & (along_wall <= 1.0)
    )

    front_sector = tl.load(
        portal_wall_sectors + wall_index * 2,
        mask=valid_wall,
        other=-1,
    )
    back_sector = tl.load(
        portal_wall_sectors + wall_index * 2 + 1,
        mask=valid_wall,
        other=-1,
    )
    valid_portal = (front_sector >= 0) & (back_sector >= 0)
    safe_front = tl.maximum(front_sector, 0)
    safe_back = tl.maximum(back_sector, 0)
    opening_bottom = tl.maximum(
        tl.load(sector_heights + safe_front * 2),
        tl.load(sector_heights + safe_back * 2),
    )
    opening_top = tl.minimum(
        tl.load(sector_heights + safe_front * 2 + 1),
        tl.load(sector_heights + safe_back * 2 + 1),
    )
    blocks_sight = tl.load(
        portal_wall_blocks_sight + wall_index,
        mask=valid_wall,
        other=0,
    ).to(tl.int1)
    solid = intersects & (blocks_sight | ~valid_portal)
    portal = intersects & ~blocks_sight & valid_portal
    safe_fraction = tl.where(portal, along_ray, 1.0)
    bottom_clip = tl.where(
        portal,
        (opening_bottom - sight_z) / safe_fraction,
        -float("inf"),
    )
    top_clip = tl.where(
        portal,
        (opening_top - sight_z) / safe_fraction,
        float("inf"),
    )
    bottom_slope = tl.maximum(
        target_z - sight_z,
        tl.max(bottom_clip, axis=0),
    )
    top_slope = tl.minimum(
        target_z + target_height - sight_z,
        tl.min(top_clip, axis=0),
    )
    solid_blocked = tl.max(solid.to(tl.int32), axis=0) != 0
    return solid_blocked | (top_slope <= bottom_slope)


@triton.jit
def _enemy_sight_blocked_kernel(
    requested,
    origin_x,
    origin_y,
    sight_z,
    target_x,
    target_y,
    target_z,
    target_height,
    portal_walls,
    portal_wall_sectors,
    portal_wall_blocks_sight,
    sector_heights,
    blocked_output,
    enemy_slots: tl.constexpr,
    target_slots: tl.constexpr,
    wall_count: tl.constexpr,
    block_walls: tl.constexpr,
):
    actor_index = tl.program_id(0)
    tl.store(blocked_output + actor_index, 0)
    if not tl.load(requested + actor_index):
        return
    env_index = actor_index // enemy_slots
    actor_slot = actor_index - env_index * enemy_slots
    target_index = env_index * target_slots + tl.minimum(
        actor_slot,
        target_slots - 1,
    )
    blocked = _sight_blocked_point(
        tl.load(origin_x + actor_index),
        tl.load(origin_y + actor_index),
        tl.load(sight_z + actor_index),
        tl.load(target_x + target_index),
        tl.load(target_y + target_index),
        tl.load(target_z + target_index),
        tl.load(target_height + target_index),
        portal_walls,
        portal_wall_sectors,
        portal_wall_blocks_sight,
        sector_heights,
        wall_count,
        block_walls,
    )
    tl.store(blocked_output + actor_index, blocked)


@torch.library.custom_op(
    "gradoom::enemy_sight_blocked",
    mutates_args=(),
    device_types="cuda",
)
def enemy_sight_blocked(
    requested: torch.Tensor,
    origin_x: torch.Tensor,
    origin_y: torch.Tensor,
    sight_z: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_z: torch.Tensor,
    target_height: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    portal_wall_blocks_sight: torch.Tensor,
    sector_heights: torch.Tensor,
) -> torch.Tensor:
    """Run portal-clipped line of sight only for requested monster slots."""

    blocked = torch.empty_like(requested)
    enemy_slots = origin_x.shape[1]
    target_slots = target_x.numel() // origin_x.shape[0]
    wall_count = portal_walls.shape[0]
    grid = (requested.numel(),)
    torch.library.wrap_triton(_enemy_sight_blocked_kernel)[grid](
        requested,
        origin_x,
        origin_y,
        sight_z,
        target_x,
        target_y,
        target_z,
        target_height,
        portal_walls,
        portal_wall_sectors,
        portal_wall_blocks_sight,
        sector_heights,
        blocked,
        enemy_slots,
        target_slots,
        wall_count,
        triton.next_power_of_2(wall_count),
        num_warps=8,
    )
    return blocked


@enemy_sight_blocked.register_fake
def _enemy_sight_blocked_fake(
    requested: torch.Tensor,
    origin_x: torch.Tensor,
    origin_y: torch.Tensor,
    sight_z: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_z: torch.Tensor,
    target_height: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    portal_wall_blocks_sight: torch.Tensor,
    sector_heights: torch.Tensor,
) -> torch.Tensor:
    del (
        origin_x,
        origin_y,
        sight_z,
        target_x,
        target_y,
        target_z,
        target_height,
        portal_walls,
        portal_wall_sectors,
        portal_wall_blocks_sight,
        sector_heights,
    )
    return torch.empty_like(requested)


@triton.jit
def _initialize_enemy_spawn_kernel(
    spawn,
    slot,
    spawn_x,
    spawn_y,
    spawn_angle,
    spawn_x_fixed,
    spawn_y_fixed,
    portal_walls,
    sector_edge_mask,
    sector_heights,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_angle,
    enemy_x_fixed,
    enemy_y_fixed,
    enemy_z_fixed,
    enemy_floor_z_fixed,
    enemy_ceiling_z_fixed,
    enemy_opening_initialized,
    enemy_momentum_x_fixed,
    enemy_momentum_y_fixed,
    enemy_velocity_z_fixed,
    enemy_type,
    enemy_health,
    enemy_alive,
    enemy_cooldown,
    enemy_attack_phase,
    enemy_just_attacked,
    enemy_just_hit,
    enemy_reaction_time,
    enemy_target_slot,
    enemy_target_threshold,
    enemy_heard_player,
    enemy_move_direction,
    enemy_move_count,
    enemy_move_cooldown,
    enemy_animation_tics,
    enemy_death_type,
    enemy_death_extreme,
    enemy_death_tics,
    enemy_death_elapsed,
    drop_spawned,
    drop_velocity_x_fixed,
    drop_velocity_y_fixed,
    drop_velocity_z_fixed,
    teleport_fog_x,
    teleport_fog_y,
    teleport_fog_z,
    teleport_fog_tics,
    spawned_output,
    enemy_type_value,
    enemy_health_value,
    enemy_look_interval_value,
    enemy_slots: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    block_portal_walls: tl.constexpr,
    block_enemies: tl.constexpr,
):
    env_index = tl.program_id(0)
    should_spawn = tl.load(spawn + env_index).to(tl.int1)
    tl.store(spawned_output + env_index, should_spawn)
    if not should_spawn:
        return

    enemy_slot = tl.load(slot + env_index)
    actor_index = env_index * enemy_slots + enemy_slot
    x = tl.load(spawn_x + env_index)
    y = tl.load(spawn_y + env_index)
    angle = tl.load(spawn_angle + env_index)
    x_fixed = tl.load(spawn_x_fixed + env_index)
    y_fixed = tl.load(spawn_y_fixed + env_index)
    sector = _sector_at_point(
        x,
        y,
        portal_walls,
        sector_edge_mask,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )
    floor = tl.load(sector_heights + sector * 2)
    ceiling = tl.load(sector_heights + sector * 2 + 1)
    floor_fixed = (floor * 65536.0).to(tl.int64)
    ceiling_fixed = (ceiling * 65536.0).to(tl.int64)

    tl.store(enemy_x + actor_index, x)
    tl.store(enemy_y + actor_index, y)
    tl.store(enemy_z + actor_index, 0.0)
    tl.store(enemy_angle + actor_index, angle)
    tl.store(enemy_x_fixed + actor_index, x_fixed)
    tl.store(enemy_y_fixed + actor_index, y_fixed)
    tl.store(enemy_z_fixed + actor_index, 0)
    tl.store(enemy_floor_z_fixed + actor_index, floor_fixed)
    tl.store(enemy_ceiling_z_fixed + actor_index, ceiling_fixed)
    tl.store(enemy_opening_initialized + actor_index, 1)
    tl.store(enemy_momentum_x_fixed + actor_index, 0)
    tl.store(enemy_momentum_y_fixed + actor_index, 0)
    tl.store(
        enemy_velocity_z_fixed + actor_index,
        tl.where(floor < 0.0, -65536, 0),
    )
    tl.store(enemy_type + actor_index, enemy_type_value)
    tl.store(enemy_health + actor_index, enemy_health_value)
    tl.store(enemy_alive + actor_index, 1)
    tl.store(enemy_cooldown + actor_index, 0)
    tl.store(enemy_attack_phase + actor_index, 0)
    tl.store(enemy_just_attacked + actor_index, 0)
    tl.store(enemy_just_hit + actor_index, 0)
    tl.store(enemy_reaction_time + actor_index, 8)
    tl.store(enemy_target_slot + actor_index, -2)
    tl.store(enemy_target_threshold + actor_index, 0)
    tl.store(enemy_heard_player + actor_index, 0)
    tl.store(enemy_move_direction + actor_index, 0)
    tl.store(enemy_move_count + actor_index, 0)
    tl.store(
        enemy_move_cooldown + actor_index,
        enemy_look_interval_value - 2,
    )
    tl.store(enemy_animation_tics + actor_index, 1)
    tl.store(enemy_death_type + actor_index, -1)
    tl.store(enemy_death_extreme + actor_index, 0)
    tl.store(enemy_death_tics + actor_index, 0)
    tl.store(enemy_death_elapsed + actor_index, 0)
    tl.store(drop_spawned + actor_index, 0)
    tl.store(drop_velocity_x_fixed + actor_index, 0)
    tl.store(drop_velocity_y_fixed + actor_index, 0)
    tl.store(drop_velocity_z_fixed + actor_index, 0)

    fog_slot = tl.arange(0, block_enemies)
    valid_fog = fog_slot < enemy_slots
    fog_index = env_index * enemy_slots + fog_slot
    fog_free = valid_fog & (tl.load(teleport_fog_tics + fog_index, mask=valid_fog, other=1) <= 0)
    has_fog_slot = tl.max(fog_free.to(tl.int32), axis=0) != 0
    selected_fog_slot = tl.argmax(
        fog_free.to(tl.int32),
        axis=0,
        tie_break_left=True,
    )
    if has_fog_slot:
        selected_fog = env_index * enemy_slots + selected_fog_slot
        tl.store(teleport_fog_x + selected_fog, x)
        tl.store(teleport_fog_y + selected_fog, y)
        tl.store(teleport_fog_z + selected_fog, 0.0)
        tl.store(teleport_fog_tics + selected_fog, 71)


_SPAWN_MUTATED_ARGUMENTS = (
    "enemy_x",
    "enemy_y",
    "enemy_z",
    "enemy_angle",
    "enemy_x_fixed",
    "enemy_y_fixed",
    "enemy_z_fixed",
    "enemy_floor_z_fixed",
    "enemy_ceiling_z_fixed",
    "enemy_opening_initialized",
    "enemy_momentum_x_fixed",
    "enemy_momentum_y_fixed",
    "enemy_velocity_z_fixed",
    "enemy_type",
    "enemy_health",
    "enemy_alive",
    "enemy_cooldown",
    "enemy_attack_phase",
    "enemy_just_attacked",
    "enemy_just_hit",
    "enemy_reaction_time",
    "enemy_target_slot",
    "enemy_target_threshold",
    "enemy_heard_player",
    "enemy_move_direction",
    "enemy_move_count",
    "enemy_move_cooldown",
    "enemy_animation_tics",
    "enemy_death_type",
    "enemy_death_extreme",
    "enemy_death_tics",
    "enemy_death_elapsed",
    "drop_spawned",
    "drop_velocity_x_fixed",
    "drop_velocity_y_fixed",
    "drop_velocity_z_fixed",
    "teleport_fog_x",
    "teleport_fog_y",
    "teleport_fog_z",
    "teleport_fog_tics",
)


@torch.library.custom_op(
    "gradoom::initialize_enemy_spawn",
    mutates_args=_SPAWN_MUTATED_ARGUMENTS,
    device_types="cuda",
)
def initialize_enemy_spawn(
    spawn: torch.Tensor,
    slot: torch.Tensor,
    spawn_x: torch.Tensor,
    spawn_y: torch.Tensor,
    spawn_angle: torch.Tensor,
    spawn_x_fixed: torch.Tensor,
    spawn_y_fixed: torch.Tensor,
    portal_walls: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_angle: torch.Tensor,
    enemy_x_fixed: torch.Tensor,
    enemy_y_fixed: torch.Tensor,
    enemy_z_fixed: torch.Tensor,
    enemy_floor_z_fixed: torch.Tensor,
    enemy_ceiling_z_fixed: torch.Tensor,
    enemy_opening_initialized: torch.Tensor,
    enemy_momentum_x_fixed: torch.Tensor,
    enemy_momentum_y_fixed: torch.Tensor,
    enemy_velocity_z_fixed: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_health: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_cooldown: torch.Tensor,
    enemy_attack_phase: torch.Tensor,
    enemy_just_attacked: torch.Tensor,
    enemy_just_hit: torch.Tensor,
    enemy_reaction_time: torch.Tensor,
    enemy_target_slot: torch.Tensor,
    enemy_target_threshold: torch.Tensor,
    enemy_heard_player: torch.Tensor,
    enemy_move_direction: torch.Tensor,
    enemy_move_count: torch.Tensor,
    enemy_move_cooldown: torch.Tensor,
    enemy_animation_tics: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    drop_spawned: torch.Tensor,
    drop_velocity_x_fixed: torch.Tensor,
    drop_velocity_y_fixed: torch.Tensor,
    drop_velocity_z_fixed: torch.Tensor,
    teleport_fog_x: torch.Tensor,
    teleport_fog_y: torch.Tensor,
    teleport_fog_z: torch.Tensor,
    teleport_fog_tics: torch.Tensor,
    enemy_type_value: int,
    enemy_health_value: float,
    enemy_look_interval_value: int,
) -> torch.Tensor:
    """Initialize only successful monster spawns and their teleport fog."""

    spawned = torch.empty_like(spawn)
    enemy_slots = enemy_x.shape[1]
    portal_wall_count = portal_walls.shape[0]
    sector_count = sector_edge_mask.shape[0]
    grid = (spawn.numel(),)
    torch.library.wrap_triton(_initialize_enemy_spawn_kernel)[grid](
        spawn,
        slot,
        spawn_x,
        spawn_y,
        spawn_angle,
        spawn_x_fixed,
        spawn_y_fixed,
        portal_walls,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_angle,
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_z_fixed,
        enemy_floor_z_fixed,
        enemy_ceiling_z_fixed,
        enemy_opening_initialized,
        enemy_momentum_x_fixed,
        enemy_momentum_y_fixed,
        enemy_velocity_z_fixed,
        enemy_type,
        enemy_health,
        enemy_alive,
        enemy_cooldown,
        enemy_attack_phase,
        enemy_just_attacked,
        enemy_just_hit,
        enemy_reaction_time,
        enemy_target_slot,
        enemy_target_threshold,
        enemy_heard_player,
        enemy_move_direction,
        enemy_move_count,
        enemy_move_cooldown,
        enemy_animation_tics,
        enemy_death_type,
        enemy_death_extreme,
        enemy_death_tics,
        enemy_death_elapsed,
        drop_spawned,
        drop_velocity_x_fixed,
        drop_velocity_y_fixed,
        drop_velocity_z_fixed,
        teleport_fog_x,
        teleport_fog_y,
        teleport_fog_z,
        teleport_fog_tics,
        spawned,
        enemy_type_value,
        enemy_health_value,
        enemy_look_interval_value,
        enemy_slots,
        portal_wall_count,
        sector_count,
        triton.next_power_of_2(portal_wall_count),
        triton.next_power_of_2(enemy_slots),
        num_warps=8,
    )
    return spawned


@initialize_enemy_spawn.register_fake
def _initialize_enemy_spawn_fake(
    spawn: torch.Tensor,
    slot: torch.Tensor,
    spawn_x: torch.Tensor,
    spawn_y: torch.Tensor,
    spawn_angle: torch.Tensor,
    spawn_x_fixed: torch.Tensor,
    spawn_y_fixed: torch.Tensor,
    portal_walls: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_angle: torch.Tensor,
    enemy_x_fixed: torch.Tensor,
    enemy_y_fixed: torch.Tensor,
    enemy_z_fixed: torch.Tensor,
    enemy_floor_z_fixed: torch.Tensor,
    enemy_ceiling_z_fixed: torch.Tensor,
    enemy_opening_initialized: torch.Tensor,
    enemy_momentum_x_fixed: torch.Tensor,
    enemy_momentum_y_fixed: torch.Tensor,
    enemy_velocity_z_fixed: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_health: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_cooldown: torch.Tensor,
    enemy_attack_phase: torch.Tensor,
    enemy_just_attacked: torch.Tensor,
    enemy_just_hit: torch.Tensor,
    enemy_reaction_time: torch.Tensor,
    enemy_target_slot: torch.Tensor,
    enemy_target_threshold: torch.Tensor,
    enemy_heard_player: torch.Tensor,
    enemy_move_direction: torch.Tensor,
    enemy_move_count: torch.Tensor,
    enemy_move_cooldown: torch.Tensor,
    enemy_animation_tics: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    drop_spawned: torch.Tensor,
    drop_velocity_x_fixed: torch.Tensor,
    drop_velocity_y_fixed: torch.Tensor,
    drop_velocity_z_fixed: torch.Tensor,
    teleport_fog_x: torch.Tensor,
    teleport_fog_y: torch.Tensor,
    teleport_fog_z: torch.Tensor,
    teleport_fog_tics: torch.Tensor,
    enemy_type_value: int,
    enemy_health_value: float,
    enemy_look_interval_value: int,
) -> torch.Tensor:
    del (
        slot,
        spawn_x,
        spawn_y,
        spawn_angle,
        spawn_x_fixed,
        spawn_y_fixed,
        portal_walls,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_angle,
        enemy_x_fixed,
        enemy_y_fixed,
        enemy_z_fixed,
        enemy_floor_z_fixed,
        enemy_ceiling_z_fixed,
        enemy_opening_initialized,
        enemy_momentum_x_fixed,
        enemy_momentum_y_fixed,
        enemy_velocity_z_fixed,
        enemy_type,
        enemy_health,
        enemy_alive,
        enemy_cooldown,
        enemy_attack_phase,
        enemy_just_attacked,
        enemy_just_hit,
        enemy_reaction_time,
        enemy_target_slot,
        enemy_target_threshold,
        enemy_heard_player,
        enemy_move_direction,
        enemy_move_count,
        enemy_move_cooldown,
        enemy_animation_tics,
        enemy_death_type,
        enemy_death_extreme,
        enemy_death_tics,
        enemy_death_elapsed,
        drop_spawned,
        drop_velocity_x_fixed,
        drop_velocity_y_fixed,
        drop_velocity_z_fixed,
        teleport_fog_x,
        teleport_fog_y,
        teleport_fog_z,
        teleport_fog_tics,
        enemy_type_value,
        enemy_health_value,
        enemy_look_interval_value,
    )
    return torch.empty_like(spawn)


@triton.jit
def _enemy_spawn_requests_kernel(
    active,
    episode_time,
    next_spawn_check,
    rng_state,
    thresholds,
    requested_output,
    enemy_type_count: tl.constexpr,
):
    env_index = tl.program_id(0)
    for enemy_type in tl.static_range(enemy_type_count):
        tl.store(
            requested_output + env_index * enemy_type_count + enemy_type,
            0,
        )
    check = tl.load(active + env_index).to(tl.int1) & (
        tl.load(episode_time + env_index) >= tl.load(next_spawn_check + env_index)
    )
    if not check:
        return
    tl.store(
        next_spawn_check + env_index,
        tl.load(next_spawn_check + env_index) + 10,
    )
    state = tl.load(rng_state + env_index)
    for enemy_type in tl.static_range(enemy_type_count):
        state = _xorshift32_state(state)
        roll = state % 65537
        threshold = tl.load(thresholds + enemy_type)
        tl.store(
            requested_output + env_index * enemy_type_count + enemy_type,
            roll <= threshold,
        )
    tl.store(rng_state + env_index, state)


@torch.library.custom_op(
    "gradoom::enemy_spawn_requests",
    mutates_args=("next_spawn_check", "rng_state"),
    device_types="cuda",
)
def enemy_spawn_requests(
    active: torch.Tensor,
    episode_time: torch.Tensor,
    next_spawn_check: torch.Tensor,
    rng_state: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Advance all six ACS spawn rolls in one masked lane-local kernel."""

    requested = torch.empty(
        (active.numel(), thresholds.numel()),
        device=active.device,
        dtype=torch.bool,
    )
    grid = (active.numel(),)
    torch.library.wrap_triton(_enemy_spawn_requests_kernel)[grid](
        active,
        episode_time,
        next_spawn_check,
        rng_state,
        thresholds,
        requested,
        thresholds.numel(),
        num_warps=1,
    )
    return requested


@enemy_spawn_requests.register_fake
def _enemy_spawn_requests_fake(
    active: torch.Tensor,
    episode_time: torch.Tensor,
    next_spawn_check: torch.Tensor,
    rng_state: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    del episode_time, next_spawn_check, rng_state
    return torch.empty(
        (active.numel(), thresholds.numel()),
        device=active.device,
        dtype=torch.bool,
    )


@triton.jit
def _first_free_enemy_slot_kernel(
    enemy_alive,
    enemy_death_tics,
    drop_type,
    slot_output,
    available_output,
    enemy_slots: tl.constexpr,
    block_enemies: tl.constexpr,
):
    env_index = tl.program_id(0)
    slot = tl.arange(0, block_enemies)
    valid = slot < enemy_slots
    index = env_index * enemy_slots + slot
    free = (
        valid
        & ~tl.load(enemy_alive + index, mask=valid, other=1).to(tl.int1)
        & (tl.load(enemy_death_tics + index, mask=valid, other=1) <= 0)
        & (tl.load(drop_type + index, mask=valid, other=0) < 0)
    )
    available = tl.max(free.to(tl.int32), axis=0) != 0
    selected = tl.argmax(free.to(tl.int32), axis=0, tie_break_left=True)
    tl.store(slot_output + env_index, selected)
    tl.store(available_output + env_index, available)


@torch.library.custom_op(
    "gradoom::first_free_enemy_slot",
    mutates_args=(),
    device_types="cuda",
)
def first_free_enemy_slot(
    enemy_alive: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    drop_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first reusable monster slot in each environment lane."""

    env_count, enemy_slots = enemy_alive.shape
    slot = torch.empty(env_count, device=enemy_alive.device, dtype=torch.int64)
    available = torch.empty(env_count, device=enemy_alive.device, dtype=torch.bool)
    grid = (env_count,)
    torch.library.wrap_triton(_first_free_enemy_slot_kernel)[grid](
        enemy_alive,
        enemy_death_tics,
        drop_type,
        slot,
        available,
        enemy_slots,
        triton.next_power_of_2(enemy_slots),
        num_warps=4,
    )
    return slot, available


@first_free_enemy_slot.register_fake
def _first_free_enemy_slot_fake(
    enemy_alive: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    drop_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del enemy_death_tics, drop_type
    return (
        torch.empty(
            enemy_alive.shape[0],
            device=enemy_alive.device,
            dtype=torch.int64,
        ),
        torch.empty(
            enemy_alive.shape[0],
            device=enemy_alive.device,
            dtype=torch.bool,
        ),
    )


@triton.jit
def _enemy_spawn_point_is_valid(
    x,
    y,
    radius,
    actor_height,
    env_index,
    blocking_walls,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_alive,
    enemy_death_type,
    enemy_death_tics,
    enemy_death_elapsed,
    enemy_death_extreme,
    enemy_radius,
    enemy_height,
    enemy_no_block_delay,
    enemy_xdeath_no_block_delay,
    player_x,
    player_y,
    player_z,
    doll_starts,
    doll_z,
    planned_spawn,
    planned_x,
    planned_y,
    enemy_slots: tl.constexpr,
    enemy_type_count: tl.constexpr,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_enemies: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    valid = ~_box_collides_blocking_walls(
        x,
        y,
        radius,
        blocking_walls,
        blocking_wall_count,
        block_blocking_walls,
    )
    floor, ceiling, _dropoff = _actor_opening_bounds_point(
        x,
        y,
        radius,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        portal_wall_count,
        sector_count,
        block_portal_walls,
    )
    valid = valid & (floor <= 0.0) & (actor_height <= ceiling)

    other_slot = tl.arange(0, block_enemies)
    valid_other = other_slot < enemy_slots
    other_index = env_index * enemy_slots + other_slot
    raw_type = tl.load(enemy_type + other_index, mask=valid_other, other=-1)
    death_type = tl.load(
        enemy_death_type + other_index,
        mask=valid_other,
        other=-1,
    )
    effective_type = tl.maximum(
        tl.where(raw_type >= 0, raw_type, death_type),
        0,
    )
    other_radius = tl.load(enemy_radius + effective_type, mask=valid_other, other=0.0)
    other_height = tl.load(enemy_height + effective_type, mask=valid_other, other=0.0)
    other_alive = tl.load(
        enemy_alive + other_index,
        mask=valid_other,
        other=0,
    ).to(tl.int1)
    death_tic = tl.load(
        enemy_death_tics + other_index,
        mask=valid_other,
        other=0,
    )
    death_elapsed = tl.load(
        enemy_death_elapsed + other_index,
        mask=valid_other,
        other=0,
    )
    death_extreme = tl.load(
        enemy_death_extreme + other_index,
        mask=valid_other,
        other=0,
    ).to(tl.int1)
    normal_delay = tl.load(
        enemy_no_block_delay + effective_type,
        mask=valid_other,
        other=0,
    )
    extreme_delay = tl.load(
        enemy_xdeath_no_block_delay + effective_type,
        mask=valid_other,
        other=0,
    )
    no_block_delay = tl.where(death_extreme, extreme_delay, normal_delay)
    dying_solid = (death_type >= 0) & (death_tic > 0) & (death_elapsed < no_block_delay)
    other_solid = valid_other & (other_alive | dying_solid)
    other_height = tl.where(other_alive, other_height, other_height * 0.25)
    other_z = tl.load(enemy_z + other_index, mask=valid_other, other=0.0)
    vertical_overlap = (other_z + other_height > 0.0) & (other_z < actor_height)
    enemy_dx = x - tl.load(enemy_x + other_index, mask=valid_other, other=0.0)
    enemy_dy = y - tl.load(enemy_y + other_index, mask=valid_other, other=0.0)
    enemy_overlap = (
        other_solid
        & vertical_overlap
        & (tl.abs(enemy_dx) < radius + other_radius)
        & (tl.abs(enemy_dy) < radius + other_radius)
    )
    valid = valid & (tl.max(enemy_overlap.to(tl.int32), axis=0) == 0)

    plan_base = env_index * enemy_type_count
    for planned_type in tl.static_range(enemy_type_count):
        planned_index = plan_base + planned_type
        occupied = tl.load(planned_spawn + planned_index).to(tl.int1)
        planned_dx = x - tl.load(planned_x + planned_index)
        planned_dy = y - tl.load(planned_y + planned_index)
        planned_radius = tl.load(enemy_radius + planned_type)
        valid = valid & ~(
            occupied
            & (tl.abs(planned_dx) < radius + planned_radius)
            & (tl.abs(planned_dy) < radius + planned_radius)
        )

    current_player_z = tl.load(player_z + env_index)
    player_overlap = (current_player_z + 56.0 > 0.0) & (current_player_z < actor_height)
    player_dx = x - tl.load(player_x + env_index)
    player_dy = y - tl.load(player_y + env_index)
    valid = valid & ~(
        player_overlap & (tl.abs(player_dx) < radius + 16.0) & (tl.abs(player_dy) < radius + 16.0)
    )
    for doll in tl.static_range(doll_count):
        current_doll_z = tl.load(doll_z + doll)
        doll_overlap = (current_doll_z + 56.0 > 0.0) & (current_doll_z < actor_height)
        doll_dx = x - tl.load(doll_starts + doll * 3)
        doll_dy = y - tl.load(doll_starts + doll * 3 + 1)
        valid = valid & ~(
            doll_overlap & (tl.abs(doll_dx) < radius + 16.0) & (tl.abs(doll_dy) < radius + 16.0)
        )
    return valid


@triton.jit
def _enemy_spawn_plan_kernel(
    active,
    episode_time,
    next_spawn_check,
    rng_state,
    thresholds,
    spawn_bounds,
    blocking_walls,
    portal_walls,
    portal_wall_sectors,
    sector_edge_mask,
    sector_heights,
    enemy_x,
    enemy_y,
    enemy_z,
    enemy_type,
    enemy_alive,
    enemy_death_type,
    enemy_death_tics,
    enemy_death_elapsed,
    enemy_death_extreme,
    drop_type,
    enemy_radius,
    enemy_height,
    enemy_no_block_delay,
    enemy_xdeath_no_block_delay,
    player_x,
    player_y,
    player_z,
    doll_starts,
    doll_z,
    fallback,
    request_scratch,
    candidate_x_scratch,
    candidate_y_scratch,
    spawn_output,
    slot_output,
    x_output,
    y_output,
    angle_output,
    enemy_slots: tl.constexpr,
    enemy_type_count: tl.constexpr,
    candidate_count: tl.constexpr,
    blocking_wall_count: tl.constexpr,
    portal_wall_count: tl.constexpr,
    sector_count: tl.constexpr,
    doll_count: tl.constexpr,
    block_enemies: tl.constexpr,
    block_blocking_walls: tl.constexpr,
    block_portal_walls: tl.constexpr,
):
    env_index = tl.program_id(0)
    plan_base = env_index * enemy_type_count
    fallback_x = tl.load(fallback)
    fallback_y = tl.load(fallback + 1)
    for kind in tl.static_range(enemy_type_count):
        output_index = plan_base + kind
        tl.store(spawn_output + output_index, 0)
        tl.store(slot_output + output_index, 0)
        tl.store(x_output + output_index, fallback_x)
        tl.store(y_output + output_index, fallback_y)
        tl.store(angle_output + output_index, 0.0)

    check = tl.load(active + env_index).to(tl.int1) & (
        tl.load(episode_time + env_index) >= tl.load(next_spawn_check + env_index)
    )
    if not check:
        return
    tl.store(
        next_spawn_check + env_index,
        tl.load(next_spawn_check + env_index) + 10,
    )

    state = tl.load(rng_state + env_index)
    for kind in tl.static_range(enemy_type_count):
        state = _xorshift32_state(state)
        requested = (state % 65537) <= tl.load(thresholds + kind)
        tl.store(request_scratch + plan_base + kind, requested)

    low_x = tl.load(spawn_bounds)
    high_x = tl.load(spawn_bounds + 1)
    low_y = tl.load(spawn_bounds + 2)
    high_y = tl.load(spawn_bounds + 3)
    candidate_base = env_index * candidate_count
    enemy_slot = tl.arange(0, block_enemies)
    valid_slot = enemy_slot < enemy_slots
    actor_base = env_index * enemy_slots

    for kind in tl.static_range(enemy_type_count):
        requested = tl.load(request_scratch + plan_base + kind).to(tl.int1)
        actor_index = actor_base + enemy_slot
        free = (
            valid_slot
            & ~tl.load(
                enemy_alive + actor_index,
                mask=valid_slot,
                other=1,
            ).to(tl.int1)
            & (
                tl.load(
                    enemy_death_tics + actor_index,
                    mask=valid_slot,
                    other=1,
                )
                <= 0
            )
            & (tl.load(drop_type + actor_index, mask=valid_slot, other=0) < 0)
        )
        for prior_kind in tl.static_range(enemy_type_count):
            prior_index = plan_base + prior_kind
            reserved = tl.load(spawn_output + prior_index).to(tl.int1) & (
                enemy_slot == tl.load(slot_output + prior_index)
            )
            free = free & ~reserved
        has_free_slot = tl.max(free.to(tl.int32), axis=0) != 0
        selected_slot = tl.argmax(
            free.to(tl.int32),
            axis=0,
            tie_break_left=True,
        )
        attempt = requested & has_free_slot
        if attempt:
            for candidate in tl.static_range(candidate_count):
                state = _xorshift32_state(state)
                unit = state.to(tl.float32) * (1.0 / 4294967296.0)
                scaled = tl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    [unit, high_x - low_x],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                value = tl.inline_asm_elementwise(
                    "add.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    [low_x, scaled],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                tl.store(candidate_x_scratch + candidate_base + candidate, value)
            for candidate in tl.static_range(candidate_count):
                state = _xorshift32_state(state)
                unit = state.to(tl.float32) * (1.0 / 4294967296.0)
                scaled = tl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    [unit, high_y - low_y],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                value = tl.inline_asm_elementwise(
                    "add.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    [low_y, scaled],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                tl.store(candidate_y_scratch + candidate_base + candidate, value)
            state = _xorshift32_state(state)
            angle_unit = state.to(tl.float32) * (1.0 / 4294967296.0)
            angle = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [angle_unit, 2.0 * 3.141592653589793],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            radius = tl.load(enemy_radius + kind)
            actor_height = tl.load(enemy_height + kind)
            found = False
            selected_x = fallback_x
            selected_y = fallback_y
            for candidate in tl.static_range(candidate_count):
                candidate_x = tl.load(candidate_x_scratch + candidate_base + candidate)
                candidate_y = tl.load(candidate_y_scratch + candidate_base + candidate)
                valid = _enemy_spawn_point_is_valid(
                    candidate_x,
                    candidate_y,
                    radius,
                    actor_height,
                    env_index,
                    blocking_walls,
                    portal_walls,
                    portal_wall_sectors,
                    sector_edge_mask,
                    sector_heights,
                    enemy_x,
                    enemy_y,
                    enemy_z,
                    enemy_type,
                    enemy_alive,
                    enemy_death_type,
                    enemy_death_tics,
                    enemy_death_elapsed,
                    enemy_death_extreme,
                    enemy_radius,
                    enemy_height,
                    enemy_no_block_delay,
                    enemy_xdeath_no_block_delay,
                    player_x,
                    player_y,
                    player_z,
                    doll_starts,
                    doll_z,
                    spawn_output,
                    x_output,
                    y_output,
                    enemy_slots,
                    enemy_type_count,
                    blocking_wall_count,
                    portal_wall_count,
                    sector_count,
                    doll_count,
                    block_enemies,
                    block_blocking_walls,
                    block_portal_walls,
                )
                select = valid & ~found
                selected_x = tl.where(select, candidate_x, selected_x)
                selected_y = tl.where(select, candidate_y, selected_y)
                found = found | valid
            output_index = plan_base + kind
            tl.store(spawn_output + output_index, found)
            tl.store(slot_output + output_index, selected_slot)
            tl.store(x_output + output_index, selected_x)
            tl.store(y_output + output_index, selected_y)
            tl.store(angle_output + output_index, angle)
    tl.store(rng_state + env_index, state)


@torch.library.custom_op(
    "gradoom::enemy_spawn_plan",
    mutates_args=("next_spawn_check", "rng_state"),
    device_types="cuda",
)
def enemy_spawn_plan(
    active: torch.Tensor,
    episode_time: torch.Tensor,
    next_spawn_check: torch.Tensor,
    rng_state: torch.Tensor,
    thresholds: torch.Tensor,
    spawn_bounds: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    drop_type: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan all six ordered ACS monster spawns in one lane-local kernel."""

    env_count, enemy_slots = enemy_alive.shape
    enemy_type_count = thresholds.numel()
    candidate_count = 16
    shape = (env_count, enemy_type_count)
    request_scratch = torch.empty(shape, device=active.device, dtype=torch.bool)
    candidate_shape = (env_count, candidate_count)
    candidate_x_scratch = torch.empty(
        candidate_shape,
        device=active.device,
        dtype=torch.float32,
    )
    candidate_y_scratch = torch.empty_like(candidate_x_scratch)
    spawn = torch.empty(shape, device=active.device, dtype=torch.bool)
    slot = torch.empty(shape, device=active.device, dtype=torch.int64)
    x = torch.empty(shape, device=active.device, dtype=torch.float32)
    y = torch.empty_like(x)
    angle = torch.empty_like(x)
    blocking_wall_count = blocking_walls.shape[0]
    portal_wall_count = portal_walls.shape[0]
    sector_count = sector_edge_mask.shape[0]
    doll_count = doll_z.shape[0]
    grid = (env_count,)
    torch.library.wrap_triton(_enemy_spawn_plan_kernel)[grid](
        active,
        episode_time,
        next_spawn_check,
        rng_state,
        thresholds,
        spawn_bounds,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        drop_type,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        doll_starts,
        doll_z,
        fallback,
        request_scratch,
        candidate_x_scratch,
        candidate_y_scratch,
        spawn,
        slot,
        x,
        y,
        angle,
        enemy_slots,
        enemy_type_count,
        candidate_count,
        blocking_wall_count,
        portal_wall_count,
        sector_count,
        doll_count,
        triton.next_power_of_2(enemy_slots),
        triton.next_power_of_2(blocking_wall_count),
        triton.next_power_of_2(portal_wall_count),
        num_warps=8,
    )
    return spawn, slot, x, y, angle


@enemy_spawn_plan.register_fake
def _enemy_spawn_plan_fake(
    active: torch.Tensor,
    episode_time: torch.Tensor,
    next_spawn_check: torch.Tensor,
    rng_state: torch.Tensor,
    thresholds: torch.Tensor,
    spawn_bounds: torch.Tensor,
    blocking_walls: torch.Tensor,
    portal_walls: torch.Tensor,
    portal_wall_sectors: torch.Tensor,
    sector_edge_mask: torch.Tensor,
    sector_heights: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_z: torch.Tensor,
    enemy_type: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_death_type: torch.Tensor,
    enemy_death_tics: torch.Tensor,
    enemy_death_elapsed: torch.Tensor,
    enemy_death_extreme: torch.Tensor,
    drop_type: torch.Tensor,
    enemy_radius: torch.Tensor,
    enemy_height: torch.Tensor,
    enemy_no_block_delay: torch.Tensor,
    enemy_xdeath_no_block_delay: torch.Tensor,
    player_x: torch.Tensor,
    player_y: torch.Tensor,
    player_z: torch.Tensor,
    doll_starts: torch.Tensor,
    doll_z: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        episode_time,
        next_spawn_check,
        rng_state,
        spawn_bounds,
        blocking_walls,
        portal_walls,
        portal_wall_sectors,
        sector_edge_mask,
        sector_heights,
        enemy_x,
        enemy_y,
        enemy_z,
        enemy_type,
        enemy_alive,
        enemy_death_type,
        enemy_death_tics,
        enemy_death_elapsed,
        enemy_death_extreme,
        drop_type,
        enemy_radius,
        enemy_height,
        enemy_no_block_delay,
        enemy_xdeath_no_block_delay,
        player_x,
        player_y,
        player_z,
        doll_starts,
        doll_z,
        fallback,
    )
    shape = (active.numel(), thresholds.numel())
    spawn = torch.empty(shape, device=active.device, dtype=torch.bool)
    slot = torch.empty(shape, device=active.device, dtype=torch.int64)
    x = torch.empty(shape, device=active.device, dtype=torch.float32)
    return spawn, slot, x, torch.empty_like(x), torch.empty_like(x)
