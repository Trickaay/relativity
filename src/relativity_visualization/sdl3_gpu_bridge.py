"""
Off-screen SDL3 GPU renderer -- originally built to bridge a larger,
dormant GLSL shader library (a spectral path-tracing renderer, blocked
because macOS caps OpenGL at 4.1 with no compute-shader support) into a
pygame-ce application; this repo repurposes it as the shared GPU device
underneath every renderer here (GPUPlanetRenderer, GPUCelestialRenderer,
GPUBVHPipeline).

pygame-ce runs on SDL2; SDL3's GPU API (used here) is a completely
different native library with no shared window/device -- so this
renders entirely OFF-SCREEN (a headless GPU device + a color texture,
no visible SDL3 window or swapchain at all), downloads the rendered
texture to CPU as pixels each frame, and hands that back as a numpy
array for the caller to composite into a real pygame.Surface via
pygame.image.frombuffer().

Uses the manual glslang (GLSL->SPIR-V) -> spirv-cross (SPIR-V->MSL text)
-> SDL_CreateGPUShader(format=MSL) path, NOT the SDL_ShaderCross helper
API -- tried that first, but `SDL_ShaderCross_Init()` raised "Invoked
an unimplemented function": the installed PySDL3 build declares the
ShaderCross bindings but the native library backing them doesn't
actually implement them, even though an earlier scratch test of
ShaderCross for a COMPUTE shader looked promising on paper -- it was
never actually the one that worked. Since resource counts aren't
reflected automatically without ShaderCross, they're hardcoded here per
shader (matches this module's own known shader pair, not general).

SDL_GPU's documented SPIR-V binding convention (used by the paired
.vert/.frag shaders this loads): set 0/1 = vertex textures/uniforms,
set 2/3 = fragment textures/uniforms.
"""

import ctypes

import numpy as np
import sdl3


class SDL3GPUOffscreenRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._keepalive = []  # ctypes buffers must outlive the calls that reference them

        assert sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO), sdl3.SDL_GetError()

        self.device = sdl3.SDL_CreateGPUDevice(sdl3.SDL_GPU_SHADERFORMAT_MSL, True, None)
        assert self.device, sdl3.SDL_GetError()

        self.color_format = sdl3.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM
        tex_info = sdl3.SDL_GPUTextureCreateInfo(
            type=sdl3.SDL_GPU_TEXTURETYPE_2D,
            format=self.color_format,
            usage=sdl3.SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
            width=width, height=height, layer_count_or_depth=1, num_levels=1,
            sample_count=sdl3.SDL_GPU_SAMPLECOUNT_1, props=0,
        )
        self.color_texture = sdl3.SDL_CreateGPUTexture(self.device, ctypes.byref(tex_info))
        assert self.color_texture, sdl3.SDL_GetError()

        transfer_info = sdl3.SDL_GPUTransferBufferCreateInfo(
            usage=sdl3.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, size=width * height * 4, props=0,
        )
        self.transfer_buffer = sdl3.SDL_CreateGPUTransferBuffer(self.device, ctypes.byref(transfer_info))
        assert self.transfer_buffer, sdl3.SDL_GetError()

        self.pipeline = None

    def close(self):
        """Releases the GPU device and shuts down SDL video. Not calling
        this between short-lived renderer instances created within the
        same process (e.g. one per pytest test) was confirmed to cause
        real cross-instance interference -- a second pipeline's compute
        dispatches produced a subtly wrong (though not obviously crashing)
        result while an earlier instance's device/resources were still
        alive, only surfacing as a correctness failure, not an exception."""
        sdl3.SDL_WaitForGPUIdle(self.device)
        sdl3.SDL_DestroyGPUDevice(self.device)
        sdl3.SDL_Quit()

    def _compile_shader(self, msl_path, stage, num_uniform_buffers=0, num_samplers=0, num_storage_buffers=0):
        with open(msl_path, "rb") as f:
            msl_bytes = f.read()
        buf = (ctypes.c_ubyte * len(msl_bytes)).from_buffer_copy(msl_bytes)
        self._keepalive.append(buf)
        code_ptr = ctypes.cast(buf, sdl3.LP_c_ubyte)

        shader_info = sdl3.SDL_GPUShaderCreateInfo(
            code_size=len(msl_bytes), code=code_ptr, entrypoint=b"main0",
            format=sdl3.SDL_GPU_SHADERFORMAT_MSL, stage=stage,
            num_samplers=num_samplers, num_storage_textures=0, num_storage_buffers=num_storage_buffers,
            num_uniform_buffers=num_uniform_buffers, props=0,
        )
        shader = sdl3.SDL_CreateGPUShader(self.device, ctypes.byref(shader_info))
        assert shader, "shader compile failed: " + str(sdl3.SDL_GetError())
        return shader

    def load_shader_pair(self, vert_msl_path, frag_msl_path, frag_uniform_buffers=1, frag_samplers=0,
                          frag_storage_buffers=0):
        vert_shader = self._compile_shader(vert_msl_path, sdl3.SDL_GPU_SHADERSTAGE_VERTEX)
        frag_shader = self._compile_shader(frag_msl_path, sdl3.SDL_GPU_SHADERSTAGE_FRAGMENT,
                                            num_uniform_buffers=frag_uniform_buffers,
                                            num_samplers=frag_samplers,
                                            num_storage_buffers=frag_storage_buffers)

        color_target_desc = sdl3.SDL_GPUColorTargetDescription(
            format=self.color_format, blend_state=sdl3.SDL_GPUColorTargetBlendState())
        target_info = sdl3.SDL_GPUGraphicsPipelineTargetInfo(
            color_target_descriptions=ctypes.pointer(color_target_desc),
            num_color_targets=1, depth_stencil_format=0, has_depth_stencil_target=False,
        )
        pipeline_info = sdl3.SDL_GPUGraphicsPipelineCreateInfo(
            vertex_shader=vert_shader,
            fragment_shader=frag_shader,
            vertex_input_state=sdl3.SDL_GPUVertexInputState(),  # no vertex buffers -- fullscreen-triangle trick
            primitive_type=sdl3.SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=sdl3.SDL_GPURasterizerState(
                fill_mode=sdl3.SDL_GPU_FILLMODE_FILL, cull_mode=sdl3.SDL_GPU_CULLMODE_NONE,
                front_face=sdl3.SDL_GPU_FRONTFACE_CLOCKWISE),
            multisample_state=sdl3.SDL_GPUMultisampleState(sample_count=sdl3.SDL_GPU_SAMPLECOUNT_1),
            depth_stencil_state=sdl3.SDL_GPUDepthStencilState(),
            target_info=target_info,
            props=0,
        )
        self.pipeline = sdl3.SDL_CreateGPUGraphicsPipeline(self.device, ctypes.byref(pipeline_info))
        assert self.pipeline, "graphics pipeline creation failed: " + str(sdl3.SDL_GetError())

    def upload_texture3d(self, data):
        """data: float32 numpy array shaped (depth, height, width).
        Returns (texture, sampler) ready to pass as
        render_frame(fragment_texture_sampler=...)."""
        depth, height, width = data.shape
        tex_info = sdl3.SDL_GPUTextureCreateInfo(
            type=sdl3.SDL_GPU_TEXTURETYPE_3D,
            format=sdl3.SDL_GPU_TEXTUREFORMAT_R32_FLOAT,
            usage=sdl3.SDL_GPU_TEXTUREUSAGE_SAMPLER,
            width=width, height=height, layer_count_or_depth=depth, num_levels=1,
            sample_count=sdl3.SDL_GPU_SAMPLECOUNT_1, props=0,
        )
        texture = sdl3.SDL_CreateGPUTexture(self.device, ctypes.byref(tex_info))
        assert texture, sdl3.SDL_GetError()

        data_bytes = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        upload_info = sdl3.SDL_GPUTransferBufferCreateInfo(
            usage=sdl3.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, size=len(data_bytes), props=0,
        )
        upload_buffer = sdl3.SDL_CreateGPUTransferBuffer(self.device, ctypes.byref(upload_info))
        assert upload_buffer, sdl3.SDL_GetError()

        ptr = sdl3.SDL_MapGPUTransferBuffer(self.device, upload_buffer, False)
        assert ptr, sdl3.SDL_GetError()
        ctypes.memmove(ptr, data_bytes, len(data_bytes))
        sdl3.SDL_UnmapGPUTransferBuffer(self.device, upload_buffer)

        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        copy_pass = sdl3.SDL_BeginGPUCopyPass(cmd)
        src_info = sdl3.SDL_GPUTextureTransferInfo(
            transfer_buffer=upload_buffer, offset=0, pixels_per_row=width, rows_per_layer=height)
        dst_region = sdl3.SDL_GPUTextureRegion(
            texture=texture, mip_level=0, layer=0, x=0, y=0, z=0, w=width, h=height, d=depth)
        sdl3.SDL_UploadToGPUTexture(copy_pass, ctypes.byref(src_info), ctypes.byref(dst_region), False)
        sdl3.SDL_EndGPUCopyPass(copy_pass)
        assert sdl3.SDL_SubmitGPUCommandBuffer(cmd), sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)

        sampler_info = sdl3.SDL_GPUSamplerCreateInfo(
            min_filter=sdl3.SDL_GPU_FILTER_LINEAR, mag_filter=sdl3.SDL_GPU_FILTER_LINEAR,
            mipmap_mode=sdl3.SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
            address_mode_u=sdl3.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_v=sdl3.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_w=sdl3.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            mip_lod_bias=0.0, max_anisotropy=1.0, compare_op=0,
            min_lod=0.0, max_lod=0.0, enable_anisotropy=False, enable_compare=False,
        )
        sampler = sdl3.SDL_CreateGPUSampler(self.device, ctypes.byref(sampler_info))
        assert sampler, sdl3.SDL_GetError()
        return texture, sampler

    def create_compute_pipeline(self, msl_path, num_readonly_storage_textures=0,
                                 num_readwrite_storage_textures=0, num_uniform_buffers=0,
                                 num_readwrite_storage_buffers=0,
                                 threadcount=(8, 8, 8)):
        """Compute pipelines are independent of the graphics pipeline
        above (self.pipeline) -- returns the SDL_GPUComputePipeline
        directly rather than storing it on self, since a renderer may
        want more than one (unlike the single graphics pipeline the
        fullscreen-triangle path assumes).

        `num_readwrite_storage_buffers` covers ALL storage buffers a
        kernel touches, not just the ones it writes to -- see
        dispatch_compute()'s docstring for why (spirv-cross doesn't
        preserve the readonly/readwrite distinction into the compiled
        MSL for buffers the way it does for storage textures, confirmed
        empirically while building relativity_kernel_dsl; binding a read-only
        buffer through the read-write category is still fully correct
        since the compiled shader body simply never writes to it)."""
        with open(msl_path, "rb") as f:
            msl_bytes = f.read()
        buf = (ctypes.c_ubyte * len(msl_bytes)).from_buffer_copy(msl_bytes)
        self._keepalive.append(buf)
        code_ptr = ctypes.cast(buf, sdl3.LP_c_ubyte)

        pipeline_info = sdl3.SDL_GPUComputePipelineCreateInfo(
            code_size=len(msl_bytes), code=code_ptr, entrypoint=b"main0",
            format=sdl3.SDL_GPU_SHADERFORMAT_MSL,
            num_samplers=0,
            num_readonly_storage_textures=num_readonly_storage_textures,
            num_readonly_storage_buffers=0,
            num_readwrite_storage_textures=num_readwrite_storage_textures,
            num_readwrite_storage_buffers=num_readwrite_storage_buffers,
            num_uniform_buffers=num_uniform_buffers,
            threadcount_x=threadcount[0], threadcount_y=threadcount[1], threadcount_z=threadcount[2],
            props=0,
        )
        pipeline = sdl3.SDL_CreateGPUComputePipeline(self.device, ctypes.byref(pipeline_info))
        assert pipeline, "compute pipeline creation failed: " + str(sdl3.SDL_GetError())
        return pipeline

    def create_storage_buffer(self, nbytes, initial_data=None, graphics_readable=False):
        """Allocates an SDL_GPUBuffer usable as a compute read-write
        storage buffer (and also as a graphics/fragment-stage readable
        storage buffer if `graphics_readable=True`, needed for e.g. a
        raytrace fragment shader reading a BVH-node buffer built by an
        earlier compute pass). Mirrors create_storage_texture3d's
        create-then-optionally-upload shape."""
        usage = sdl3.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | sdl3.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE
        if graphics_readable:
            usage |= sdl3.SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ
        buf_info = sdl3.SDL_GPUBufferCreateInfo(usage=usage, size=nbytes, props=0)
        buffer = sdl3.SDL_CreateGPUBuffer(self.device, ctypes.byref(buf_info))
        assert buffer, sdl3.SDL_GetError()
        if initial_data is not None:
            self.upload_buffer_data(buffer, initial_data)
        return buffer

    def upload_buffer_data(self, buffer, data):
        data_bytes = np.ascontiguousarray(data).tobytes()
        upload_info = sdl3.SDL_GPUTransferBufferCreateInfo(
            usage=sdl3.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, size=len(data_bytes), props=0)
        upload_buffer = sdl3.SDL_CreateGPUTransferBuffer(self.device, ctypes.byref(upload_info))
        assert upload_buffer, sdl3.SDL_GetError()

        ptr = sdl3.SDL_MapGPUTransferBuffer(self.device, upload_buffer, False)
        assert ptr, sdl3.SDL_GetError()
        ctypes.memmove(ptr, data_bytes, len(data_bytes))
        sdl3.SDL_UnmapGPUTransferBuffer(self.device, upload_buffer)

        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        copy_pass = sdl3.SDL_BeginGPUCopyPass(cmd)
        src_loc = sdl3.SDL_GPUTransferBufferLocation(transfer_buffer=upload_buffer, offset=0)
        dst_region = sdl3.SDL_GPUBufferRegion(buffer=buffer, offset=0, size=len(data_bytes))
        sdl3.SDL_UploadToGPUBuffer(copy_pass, ctypes.byref(src_loc), ctypes.byref(dst_region), False)
        sdl3.SDL_EndGPUCopyPass(copy_pass)
        assert sdl3.SDL_SubmitGPUCommandBuffer(cmd), sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)
        sdl3.SDL_ReleaseGPUTransferBuffer(self.device, upload_buffer)

    def download_buffer(self, buffer, nbytes, dtype=np.float32):
        transfer_info = sdl3.SDL_GPUTransferBufferCreateInfo(
            usage=sdl3.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, size=nbytes, props=0)
        transfer_buffer = sdl3.SDL_CreateGPUTransferBuffer(self.device, ctypes.byref(transfer_info))
        assert transfer_buffer, sdl3.SDL_GetError()

        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        copy_pass = sdl3.SDL_BeginGPUCopyPass(cmd)
        src_region = sdl3.SDL_GPUBufferRegion(buffer=buffer, offset=0, size=nbytes)
        dst_loc = sdl3.SDL_GPUTransferBufferLocation(transfer_buffer=transfer_buffer, offset=0)
        sdl3.SDL_DownloadFromGPUBuffer(copy_pass, ctypes.byref(src_region), ctypes.byref(dst_loc))
        sdl3.SDL_EndGPUCopyPass(copy_pass)
        assert sdl3.SDL_SubmitGPUCommandBuffer(cmd), sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)

        ptr = sdl3.SDL_MapGPUTransferBuffer(self.device, transfer_buffer, False)
        assert ptr, sdl3.SDL_GetError()
        raw = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte * nbytes)).contents
        arr = np.frombuffer(bytes(raw), dtype=dtype).copy()
        sdl3.SDL_UnmapGPUTransferBuffer(self.device, transfer_buffer)
        sdl3.SDL_ReleaseGPUTransferBuffer(self.device, transfer_buffer)
        return arr

    def create_storage_texture3d(self, width, height, depth, gpu_format, initial_data=None):
        """A 3D texture usable for compute read/write (as opposed to
        upload_texture3d()'s sampler-usage texture for fragment reads).
        `gpu_format` is one of the SDL_GPU_TEXTUREFORMAT_* constants --
        callers pick to match their shader (e.g. R32_FLOAT for
        solidMask, R32G32B32A32_FLOAT for velocity, matching
        boundary.comp's r32f/rgba32f image formats)."""
        tex_info = sdl3.SDL_GPUTextureCreateInfo(
            type=sdl3.SDL_GPU_TEXTURETYPE_3D,
            format=gpu_format,
            usage=(sdl3.SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ
                   | sdl3.SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE),
            width=width, height=height, layer_count_or_depth=depth, num_levels=1,
            sample_count=sdl3.SDL_GPU_SAMPLECOUNT_1, props=0,
        )
        texture = sdl3.SDL_CreateGPUTexture(self.device, ctypes.byref(tex_info))
        assert texture, sdl3.SDL_GetError()
        if initial_data is not None:
            self._upload_texture3d_data(texture, width, height, depth, initial_data)
        return texture

    def _upload_texture3d_data(self, texture, width, height, depth, data):
        data_bytes = np.ascontiguousarray(data).tobytes()
        upload_info = sdl3.SDL_GPUTransferBufferCreateInfo(
            usage=sdl3.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, size=len(data_bytes), props=0)
        upload_buffer = sdl3.SDL_CreateGPUTransferBuffer(self.device, ctypes.byref(upload_info))
        assert upload_buffer, sdl3.SDL_GetError()

        ptr = sdl3.SDL_MapGPUTransferBuffer(self.device, upload_buffer, False)
        assert ptr, sdl3.SDL_GetError()
        ctypes.memmove(ptr, data_bytes, len(data_bytes))
        sdl3.SDL_UnmapGPUTransferBuffer(self.device, upload_buffer)

        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        copy_pass = sdl3.SDL_BeginGPUCopyPass(cmd)
        src_info = sdl3.SDL_GPUTextureTransferInfo(
            transfer_buffer=upload_buffer, offset=0, pixels_per_row=width, rows_per_layer=height)
        dst_region = sdl3.SDL_GPUTextureRegion(
            texture=texture, mip_level=0, layer=0, x=0, y=0, z=0, w=width, h=height, d=depth)
        sdl3.SDL_UploadToGPUTexture(copy_pass, ctypes.byref(src_info), ctypes.byref(dst_region), False)
        sdl3.SDL_EndGPUCopyPass(copy_pass)
        assert sdl3.SDL_SubmitGPUCommandBuffer(cmd), sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)

    def download_texture3d(self, texture, width, height, depth, channels, dtype=np.float32):
        """Reads a 3D texture back to CPU as a (depth, height, width,
        channels) numpy array (channels squeezed out if 1)."""
        itemsize = np.dtype(dtype).itemsize
        n_bytes = width * height * depth * channels * itemsize
        transfer_info = sdl3.SDL_GPUTransferBufferCreateInfo(
            usage=sdl3.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, size=n_bytes, props=0)
        transfer_buffer = sdl3.SDL_CreateGPUTransferBuffer(self.device, ctypes.byref(transfer_info))
        assert transfer_buffer, sdl3.SDL_GetError()

        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        copy_pass = sdl3.SDL_BeginGPUCopyPass(cmd)
        src_region = sdl3.SDL_GPUTextureRegion(
            texture=texture, mip_level=0, layer=0, x=0, y=0, z=0, w=width, h=height, d=depth)
        dst_info = sdl3.SDL_GPUTextureTransferInfo(
            transfer_buffer=transfer_buffer, offset=0, pixels_per_row=width, rows_per_layer=height)
        sdl3.SDL_DownloadFromGPUTexture(copy_pass, ctypes.byref(src_region), ctypes.byref(dst_info))
        sdl3.SDL_EndGPUCopyPass(copy_pass)
        assert sdl3.SDL_SubmitGPUCommandBuffer(cmd), sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)

        ptr = sdl3.SDL_MapGPUTransferBuffer(self.device, transfer_buffer, False)
        assert ptr, sdl3.SDL_GetError()
        raw = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte * n_bytes)).contents
        arr = np.frombuffer(bytes(raw), dtype=dtype).reshape(depth, height, width, channels).copy()
        sdl3.SDL_UnmapGPUTransferBuffer(self.device, transfer_buffer)
        sdl3.SDL_ReleaseGPUTransferBuffer(self.device, transfer_buffer)
        return arr.squeeze(-1) if channels == 1 else arr

    def copy_texture3d(self, src_texture, dst_texture, width, height, depth):
        """Ping-pong helper (e.g. velocity -> velocity_prev between the
        advect and diffuse passes, matching GridFluidSolver.step()'s
        own .copy_from() calls in the Taichi reference implementation)."""
        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        copy_pass = sdl3.SDL_BeginGPUCopyPass(cmd)
        src_loc = sdl3.SDL_GPUTextureLocation(texture=src_texture, mip_level=0, layer=0, x=0, y=0, z=0)
        dst_loc = sdl3.SDL_GPUTextureLocation(texture=dst_texture, mip_level=0, layer=0, x=0, y=0, z=0)
        sdl3.SDL_CopyGPUTextureToTexture(copy_pass, ctypes.byref(src_loc), ctypes.byref(dst_loc),
                                          width, height, depth, False)
        sdl3.SDL_EndGPUCopyPass(copy_pass)
        assert sdl3.SDL_SubmitGPUCommandBuffer(cmd), sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)

    def dispatch_compute(self, pipeline, readonly_textures, readwrite_textures, group_counts,
                          uniform_bytes=None, storage_buffers=None):
        """readonly_textures / readwrite_textures: lists of SDL_GPUTexture
        handles, bound at slots 0..N-1 within their own category (NOT a
        combined index with the other category) -- see this module's
        docstring/memory notes on why the slot<->GLSL-binding mapping
        isn't assumable from the shader source and was verified
        empirically for boundary.comp specifically.

        `storage_buffers`: a list of SDL_GPUBuffer handles, ALL bound
        through the read-write storage-buffer category regardless of
        whether the kernel actually writes each one (see
        create_compute_pipeline's docstring for why) -- order must match
        the pipeline's compiled binding order (relativity_kernel_dsl's compiler
        already resolves this from the real compiled MSL, not assumed
        from GLSL source order)."""
        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        assert cmd, sdl3.SDL_GetError()

        rw_tex_bindings = (sdl3.SDL_GPUStorageTextureReadWriteBinding * max(len(readwrite_textures), 1))()
        for i, tex in enumerate(readwrite_textures):
            rw_tex_bindings[i] = sdl3.SDL_GPUStorageTextureReadWriteBinding(
                texture=tex, mip_level=0, layer=0, cycle=False)

        storage_buffers = storage_buffers or []
        rw_buf_bindings = (sdl3.SDL_GPUStorageBufferReadWriteBinding * max(len(storage_buffers), 1))()
        for i, buf in enumerate(storage_buffers):
            rw_buf_bindings[i] = sdl3.SDL_GPUStorageBufferReadWriteBinding(buffer=buf, cycle=False)

        compute_pass = sdl3.SDL_BeginGPUComputePass(
            cmd,
            rw_tex_bindings if readwrite_textures else None, len(readwrite_textures),
            rw_buf_bindings if storage_buffers else None, len(storage_buffers),
        )
        sdl3.SDL_BindGPUComputePipeline(compute_pass, pipeline)

        if readonly_textures:
            tex_array = (sdl3.LP_SDL_GPUTexture * len(readonly_textures))(*readonly_textures)
            sdl3.SDL_BindGPUComputeStorageTextures(compute_pass, 0, tex_array, len(readonly_textures))

        if uniform_bytes is not None:
            ubuf = (ctypes.c_ubyte * len(uniform_bytes)).from_buffer_copy(uniform_bytes)
            sdl3.SDL_PushGPUComputeUniformData(cmd, 0, ctypes.cast(ubuf, ctypes.c_void_p), len(uniform_bytes))

        sdl3.SDL_DispatchGPUCompute(compute_pass, *group_counts)
        sdl3.SDL_EndGPUComputePass(compute_pass)

        assert sdl3.SDL_SubmitGPUCommandBuffer(cmd), sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)

    def render_frame(self, fragment_uniform_bytes=None, fragment_texture_sampler=None,
                      fragment_storage_buffers=None):
        """Draws the fullscreen triangle through the loaded pipeline,
        optionally pushing `fragment_uniform_bytes` as the fragment
        shader's UBO and binding `fragment_texture_sampler` (a
        (texture, sampler) pair from upload_texture3d()) at slot 0, and
        returns the result as an (H, W, 4) uint8 RGBA numpy array.
        `fragment_storage_buffers`: SDL_GPUBuffer handles the fragment
        shader reads directly (e.g. a BVH-node buffer for a raytrace
        shader), bound in order via SDL_BindGPUFragmentStorageBuffers --
        those buffers must have been created with graphics_readable=True."""
        cmd = sdl3.SDL_AcquireGPUCommandBuffer(self.device)
        assert cmd, sdl3.SDL_GetError()

        color_target_info = sdl3.SDL_GPUColorTargetInfo(
            texture=self.color_texture, mip_level=0, layer_or_depth_plane=0,
            clear_color=sdl3.SDL_FColor(r=0.0, g=0.0, b=0.0, a=1.0),
            load_op=sdl3.SDL_GPU_LOADOP_CLEAR, store_op=sdl3.SDL_GPU_STOREOP_STORE,
        )
        render_pass = sdl3.SDL_BeginGPURenderPass(cmd, ctypes.byref(color_target_info), 1, None)
        sdl3.SDL_BindGPUGraphicsPipeline(render_pass, self.pipeline)

        if fragment_texture_sampler is not None:
            texture, sampler = fragment_texture_sampler
            binding = sdl3.SDL_GPUTextureSamplerBinding(texture=texture, sampler=sampler)
            sdl3.SDL_BindGPUFragmentSamplers(render_pass, 0, ctypes.byref(binding), 1)

        if fragment_storage_buffers:
            buf_array = (sdl3.LP_SDL_GPUBuffer * len(fragment_storage_buffers))(*fragment_storage_buffers)
            sdl3.SDL_BindGPUFragmentStorageBuffers(render_pass, 0, buf_array, len(fragment_storage_buffers))

        if fragment_uniform_bytes is not None:
            ubuf = (ctypes.c_ubyte * len(fragment_uniform_bytes)).from_buffer_copy(fragment_uniform_bytes)
            sdl3.SDL_PushGPUFragmentUniformData(cmd, 0, ctypes.cast(ubuf, ctypes.c_void_p),
                                                 len(fragment_uniform_bytes))

        sdl3.SDL_DrawGPUPrimitives(render_pass, 3, 1, 0, 0)
        sdl3.SDL_EndGPURenderPass(render_pass)

        copy_pass = sdl3.SDL_BeginGPUCopyPass(cmd)
        src_region = sdl3.SDL_GPUTextureRegion(
            texture=self.color_texture, mip_level=0, layer=0,
            x=0, y=0, z=0, w=self.width, h=self.height, d=1,
        )
        dst_info = sdl3.SDL_GPUTextureTransferInfo(
            transfer_buffer=self.transfer_buffer, offset=0,
            pixels_per_row=self.width, rows_per_layer=self.height,
        )
        sdl3.SDL_DownloadFromGPUTexture(copy_pass, ctypes.byref(src_region), ctypes.byref(dst_info))
        sdl3.SDL_EndGPUCopyPass(copy_pass)

        ok = sdl3.SDL_SubmitGPUCommandBuffer(cmd)
        assert ok, sdl3.SDL_GetError()
        sdl3.SDL_WaitForGPUIdle(self.device)

        ptr = sdl3.SDL_MapGPUTransferBuffer(self.device, self.transfer_buffer, False)
        assert ptr, sdl3.SDL_GetError()
        n_bytes = self.width * self.height * 4
        raw = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte * n_bytes)).contents
        pixels = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(self.height, self.width, 4).copy()
        sdl3.SDL_UnmapGPUTransferBuffer(self.device, self.transfer_buffer)
        return pixels
