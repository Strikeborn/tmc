/*
 * port_voxel.cpp — experimental 3D room view (see port_voxel.h).
 *
 * World space: X = room-local pixel x (east), Y = height (up), Z = room-local
 * pixel y (south). 1 unit = 1 GBA pixel. The camera sits south of the game's
 * scroll centre and looks north-down, so the room reads like the 2D game
 * tilted back.
 *
 * Per frame we upload VRAM (96 KB), both room sub-tile maps (64 KB) and the
 * 512-colour palette, then draw a few hundred quads; every pixel is decoded in
 * voxel.frag. No per-frame heap allocation: all buffers are static or
 * created once.
 */

#include "port_voxel.h"

#include "port_gpu_renderer.h"
#include "port_ppu.h"
#include "port_runtime_config.h"

#ifndef TMC_GPU_RENDERER

bool Port_Voxel_Present(SDL_GPUCommandBuffer*, SDL_GPUTexture*, int, int) {
    return false;
}
bool Port_Voxel_IsDrawing(void) {
    return false;
}
void Port_Voxel_Shutdown(void) {
}
PortVoxelTileAhead Port_Voxel_TileAhead(void) {
    return {};
}
void Port_Voxel_SetTileShape(int, int, int) {
}
int Port_Voxel_AreaWallTiles(int) {
    return 2;
}
void Port_Voxel_SetAreaWallTiles(int, int) {
}
int Port_Voxel_CurrentArea(void) {
    return -1;
}
void Port_Voxel_RequestShot(const char*) {
}
void Port_Voxel_HandleEvent(const union SDL_Event*) {
}
void Port_Voxel_RemapDpad(uint16_t*) {
}
int Port_Voxel_ViewTurn(void) {
    return 0;
}

#else

#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>

#include <cpu/mode1.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>

#include <nlohmann/json.hpp>
#include <png.h>

extern "C" {
#define this this_ptr
#include "global.h"
#include "area.h"
#include "game.h"
#include "main.h"
#include "map.h"
#include "player.h"
#include "room.h"
#include "screen.h"
#include "message.h"
#undef this
#include "port_asset_loader.h"
#include "port_debug_menu.h"
#include "port_gba_mem.h"
#include "port_imgui_menu.h"
#include "port_widescreen.h"
extern u16 gMapDataBottomSpecial[0x4000];
extern u16 gMapDataTopSpecial[0x4000];
}

static const unsigned char kVoxelVertSpv[] = {
#include "voxel.vert.spv.h"
};
static const unsigned char kVoxelFragSpv[] = {
#include "voxel.frag.spv.h"
};

namespace {

/* ponytail: fixed distance/FOV; pitch comes from config (F8 stepper). */
constexpr float kDistance = 250.0f;  /* camera distance from the scroll centre */

/* Walls out of the way: room geometry between the camera and an actor
 * (Link, any enemy) thins to a dither around the sight line, so the actor
 * stays in sight from any camera angle -- voxel.frag fadeKeep(). Only what
 * stands above the actor's feet: the floor under it never thins. Layout is
 * voxel.frag's Fade block (std140). */
constexpr int kFadeMaxActors = 16;
constexpr float kFadeKeep = 0.2f;     /* share of a faded surface's pixels kept */
constexpr float kFadeRadius = 20.0f;  /* px round the sight line: a character's width */
constexpr float kFadeFeather = 14.0f; /* px over which it comes back */
constexpr float kFadeAim = 12.0f;     /* sight line aims this far above the feet */
struct PortVoxelFade {
    float cam[4];     /* xyz camera, w 1 = on */
    float fade[4];    /* keep, radius, feather, aim */
    Sint32 count[4];  /* x: actors */
    float actors[kFadeMaxActors][4]; /* xyz feet, world space */
};
constexpr float kFovYDeg = 45.0f;

/* Orbit camera round the scroll centre. Yaw 0 is the GBA's straight-on view;
 * let go, it settles on the nearest 45 deg so the turned D-pad
 * (Port_Voxel_RemapDpad) walks exactly away from / across the view. Pitch
 * starts at the F8 value and follows it when that changes. */
struct OrbitCam {
    float yaw = 0.0f;   /* degrees, eye swings toward +x as it grows */
    float pitch = 0.0f; /* degrees above the ground */
    float dist = kDistance, distGoal = kDistance;
    int cfgPitch = -1;
    bool dragging = false;
    Uint64 lastNs = 0;
};
OrbitCam sCam;
constexpr float kCamMinDist = 80.0f, kCamMaxDist = 900.0f;
constexpr float kCamMinPitch = 10.0f, kCamMaxPitch = 85.0f;
constexpr float kCamTurnRate = 120.0f; /* deg/s, keys and stick */
constexpr float kCamTiltRate = 60.0f;
constexpr float kCamZoomRate = 2.0f; /* distance factor per second */
constexpr float kCamMouseDeg = 0.3f; /* per pixel of right-drag */

float SnappedYaw(void) {
    return std::round(sCam.yaw / 45.0f) * 45.0f;
}

void ResetCam(void) {
    sCam.yaw = 0.0f;
    sCam.pitch = (float)Port_Config_GetVoxelPitch();
    sCam.dist = sCam.distGoal = kDistance;
}

void UpdateCam(void) {
    const Uint64 now = SDL_GetTicksNS();
    const float dt = sCam.lastNs ? std::min((float)(now - sCam.lastNs) * 1e-9f, 0.1f) : 0.0f;
    sCam.lastNs = now;
    if (sCam.cfgPitch != Port_Config_GetVoxelPitch()) {
        sCam.cfgPitch = Port_Config_GetVoxelPitch();
        sCam.pitch = (float)sCam.cfgPitch;
        /* ponytail: debug knob — fixed start angle for scripted shots. */
        if (const char* y = std::getenv("TMC_VOXEL_YAW"))
            sCam.yaw = (float)std::atof(y);
        if (const char* p = std::getenv("TMC_VOXEL_PITCH"))
            sCam.pitch = (float)std::atof(p);
    }
    float turn = 0.0f, tilt = 0.0f, zoom = 0.0f;
    if (!Port_DebugMenu_IsOpen() && !Port_ImGui_WantsTextInput()) {
        const bool* k = SDL_GetKeyboardState(nullptr);
        turn = (float)k[SDL_SCANCODE_L] - (float)k[SDL_SCANCODE_J];
        tilt = (float)k[SDL_SCANCODE_I] - (float)k[SDL_SCANCODE_K];
        zoom = (float)k[SDL_SCANCODE_O] - (float)k[SDL_SCANCODE_U];
    }
    int pads = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&pads)) {
        for (int i = 0; i < pads; ++i)
            if (SDL_Gamepad* g = SDL_GetGamepadFromID(ids[i])) {
                const float rx = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTX) / 32767.0f;
                const float ry = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTY) / 32767.0f;
                if (std::fabs(rx) > 0.25f)
                    turn += rx;
                if (std::fabs(ry) > 0.25f)
                    tilt -= ry;
            }
        SDL_free(ids);
    }
    sCam.yaw += turn * kCamTurnRate * dt;
    if (turn == 0.0f && !sCam.dragging) /* ease onto the nearest 45 deg */
        sCam.yaw += (SnappedYaw() - sCam.yaw) * std::min(1.0f, dt * 8.0f);
    sCam.yaw = std::fmod(sCam.yaw + 540.0f, 360.0f) - 180.0f;
    sCam.pitch = std::clamp(sCam.pitch + tilt * kCamTiltRate * dt, kCamMinPitch, kCamMaxPitch);
    sCam.distGoal = std::clamp(sCam.distGoal * std::pow(kCamZoomRate, zoom * dt), kCamMinDist, kCamMaxDist);
    sCam.dist += (sCam.distGoal - sCam.dist) * std::min(1.0f, dt * 10.0f);
}
constexpr float kTopLayerLift = 16.0f; /* lifted overhead art floats one tile up */
constexpr int kBossSlack = 32; /* art hanging further below the ground row stands on its bottom */

constexpr int kMaxVerts = 6 * 512;
constexpr Uint32 kVramBytes = 0x18000;
constexpr Uint32 kMapBytes = 0x8000; /* one 128x128 u16 sub-tile map */
constexpr Uint32 kPalBytes = 512 * 4;
/* Screen-space layers rendered by the PPU line renderer, stacked in one
 * texture: BG0 (text/HUD, rows 0-159), BG3 (sky/backdrop, rows 160-319) and
 * any BG1/BG2 not bound to a room map (manager-drawn foregrounds such as the
 * Minish paths' giant leaves; rows 320-479). */
constexpr Uint32 kBg0Bytes = MODE1_GBA_WIDTH * 480 * 4;

struct Vert {
    float pos[3];
    float uv[2];
    Uint32 p[4];
};
static_assert(sizeof(Vert) == 36, "Vert layout must match the pipeline");

struct Mat4 {
    float m[16]; /* column-major */
};

Mat4 Mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}

/* Right-handed view looking from eye to target, Y up. */
Mat4 LookAt(const float eye[3], const float at[3]) {
    float f[3] = { at[0] - eye[0], at[1] - eye[1], at[2] - eye[2] };
    float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (float& v : f)
        v /= fl;
    /* s = f x up(0,1,0) */
    float s[3] = { -f[2], 0.0f, f[0] };
    float sl = std::sqrt(s[0] * s[0] + s[2] * s[2]);
    s[0] /= sl;
    s[2] /= sl;
    /* u = s x f */
    float u[3] = { s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0] };
    Mat4 r{};
    r.m[0] = s[0];
    r.m[4] = s[1];
    r.m[8] = s[2];
    r.m[1] = u[0];
    r.m[5] = u[1];
    r.m[9] = u[2];
    r.m[2] = -f[0];
    r.m[6] = -f[1];
    r.m[10] = -f[2];
    r.m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    r.m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    r.m[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
    r.m[15] = 1.0f;
    return r;
}

/* Right-handed perspective, depth 0..1 (SDL_GPU clip space). */
Mat4 Perspective(float fovY, float aspect, float zn, float zf) {
    const float t = 1.0f / std::tan(fovY * 0.5f);
    Mat4 r{};
    r.m[0] = t / aspect;
    r.m[5] = t;
    r.m[10] = zf / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = zn * zf / (zn - zf);
    return r;
}

/* Screen pixels (0..w, 0..h, y down) -> clip space at depth 0 (always in front). */
Mat4 Ortho(float w, float h) {
    Mat4 r{};
    r.m[0] = 2.0f / w;
    r.m[5] = -2.0f / h;
    r.m[12] = -1.0f;
    r.m[13] = 1.0f;
    r.m[15] = 1.0f;
    return r;
}

SDL_GPUDevice* sDev = nullptr;
bool sInitTried = false;
bool sReady = false;
SDL_GPUGraphicsPipeline* sPipeline = nullptr;
/* World sprites: depth-tested against the room but not written, so sprites
 * layer among themselves by OAM order as on the GBA (a boss's parts anchored
 * at different rows still composite like the 2D frame). */
SDL_GPUGraphicsPipeline* sSpritePipeline = nullptr;
SDL_GPUShader* sVs = nullptr;
SDL_GPUShader* sFs = nullptr;
SDL_GPUSampler* sSampler = nullptr;
SDL_GPUTexture* sVramTex = nullptr;
SDL_GPUTexture* sMapTex = nullptr;
SDL_GPUTexture* sPalTex = nullptr;
SDL_GPUTexture* sBg0Tex = nullptr;
SDL_GPUTexture* sMaskTex = nullptr;
SDL_GPUTexture* sDepthTex = nullptr;
SDL_GPUTextureFormat sDepthFmt = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
int sDepthW = 0, sDepthH = 0;
/* Debug shot (repro tour): offscreen colour/depth + readback buffer. */
constexpr int kShotW = 960, kShotH = 540;
bool sShotRequested = false, sShotPending = false;
char sShotPath[512];
SDL_GPUTexture* sShotTex = nullptr;
SDL_GPUTexture* sShotDepth = nullptr;
SDL_GPUTransferBuffer* sShotXfer = nullptr;
SDL_GPUBuffer* sVertBuf = nullptr;
SDL_GPUTransferBuffer* sXfer = nullptr;
SDL_GPUTransferBuffer* sMapXfer = nullptr;

Vert sVerts[kMaxVerts];
/* Room geometry: up to ~4 quads per tile (floor/underlay, top, wall, sides)
 * plus the 24-tile edge margin around a 64x64 room. */
/* tiles and margin, plus room for voxel props (a bush is ~800 faces) */
constexpr int kMaxMapVerts = (64 * 64 * 8 + 112 * 112 * 2) * 6 + 1000000;
Vert sMapVerts[kMaxMapVerts];
int sMapVertCount = 0;
Uint64 sMapKey = 0;
constexpr int kMaxProps = 256;
Uint8 sMaskPixels[256 * 256];
bool sPropOutlined[256]; /* per prop mask slot: from its closed outline (round), not its ground */
int sPropCount = 0;
uint32_t sBg0Pixels[MODE1_GBA_WIDTH * 480];

SDL_GPUTexture* MakeTex(SDL_GPUTextureFormat fmt, Uint32 w, Uint32 h, SDL_GPUTextureUsageFlags usage) {
    SDL_GPUTextureCreateInfo ci = {};
    ci.type = SDL_GPU_TEXTURETYPE_2D;
    ci.format = fmt;
    ci.usage = usage;
    ci.width = w;
    ci.height = h;
    ci.layer_count_or_depth = 1;
    ci.num_levels = 1;
    return SDL_CreateGPUTexture(sDev, &ci);
}

bool Init(void) {
    sInitTried = true;
    sDev = Port_GPU_GetDevice();
    if (!sDev || Port_GPU_GetShaderFormat() != SDL_GPU_SHADERFORMAT_SPIRV) {
        std::fprintf(stderr, "[voxel] needs the SDL_GPU Vulkan/SPIR-V backend; voxel view unavailable\n");
        return false;
    }
    const SDL_GPUTextureUsageFlags samp = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    if (!SDL_GPUTextureSupportsFormat(sDev, SDL_GPU_TEXTUREFORMAT_R8_UINT, SDL_GPU_TEXTURETYPE_2D, samp) ||
        !SDL_GPUTextureSupportsFormat(sDev, SDL_GPU_TEXTUREFORMAT_R16_UINT, SDL_GPU_TEXTURETYPE_2D, samp)) {
        std::fprintf(stderr, "[voxel] device lacks R8_UINT/R16_UINT sampling; voxel view unavailable\n");
        return false;
    }

    SDL_GPUShaderCreateInfo vs = {};
    vs.code = kVoxelVertSpv;
    vs.code_size = sizeof(kVoxelVertSpv);
    vs.entrypoint = "main";
    vs.format = SDL_GPU_SHADERFORMAT_SPIRV;
    vs.stage = SDL_GPU_SHADERSTAGE_VERTEX;
    vs.num_uniform_buffers = 1;
    sVs = SDL_CreateGPUShader(sDev, &vs);
    SDL_GPUShaderCreateInfo fs = {};
    fs.code = kVoxelFragSpv;
    fs.code_size = sizeof(kVoxelFragSpv);
    fs.entrypoint = "main";
    fs.format = SDL_GPU_SHADERFORMAT_SPIRV;
    fs.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    fs.num_samplers = 5;
    fs.num_uniform_buffers = 1; /* PortVoxelFade */
    sFs = SDL_CreateGPUShader(sDev, &fs);
    if (!sVs || !sFs) {
        std::fprintf(stderr, "[voxel] shader load failed: %s\n", SDL_GetError());
        return false;
    }

    SDL_GPUVertexBufferDescription vbd = {};
    vbd.slot = 0;
    vbd.pitch = sizeof(Vert);
    vbd.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
    SDL_GPUVertexAttribute attrs[3] = {};
    attrs[0] = { 0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, (Uint32)offsetof(Vert, pos) };
    attrs[1] = { 1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, (Uint32)offsetof(Vert, uv) };
    attrs[2] = { 2, 0, SDL_GPU_VERTEXELEMENTFORMAT_UINT4, (Uint32)offsetof(Vert, p) };

    SDL_GPUColorTargetDescription ctd = {};
    ctd.format = Port_GPU_GetSwapchainFormat();

    SDL_GPUGraphicsPipelineCreateInfo pci = {};
    pci.vertex_shader = sVs;
    pci.fragment_shader = sFs;
    pci.vertex_input_state.vertex_buffer_descriptions = &vbd;
    pci.vertex_input_state.num_vertex_buffers = 1;
    pci.vertex_input_state.vertex_attributes = attrs;
    pci.vertex_input_state.num_vertex_attributes = 3;
    pci.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    pci.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    pci.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    pci.depth_stencil_state.enable_depth_test = true;
    pci.depth_stencil_state.enable_depth_write = true;
    pci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
    pci.target_info.color_target_descriptions = &ctd;
    pci.target_info.num_color_targets = 1;
    pci.target_info.has_depth_stencil_target = true;
    /* 32-bit float depth when the device has it: stacked layers sit 0.25-1px
     * apart at ~250px camera distance, too fine for 16-bit depth (flicker). */
    sDepthFmt = SDL_GPUTextureSupportsFormat(sDev, SDL_GPU_TEXTUREFORMAT_D32_FLOAT, SDL_GPU_TEXTURETYPE_2D,
                                             SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)
                    ? SDL_GPU_TEXTUREFORMAT_D32_FLOAT
                    : SDL_GPU_TEXTUREFORMAT_D16_UNORM;
    pci.target_info.depth_stencil_format = sDepthFmt;
    sPipeline = SDL_CreateGPUGraphicsPipeline(sDev, &pci);
    pci.depth_stencil_state.enable_depth_write = false;
    sSpritePipeline = SDL_CreateGPUGraphicsPipeline(sDev, &pci);

    SDL_GPUSamplerCreateInfo sci = {};
    sci.min_filter = SDL_GPU_FILTER_NEAREST;
    sci.mag_filter = SDL_GPU_FILTER_NEAREST;
    sci.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sci.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sci.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sci.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sSampler = SDL_CreateGPUSampler(sDev, &sci);

    sVramTex = MakeTex(SDL_GPU_TEXTUREFORMAT_R8_UINT, 256, 384, samp);
    sMapTex = MakeTex(SDL_GPU_TEXTUREFORMAT_R16_UINT, 128, 256, samp);
    sPalTex = MakeTex(SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 512, 1, samp);
    sBg0Tex = MakeTex(SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, MODE1_GBA_WIDTH, 480, samp);
    sMaskTex = MakeTex(SDL_GPU_TEXTUREFORMAT_R8_UNORM, 256, 256, samp);

    SDL_GPUBufferCreateInfo bci = {};
    bci.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    bci.size = (Uint32)(kMaxMapVerts + kMaxVerts) * sizeof(Vert); /* static room geometry, then per-frame quads */
    sVertBuf = SDL_CreateGPUBuffer(sDev, &bci);

    SDL_GPUTransferBufferCreateInfo tci = {};
    tci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    tci.size = kVramBytes + 2 * kMapBytes + kPalBytes + kBg0Bytes + sizeof(sVerts);
    sXfer = SDL_CreateGPUTransferBuffer(sDev, &tci);
    tci.size = (Uint32)(sizeof(sMapVerts) + sizeof(sMaskPixels));
    sMapXfer = SDL_CreateGPUTransferBuffer(sDev, &tci);

    if (!sPipeline || !sSpritePipeline || !sSampler || !sVramTex || !sMapTex || !sPalTex || !sBg0Tex || !sMaskTex ||
        !sVertBuf ||
        !sXfer || !sMapXfer) {
        std::fprintf(stderr, "[voxel] GPU resource creation failed: %s\n", SDL_GetError());
        return false;
    }
    std::fprintf(stderr, "[voxel] ready\n");
    return true;
}

bool EnsureShotTargets(void) {
    if (sShotTex && sShotDepth && sShotXfer)
        return true;
    sShotTex = MakeTex(Port_GPU_GetSwapchainFormat(), kShotW, kShotH, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
    sShotDepth = MakeTex(sDepthFmt, kShotW, kShotH, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
    SDL_GPUTransferBufferCreateInfo tci = {};
    tci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    tci.size = kShotW * kShotH * 4;
    sShotXfer = SDL_CreateGPUTransferBuffer(sDev, &tci);
    return sShotTex && sShotDepth && sShotXfer;
}

/* Writes the downloaded shot once the GPU is done with it. */
void WriteShot(void) {
    SDL_WaitForGPUIdle(sDev);
    sShotPending = false;
    const auto* px = static_cast<const Uint8*>(SDL_MapGPUTransferBuffer(sDev, sShotXfer, false));
    if (!px)
        return;
    const bool bgra = Port_GPU_GetSwapchainFormat() == SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM ||
                      Port_GPU_GetSwapchainFormat() == SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB;
    FILE* fp = std::fopen(sShotPath, "wb");
    png_structp png = fp ? png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr) : nullptr;
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (info && !setjmp(png_jmpbuf(png))) {
        png_init_io(png, fp);
        png_set_IHDR(png, info, kShotW, kShotH, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        static Uint8 row[kShotW * 3];
        for (int y = 0; y < kShotH; ++y) {
            const Uint8* s = px + (size_t)y * kShotW * 4;
            for (int x = 0; x < kShotW; ++x) {
                row[x * 3 + 0] = s[x * 4 + (bgra ? 2 : 0)];
                row[x * 3 + 1] = s[x * 4 + 1];
                row[x * 3 + 2] = s[x * 4 + (bgra ? 0 : 2)];
            }
            png_write_row(png, row);
        }
        png_write_end(png, nullptr);
        std::fprintf(stderr, "[voxel] shot -> %s\n", sShotPath);
    }
    if (png)
        png_destroy_write_struct(&png, info ? &info : nullptr);
    if (fp)
        std::fclose(fp);
    SDL_UnmapGPUTransferBuffer(sDev, sShotXfer);
}

bool EnsureDepth(int w, int h) {
    if (sDepthTex && sDepthW == w && sDepthH == h)
        return true;
    if (sDepthTex)
        SDL_ReleaseGPUTexture(sDev, sDepthTex);
    sDepthTex = MakeTex(sDepthFmt, (Uint32)w, (Uint32)h,
                        SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
    sDepthW = w;
    sDepthH = h;
    return sDepthTex != nullptr;
}

/* BG index (1/2) a room map layer is bound to, -1 if none. */
int LayerBg(const void* s) {
    return s == (const void*)&gScreen.bg1 ? 1 : s == (const void*)&gScreen.bg2 ? 2 : -1;
}
/* The layer's live BGxCNT (what the PPU draws with this frame), not the
 * settings struct's copy. */
u16 LayerCnt(const BgSettings* s) {
    const int bg = LayerBg(s);
    return bg < 0 ? (s ? s->control : 0) : (u16)(gIoMem[8 + bg * 2] | (gIoMem[9 + bg * 2] << 8));
}

/* Percent of the visible screen-block entries of a map layer's BG that equal
 * its room sub-tile map (`map`) at the current scroll; 0 if the BG is off.
 * A room whose bottom map scores low is drawn some other way (map-less rooms,
 * BG-drawn bosses such as Gyorg) and stays 2D. */
int MapShownPct(const MapLayer& layer, const u16* map) {
    const int bg = LayerBg(layer.bgSettings);
    if (bg < 0 || !(gIoMem[1] & (1 << bg)))
        return 0;
    const u16 cnt = LayerCnt(layer.bgSettings);
    const u16* sb = reinterpret_cast<const u16*>(&gVram[((cnt >> 8) & 31) * 0x800]);
    const int size = cnt >> 14;
    const int hofs = gIoMem[0x10 + bg * 4] | (gIoMem[0x11 + bg * 4] << 8);
    const int vofs = gIoMem[0x12 + bg * 4] | (gIoMem[0x13 + bg * 4] << 8);
    /* gRoomControls.scroll is the game's camera; the BG offset registers are
     * latched separately. While the camera moves the two can be a tile apart
     * for a frame, which sank every sample and flashed the 2D view a few times
     * a second. Score the camera's tile and its 8 neighbours, keep the best. */
    int best = 0;
    for (int oy = -8; oy <= 8; oy += 8) {
        for (int ox = -8; ox <= 8; ox += 8) {
            const int rx0 = gRoomControls.scroll_x - gRoomControls.origin_x + ox;
            const int ry0 = gRoomControls.scroll_y - gRoomControls.origin_y + oy;
            int match = 0, total = 0;
            for (int sy = 4; sy < 160; sy += 8) {
                for (int sx = 4; sx < 240; sx += 8) {
                    const int rx = rx0 + sx, ry = ry0 + sy;
                    if (rx < 0 || ry < 0 || rx >= 1024 || ry >= 1024)
                        continue;
                    const int tx = ((sx + hofs) >> 3) & (size & 1 ? 63 : 31);
                    const int ty = ((sy + vofs) >> 3) & (size & 2 ? 63 : 31);
                    const int block = (tx >> 5) + (ty >> 5) * (size & 1 ? 2 : 1);
                    const u16 e = sb[(block * 1024 + (ty & 31) * 32 + (tx & 31)) & 0x3FFF];
                    match += e == map[(ry >> 3) * 128 + (rx >> 3)];
                    ++total;
                }
            }
            if (total)
                best = std::max(best, match * 100 / total);
        }
    }
    return best;
}
bool BottomMapShown(void) {
    return MapShownPct(gMapBottom, gMapDataBottomSpecial) >= 50;
}
/* Top map on screen as the room's overhead layer; when not (BG off, or the
 * BG holds something else), its BG is composited as a screen layer instead. */
bool TopMapShown(void) {
    return gMapTop.bgSettings != nullptr && MapShownPct(gMapTop, gMapDataTopSpecial) >= 50;
}

/* Room gameplay with the bottom map on screen (not a menu, subtask, or a
 * cutscene/boss that repurposes the BGs). The beanstalk climbs are side-view
 * paintings with nothing to stand up, so they stay 2D. */
bool SceneApplicable(void) {
    return gMain.task == TASK_GAME && gMain.state == GAMETASK_MAIN && gMain.substate != GAMEMAIN_SUBTASK &&
           gMapBottom.bgSettings != nullptr && gRoomControls.width != 0 && gRoomControls.width <= 1024 &&
           gRoomControls.height <= 1024 &&
           !(gRoomControls.area == AREA_BEANSTALKS && (gRoomControls.scroll_flags & 1)) && BottomMapShown();
}

/* c: 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right; uv per corner. */
void QuadUv(Vert* out, int cap, int& n, const float c[4][3], const float uv[4][2], Uint32 p0, Uint32 p1, Uint32 p2,
            Uint32 p3) {
    if (n + 6 > cap)
        return;
    static const int kIdx[6] = { 0, 2, 1, 1, 2, 3 };
    for (int k : kIdx) {
        Vert& v = out[n++];
        std::memcpy(v.pos, c[k], sizeof(v.pos));
        v.uv[0] = uv[k][0];
        v.uv[1] = uv[k][1];
        v.p[0] = p0;
        v.p[1] = p1;
        v.p[2] = p2;
        v.p[3] = p3;
    }
}

void Quad(Vert* out, int cap, int& n, const float c[4][3], float u0, float v0, float u1, float v1, Uint32 p0,
          Uint32 p1, Uint32 p2, Uint32 p3) {
    const float uv[4][2] = { { u0, v0 }, { u1, v0 }, { u0, v1 }, { u1, v1 } };
    QuadUv(out, cap, n, c, uv, p0, p1, p2, p3);
}

/* ---- room geometry (phase 2) ----
 * GBA maps store no heights, so they are inferred per tile column from the
 * bottom layer's collision: each run of fully solid tiles (cliff faces, house
 * fronts, tree trunks, bushes) stands its southmost kMaxFaceTiles rows up as
 * a vertical wall on the run's south edge; the rest of the run (roof, cliff
 * top) becomes a flat top at the wall height, slid south to meet the wall.
 * Walkable tiles stay on the floor. Top-layer tiles follow the tile under
 * them (standing on walls, lying on tops), else float kTopLayerLift up.
 * ponytail: column heuristic, no plateau propagation; per-area overrides
 * (below, edited from F8) patch the tiles and wall heights it reads wrong. */
constexpr int kDefaultWallTiles = 2;

/* Per-area overrides (voxel_shapes.json):
 *   { "<area>": { "wall": 3, "tiles": { "<tileType>": "floor" | "block" | "prop" } } }
 * Keyed by area because tile types are per-area tileset. */
struct AreaShapes {
    int wall = kDefaultWallTiles;
    std::map<int, int> tiles; /* tileType -> PORT_VOXEL_SHAPE_* */
};
std::map<int, AreaShapes> sShapes;
bool sShapesLoaded = false;
Uint64 sShapesRev = 0; /* bumps on edit so the room geometry rebuilds */
const char* kShapesPath = "voxel_shapes.json";

void LoadShapes(void) {
    sShapesLoaded = true;
    std::ifstream f(kShapesPath);
    if (!f)
        return;
    try {
        nlohmann::json j;
        f >> j;
        for (auto& [areaKey, a] : j.items()) {
            AreaShapes& s = sShapes[std::stoi(areaKey)];
            s.wall = std::clamp(a.value("wall", kDefaultWallTiles), 1, 4);
            if (a.contains("tiles"))
                for (auto& [typeKey, shape] : a["tiles"].items()) {
                    const std::string v = shape.get<std::string>();
                    s.tiles[std::stoi(typeKey)] = v == "floor"  ? PORT_VOXEL_SHAPE_FLOOR
                                                  : v == "prop" ? PORT_VOXEL_SHAPE_PROP
                                                                : PORT_VOXEL_SHAPE_BLOCK;
                }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[voxel] ignoring malformed %s (%s)\n", kShapesPath, e.what());
        sShapes.clear();
    }
}

void SaveShapes(void) {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [area, s] : sShapes) {
        nlohmann::json a = { { "wall", s.wall }, { "tiles", nlohmann::json::object() } };
        for (const auto& [type, shape] : s.tiles)
            a["tiles"][std::to_string(type)] = shape == PORT_VOXEL_SHAPE_FLOOR  ? "floor"
                                               : shape == PORT_VOXEL_SHAPE_PROP ? "prop"
                                                                                : "block";
        j[std::to_string(area)] = a;
    }
    std::ofstream(kShapesPath) << j.dump(2) << "\n";
    ++sShapesRev;
}

const AreaShapes* CurrentShapes(void) {
    if (!sShapesLoaded)
        LoadShapes();
    const auto it = sShapes.find(gRoomControls.area);
    return it == sShapes.end() ? nullptr : &it->second;
}

int BottomTileType(int t) {
    const u16 idx = gMapBottom.mapData[t];
    return idx < TILESET_SIZE ? gMapBottom.tileTypes[idx] : -1;
}

const AreaShapes* sBuildShapes = nullptr; /* CurrentShapes() for the BuildMap in progress */

int TileOverride(int t) {
    if (!sBuildShapes)
        return PORT_VOXEL_SHAPE_AUTO;
    const auto it = sBuildShapes->tiles.find(BottomTileType(t));
    return it == sBuildShapes->tiles.end() ? PORT_VOXEL_SHAPE_AUTO : it->second;
}

bool SolidTile(int x, int y) {
    const int t = y * 64 + x;
    const int ov = TileOverride(t);
    if (ov != PORT_VOXEL_SHAPE_AUTO)
        return ov != PORT_VOXEL_SHAPE_FLOOR;
    if (gMapBottom.collisionData[t] != 0x0F)
        return false;
    /* Water, shallows and holes block walking but are not walls (a bridge
     * over a river must not become a block Link sinks into). */
    switch (gMapBottom.actTiles[t]) {
        case 0x0F: /* shallow water */
        case 0x10: /* water */
        case 0x11: /* deep water / lilypad */
        case 0x19: /* hole */
        case 0xF0: /* hole */
            return false;
        default:
            return true;
    }
}

/* Sunk surfaces: water sits a little below the ground so shorelines show a
 * lip; holes drop further. Shallows stay level (Link wades, not swims). */
float SinkDepth(int x, int y) {
    switch (gMapBottom.actTiles[y * 64 + x]) {
        case 0x10:
        case 0x11:
            return -3.0f;
        case 0x19:
        case 0xF0:
            return -8.0f;
        default:
            return 0.0f;
    }
}

/* ---- prop cutouts ----
 * A lone solid tile (bush, pot, sign, stump) is usually an outlined object
 * drawn over ground. Flood from the tile border through non-outline pixels:
 * whatever the flood can't reach is the object. It then stands as a per-pixel
 * card instead of a cube. Masks live in a 256x256 atlas of 16x16 slots. */

/* BG palette index of a room sub-tile map's pixel, or -1 if transparent. */
int MapIndex(const u16* map, int x, int y, int px, int py, Uint32 charBase, bool bpp8) {
    const u16 e = map[(y * 2 + (py >> 3)) * 128 + x * 2 + (px >> 3)];
    int sx = px & 7, sy = py & 7;
    if (e & 0x400)
        sx = 7 - sx;
    if (e & 0x800)
        sy = 7 - sy;
    const u32 tile = e & 0x3FF;
    u32 ci, idx;
    if (bpp8) {
        ci = gVram[(charBase + tile * 64 + sy * 8 + sx) % 0x18000];
        idx = ci;
    } else {
        const u8 b = gVram[(charBase + tile * 32 + sy * 4 + (sx >> 1)) % 0x18000];
        ci = (sx & 1) ? (b >> 4) : (b & 15);
        idx = (e >> 12) * 16 + ci;
    }
    return ci == 0 ? -1 : (int)idx;
}

int BottomIndex(int x, int y, int px, int py, Uint32 charBase, bool bpp8) {
    return MapIndex(gMapDataBottomSpecial, x, y, px, py, charBase, bpp8);
}

int TopIndex(int x, int y, int px, int py, Uint32 charBase, bool bpp8) {
    return MapIndex(gMapDataTopSpecial, x, y, px, py, charBase, bpp8);
}

int BottomPixel(int x, int y, int px, int py, Uint32 charBase, bool bpp8) {
    const int i = BottomIndex(x, y, px, py, charBase, bpp8);
    return i < 0 ? -1 : gBgPltt[i];
}

bool Dark555(int c) {
    return std::max(c & 31, std::max((c >> 5) & 31, (c >> 10) & 31)) < 9;
}

/* Builds the outlined-object mask of a 16x16 tile, read through `pix`
 * (px, py) -> RGB555 or -1 for transparent, into atlas slot `slot`; false if
 * the tile doesn't read as an outlined object (no enclosed region, or all). */
template <typename Pix> bool BuildMask(Pix pix, int slot, int minCount = 24) {
    bool outline[256], reached[256] = {};
    for (int i = 0; i < 256; ++i) {
        const int c = pix(i & 15, i >> 4);
        const int r = c < 0 ? 31 : c & 31, g = c < 0 ? 31 : (c >> 5) & 31, b = c < 0 ? 31 : (c >> 10) & 31;
        outline[i] = std::max(r, std::max(g, b)) < 9; /* near-black outline */
    }
    int stack[256], sp = 0;
    for (int i = 0; i < 16; ++i)
        for (int e : { i, 240 + i, i * 16, i * 16 + 15 })
            if (!outline[e] && !reached[e]) {
                reached[e] = true;
                stack[sp++] = e;
            }
    while (sp) {
        const int i = stack[--sp], px = i & 15, py = i >> 4;
        const int nb[4] = { px > 0 ? i - 1 : -1, px < 15 ? i + 1 : -1, py > 0 ? i - 16 : -1, py < 15 ? i + 16 : -1 };
        for (int j : nb)
            if (j >= 0 && !outline[j] && !reached[j]) {
                reached[j] = true;
                stack[sp++] = j;
            }
    }
    int count = 0;
    for (int i = 0; i < 256; ++i)
        count += !reached[i];
    if (count < minCount || count > 240)
        return false;
    const int ox = (slot % 16) * 16, oy = (slot / 16) * 16;
    for (int i = 0; i < 256; ++i)
        sMaskPixels[(oy + (i >> 4)) * 256 + ox + (i & 15)] = reached[i] ? 0 : 255;
    return true;
}

bool BuildPropMask(int x, int y, int slot, Uint32 charBase, bool bpp8) {
    return BuildMask([&](int px, int py) { return BottomPixel(x, y, px, py, charBase, bpp8); }, slot);
}

/* Prop whose outline is open (a sapling's leaves reach the tile edge): the
 * object is every pixel whose colour the ground tile (ux,uy) never uses, so
 * a differently arranged grass background still reads as ground. Fails when
 * that is most of the tile (not an object on that ground) or too little to
 * see. */
bool BuildDiffMask(int x, int y, int ux, int uy, int slot, Uint32 charBase, bool bpp8, int* outCount = nullptr) {
    int ground[256], ng = 0;
    for (int i = 0; i < 256; ++i) {
        const int c = BottomPixel(ux, uy, i & 15, i >> 4, charBase, bpp8);
        if (std::find(ground, ground + ng, c) == ground + ng)
            ground[ng++] = c;
    }
    /* A cast shadow is the ground's own green, darker than any ground
     * pixel: it lies on the ground, it isn't part of the object. */
    auto bright = [](int c) { return std::max(c & 31, std::max((c >> 5) & 31, (c >> 10) & 31)); };
    int groundDark = 31;
    for (int k = 0; k < ng; ++k)
        if (ground[k] >= 0)
            groundDark = std::min(groundDark, bright(ground[k]));
    auto shadow = [&](int c) {
        const int r = c & 31, g = (c >> 5) & 31, b = (c >> 10) & 31;
        return g > r && g > b && bright(c) < groundDark && !Dark555(c);
    };
    bool obj[256];
    int count = 0;
    for (int i = 0; i < 256; ++i) {
        const int c = BottomPixel(x, y, i & 15, i >> 4, charBase, bpp8);
        obj[i] = std::find(ground, ground + ng, c) == ground + ng && !shadow(c);
        count += obj[i];
    }
    if (outCount)
        *outCount = count;
    if (count < 24 || count > 200)
        return false;
    const int ox = (slot % 16) * 16, oy = (slot / 16) * 16;
    for (int i = 0; i < 256; ++i)
        sMaskPixels[(oy + (i >> 4)) * 256 + ox + (i & 15)] = obj[i] ? 255 : 0;
    return true;
}

bool TopTileEmpty(int x, int y) {
    const u16* s = &gMapDataTopSpecial[y * 2 * 128 + x * 2];
    return (s[0] | s[1] | s[128] | s[129]) == 0;
}

Uint64 MapKey(void) {
    Uint64 h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t len) {
        const auto* b = static_cast<const Uint8*>(p);
        for (size_t i = 0; i < len; ++i)
            h = (h ^ b[i]) * 1099511628211ull;
    };
    mix(gMapBottom.collisionData, sizeof(gMapBottom.collisionData));
    mix(gMapDataTopSpecial, sizeof(u16) * 0x4000);
    mix(gMapDataBottomSpecial, sizeof(u16) * 0x4000);
    mix(&gRoomControls.width, sizeof(gRoomControls.width));
    mix(&gRoomControls.height, sizeof(gRoomControls.height));
    const u16 cb = LayerCnt(gMapBottom.bgSettings), ct = LayerCnt(gMapTop.bgSettings);
    const bool topShown = TopMapShown();
    mix(&cb, sizeof(cb));
    mix(&ct, sizeof(ct));
    mix(&topShown, sizeof(topShown));
    mix(gMapBottom.mapData, sizeof(gMapBottom.mapData)); /* tile types drive overrides */
    mix(&gRoomControls.area, sizeof(gRoomControls.area));
    mix(&sShapesRev, sizeof(sShapesRev));
    return h;
}

/* ---- room geometry: inverse of the GBA's oblique projection ----
 * The 2D art draws something of height h standing at ground z at screen
 * row z - h. So a solid run of tiles (collision) stands up as a box: its
 * southmost `face` rows are the front wall; the rows above them are the
 * box's top, which belongs over the ground `topH` further south. What the
 * player sees is the "visible art": the overhead layer where it's opaque
 * (trees, forests, roofs are drawn there; the ground tiles under them are
 * never-shown filler), both layers where it's partial, the ground layer
 * otherwise. Overhead art overhanging walkable tiles next to a solid object
 * (a canopy's top rows) belongs to that object. */
void BuildMap(void) {
    int n = 0;
    sBuildShapes = CurrentShapes();
    const int wallTiles = sBuildShapes ? sBuildShapes->wall : kDefaultWallTiles;
    sPropCount = 0;
    std::memset(sMaskPixels, 0, sizeof(sMaskPixels));
    const int W = gRoomControls.width / 16, H = gRoomControls.height / 16;
    const u16 cb = LayerCnt(gMapBottom.bgSettings);
    const Uint32 bChar = ((cb >> 2) & 3u) * 0x4000u, b8 = (cb & 0x80) ? 1u : 0u;
    const u16 ct = LayerCnt(gMapTop.bgSettings);
    /* Top map drawn BELOW the bottom one (lower BG priority, e.g. the Minish
     * rafters over the house they span): it's a floor far under the walkable
     * layer, seen through its transparent gaps, not overhead art. */
    const bool topBelow = TopMapShown() && (ct & 3) > (cb & 3);
    const bool hasTop = TopMapShown() && !topBelow;
    const Uint32 tChar = ((ct >> 2) & 3u) * 0x4000u, t8 = (ct & 0x80) ? 1u : 0u;
    const bool outdoors = (gArea.areaMetadata & AR_IS_OVERWORLD) != 0;
    auto inRoom = [&](int x, int y) { return x >= 0 && y >= 0 && x < W && y < H; };

    /* ---- per-tile classification ---- */
    static Uint8 cover[64 * 64]; /* overhead layer: 0 none, 1 partial, 2 opaque */
    static Uint8 solid[64 * 64], geom[64 * 64];
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int t = y * 64 + x;
            int op = 0;
            if (hasTop && !TopTileEmpty(x, y))
                for (int i = 0; i < 256; i += 3)
                    op += TopIndex(x, y, i & 15, i >> 4, tChar, t8 != 0) >= 0;
            cover[t] = op >= 70 ? 2 : op > 0 ? 1 : 0;
            solid[t] = SolidTile(x, y);
            geom[t] = solid[t];
        }
    /* Absorb overhang: overhead art on walkable tiles touching a covered solid
     * tile (two passes: canopies overhang up to two rows). */
    for (int pass = 0; pass < 2; ++pass)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int t = y * 64 + x;
                if (geom[t] || !cover[t] || SinkDepth(x, y) != 0.0f)
                    continue;
                static const int kNb[4][2] = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
                for (const auto& d : kNb) {
                    const int nx = x + d[0], ny = y + d[1];
                    if (inRoom(nx, ny) && geom[ny * 64 + nx] && cover[ny * 64 + nx]) {
                        geom[t] = 2; /* 2 = absorbed (visual only) */
                        break;
                    }
                }
            }
    auto Geom = [&](int x, int y) { return inRoom(x, y) && geom[y * 64 + x] != 0; };
    auto Cover = [&](int x, int y) { return inRoom(x, y) ? cover[y * 64 + x] : 0; };

    /* Visible art pixel (RGB555, -1 transparent). */
    auto VisPixel = [&](int x, int y, int px, int py) -> int {
        if (Cover(x, y)) {
            const int i = TopIndex(x, y, px, py, tChar, t8 != 0);
            if (i >= 0)
                return gBgPltt[i];
            if (Cover(x, y) == 2)
                return -1;
        }
        return BottomPixel(x, y, px, py, bChar, b8 != 0);
    };
    /* Foliage: dominant lit colour greener than red (canopies, hedges). */
    auto Foliage = [&](int x, int y) -> bool {
        std::map<int, int> counts;
        int best = -1, bestN = 0;
        for (int i = 0; i < 256; i += 2) {
            const int c = VisPixel(x, y, i & 15, i >> 4);
            if (c >= 0 && !Dark555(c) && ++counts[c] > bestN)
                bestN = counts[c], best = c;
        }
        return best >= 0 && ((best >> 5) & 31) > (best & 31);
    };
    /* Mostly dark art (cast shadows, trunk undersides): >= 40% of pixels. */
    auto DarkTile = [&](int x, int y) -> bool {
        int dark = 0;
        for (int i = 0; i < 256; i += 2) {
            const int c = VisPixel(x, y, i & 15, i >> 4);
            dark += c >= 0 && std::max(c & 31, std::max((c >> 5) & 31, (c >> 10) & 31)) < 12;
        }
        return dark * 5 >= 128 * 2;
    };
    /* Silhouette mask of the visible art, cached per (bottom, top) tile pair
     * so a forest of identical canopy tiles shares a handful of slots. */
    std::map<Uint64, Uint32> maskCache;
    auto silhouette = [&](int x, int y) -> Uint32 {
        const u16* ts = &gMapDataTopSpecial[y * 2 * 128 + x * 2];
        const Uint64 key = ((Uint64)gMapBottom.mapData[y * 64 + x] << 48) | ((Uint64)ts[0] << 32) |
                           ((Uint64)ts[1] << 16) | ts[128] ^ ((Uint64)ts[129] << 8);
        const auto it = maskCache.find(key);
        if (it != maskCache.end())
            return it->second;
        Uint32 m = 0;
        if (sPropCount < kMaxProps &&
            BuildMask([&](int px, int py) { return VisPixel(x, y, px, py); }, sPropCount, 96))
            m = (Uint32)++sPropCount;
        maskCache[key] = m;
        return m;
    };
    /* Overhang pulled into an object (a canopy's edge over walkable ground):
     * only the overhead art's own opaque pixels belong to the object, so the
     * box is cut to them and the real ground shows through the rest, instead
     * of the ground tile under the canopy edge standing up as part of it. */
    std::map<Uint64, Uint32> coverCache;
    auto coverMask = [&](int x, int y) -> Uint32 {
        const u16* ts = &gMapDataTopSpecial[y * 2 * 128 + x * 2];
        const Uint64 key = ((Uint64)ts[0] << 48) | ((Uint64)ts[1] << 32) | ((Uint64)ts[128] << 16) | ts[129];
        const auto it = coverCache.find(key);
        if (it != coverCache.end())
            return it->second;
        Uint32 m = 0;
        if (sPropCount < kMaxProps) {
            const int ox = (sPropCount % 16) * 16, oy = (sPropCount / 16) * 16;
            int count = 0;
            for (int i = 0; i < 256; ++i) {
                const bool on = TopIndex(x, y, i & 15, i >> 4, tChar, t8 != 0) >= 0;
                sMaskPixels[(oy + (i >> 4)) * 256 + ox + (i & 15)] = on ? 255 : 0;
                count += on;
            }
            if (count >= 8)
                m = (Uint32)++sPropCount;
        }
        coverCache[key] = m;
        return m;
    };
    /* The room's commonest walkable ground tile: underlay where no walkable
     * neighbour exists (the middle of a forest). */
    int groundX = -1, groundY = -1;
    {
        std::map<int, int> freq;
        int bestN = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (!Geom(x, y) && Cover(x, y) < 2 && SinkDepth(x, y) == 0.0f) {
                    const int k = gMapBottom.mapData[y * 64 + x];
                    if (++freq[k] > bestN)
                        bestN = freq[k], groundX = x, groundY = y;
                }
    }
    auto groundFor = [&](int x, int y, int& ux, int& uy) -> bool {
        static const int kNb[4][2] = { { 0, 1 }, { -1, 0 }, { 1, 0 }, { 0, -1 } };
        for (const auto& d : kNb) {
            const int nx = x + d[0], ny = y + d[1];
            if (inRoom(nx, ny) && !Geom(nx, ny) && Cover(nx, ny) < 2 && SinkDepth(nx, ny) == 0.0f) {
                ux = nx, uy = ny;
                return true;
            }
        }
        ux = groundX, uy = groundY;
        return groundX >= 0;
    };

    /* ---- quad helpers ---- */
    /* One layer of tile (ux,uy) flat at height h over x-column x, z0..z1. */
    auto flatL = [&](bool top, int x, int ux, int uy, float h, float z0, float z1, Uint32 mask) {
        const float x0 = x * 16.0f, x1 = x0 + 16;
        const float c[4][3] = { { x0, h, z0 }, { x1, h, z0 }, { x0, h, z1 }, { x1, h, z1 } };
        Quad(sMapVerts, kMaxMapVerts, n, c, ux * 16.0f, uy * 16.0f, ux * 16.0f + 16, uy * 16.0f + 16, 0,
             top ? 128u : 0u, top ? tChar : bChar, (top ? t8 : b8) | (mask << 8));
    };
    /* Tile (x,y) flat, composited like the 2D game: ground layer, overhead
     * layer a hair above it (its transparent pixels show the ground art).
     * fill (palette index + 1, 0 = none): box surfaces are closed, so the
     * ground layer paints its see-through texels in the box's material. */
    auto baseParams = [&](Uint32 mask, Uint32 fill) {
        return b8 | (mask << 8) | (fill ? 2u | ((fill - 1) << 20) : 0u);
    };
    auto flatV = [&](int x, int y, float h, float z0, float z1, Uint32 mask, Uint32 fill = 0) {
        const float x0 = x * 16.0f, x1 = x0 + 16;
        const float c[4][3] = { { x0, h, z0 }, { x1, h, z0 }, { x0, h, z1 }, { x1, h, z1 } };
        Quad(sMapVerts, kMaxMapVerts, n, c, x0, y * 16.0f, x1, y * 16.0f + 16, 0, 0u, bChar, baseParams(mask, fill));
        if (Cover(x, y))
            flatL(true, x, x, y, h + 0.3f, z0, z1, mask);
    };
    /* Tile (x,y) stood up on plane z = zFace, h0..h0+hh, composited the same. */
    auto wallV = [&](int x, int y, float zFace, float h0, float hh, Uint32 mask, Uint32 fill = 0) {
        const float x0 = x * 16.0f, x1 = x0 + 16;
        const float c[4][3] = { { x0, h0 + hh, zFace }, { x1, h0 + hh, zFace }, { x0, h0, zFace }, { x1, h0, zFace } };
        Quad(sMapVerts, kMaxMapVerts, n, c, x0, y * 16.0f, x1, y * 16.0f + 16, 0, 0u, bChar, baseParams(mask, fill));
        if (Cover(x, y)) {
            float c2[4][3];
            std::memcpy(c2, c, sizeof(c2));
            for (auto& v : c2)
                v[2] += 0.3f;
            Quad(sMapVerts, kMaxMapVerts, n, c2, x0, y * 16.0f, x1, y * 16.0f + 16, 0, 128u, tChar,
                 t8 | (mask << 8));
        }
    };
    /* Box side on plane x = xs, heights 0..h over z0..z1, wearing tile (x,y)'s
     * visible art with its width along the depth; see-through texels take
     * `fill` so the box is always closed. West sides mirror (u reversed). */
    /* Side wall at x's west or east edge (or at xAt), wearing tile (sx, sy)'s
     * art: by default (x, y) itself. */
    auto sideV = [&](int x, int y, bool east, float z0, float z1, float h, Uint32 fill, Uint32 mask,
                     float xAt = -1.0f, int sx = -1, int sy = -1) {
        if (sx < 0)
            sx = x, sy = y;
        const float xs = xAt >= 0.0f ? xAt : east ? x * 16.0f + 16 : x * 16.0f;
        const float c[4][3] = { { xs, h, z0 }, { xs, h, z1 }, { xs, 0, z0 }, { xs, 0, z1 } };
        const float u0 = sx * 16.0f, v0 = sy * 16.0f;
        const bool top = Cover(sx, sy) != 0;
        Quad(sMapVerts, kMaxMapVerts, n, c, east ? u0 : u0 + 16, v0, east ? u0 + 16 : u0, v0 + 16, 0,
             top ? 128u : 0u, top ? tChar : bChar, (top ? t8 : b8) | 2u | (fill << 20) | (mask << 8));
    };
    /* A prop (sapling, sign, pot) as a voxel extrusion of its art round the
     * tile's centre: each run of art rows with the same opaque extent becomes
     * a slab that wide and at most 5 px deep, front and back wearing those
     * rows and the sides their edge column. Solid from any angle at a handful
     * of quads. slot: the prop's mask (1-based). */
    auto pillar = [&](int x, int y, Uint32 slot) {
        const int ox = (int)((slot - 1) % 16) * 16, oy = (int)((slot - 1) / 16) * 16;
        int extL[16], extR[16];
        for (int r = 0; r < 16; ++r) {
            extL[r] = 16, extR[r] = -1;
            for (int c = 0; c < 16; ++c)
                if (sMaskPixels[(oy + r) * 256 + ox + c])
                    extL[r] = std::min(extL[r], c), extR[r] = c;
        }
        const float cx = x * 16.0f + 8.0f, cz = y * 16.0f + 8.0f;
        const Uint32 params = baseParams(slot, 0);
        /* An object its outline closes in (a bush, a pot) is round and keeps
         * its full depth; one cut from its ground (a sapling) is thin. */
        const bool round = sPropOutlined[slot - 1];
        for (int r0 = 0; r0 < 16;) {
            int r1 = r0 + 1;
            while (r1 < 16 && extL[r1] == extL[r0] && extR[r1] == extR[r0])
                ++r1;
            if (extR[r0] >= 0) {
                /* rows r0..r1-1, top of the tile = 16 px up */
                /* extruded, not square: the art's width across, a few px deep */
                const float hw = (extR[r0] - extL[r0] + 1) * 0.5f, hd = round ? hw : std::min(hw, 2.5f);
                const float hTop = 16.0f - r0, hBot = 16.0f - r1;
                const float u0 = x * 16.0f + extL[r0], u1 = x * 16.0f + extR[r0] + 1;
                const float v0 = y * 16.0f + r0, v1 = y * 16.0f + r1;
                const float xa = cx - hw, xb = cx + hw, za = cz - hd, zb = cz + hd;
                const float front[4][3] = { { xa, hTop, zb }, { xb, hTop, zb }, { xa, hBot, zb }, { xb, hBot, zb } };
                const float back[4][3] = { { xb, hTop, za }, { xa, hTop, za }, { xb, hBot, za }, { xa, hBot, za } };
                const float west[4][3] = { { xa, hTop, za }, { xa, hTop, zb }, { xa, hBot, za }, { xa, hBot, zb } };
                const float east[4][3] = { { xb, hTop, zb }, { xb, hTop, za }, { xb, hBot, zb }, { xb, hBot, za } };
                Quad(sMapVerts, kMaxMapVerts, n, front, u0, v0, u1, v1, 0, 0u, bChar, params);
                Quad(sMapVerts, kMaxMapVerts, n, back, u1, v0, u0, v1, 0, 0u, bChar, params);
                /* the sides wear the art's edge column, as an extrusion would */
                Quad(sMapVerts, kMaxMapVerts, n, west, u0 + 0.25f, v0, u0 + 0.75f, v1, 0, 0u, bChar, params);
                Quad(sMapVerts, kMaxMapVerts, n, east, u1 - 0.75f, v0, u1 - 0.25f, v1, 0, 0u, bChar, params);
                /* the slab's top, where it sticks out past the one above */
                if (r0 == 0 || extR[r0 - 1] < 0 || extL[r0 - 1] > extL[r0] || extR[r0 - 1] < extR[r0]) {
                    const float top[4][3] = { { xa, hTop, za }, { xb, hTop, za }, { xa, hTop, zb }, { xb, hTop, zb } };
                    Quad(sMapVerts, kMaxMapVerts, n, top, u0, v0, u1, v0 + 1, 0, 0u, bChar, params);
                }
            }
            r0 = r1;
        }
    };
    /* A prop as a real voxel model: each art row is spun round the tile's
     * centre into a disc as wide as the row (a trunk becomes a post, leaves a
     * ball), in 2-px voxels. Every voxel wears the art pixel at the
     * same distance out from the centre on its side, so the picture reads the
     * same from any angle. Only faces with no neighbour are drawn. */
    auto voxelProp = [&](int x, int y, Uint32 slot) {
        const int ox = (int)((slot - 1) % 16) * 16, oy = (int)((slot - 1) / 16) * 16;
        auto on = [&](int c, int r) {
            return c >= 0 && c < 16 && r >= 0 && r < 16 && sMaskPixels[(oy + r) * 256 + ox + c] != 0;
        };
        /* 2-px voxels (8 per side): a voxel covers art rows 2k, 2k+1 */
        constexpr int V = 2, G = 16 / V;
        float radius[G];
        for (int k = 0; k < G; ++k) {
            radius[k] = -1.0f;
            for (int r = k * V; r < k * V + V; ++r)
                for (int c = 0; c < 16; ++c)
                    if (on(c, r))
                        radius[k] = std::max(radius[k], std::fabs(c + 0.5f - 8.0f));
        }
        /* Art column at distance d out on this voxel's side. */
        auto column = [&](int vx, int vz, int& side) {
            const float d = std::hypot((vx + 0.5f) * V - 8.0f, (vz + 0.5f) * V - 8.0f);
            side = (vx + 0.5f) * V >= 8.0f ? 1 : -1;
            return std::clamp((int)std::floor(8.0f + side * d), 0, 15);
        };
        /* A voxel exists where the art has a pixel at that distance from the
         * centre on that side: a trunk stays a post, sparse branches become
         * rings and spokes, not solid plates. Outlined (round) props fill. */
        const bool fillRows = sPropOutlined[slot - 1];
        static bool occ[G][G][G]; /* [row][vx][vz] */
        for (int k = 0; k < G; ++k)
            for (int vx = 0; vx < G; ++vx)
                for (int vz = 0; vz < G; ++vz) {
                    int side;
                    const int c = column(vx, vz, side);
                    const float d = std::hypot((vx + 0.5f) * V - 8.0f, (vz + 0.5f) * V - 8.0f);
                    bool art = false;
                    for (int r = k * V; r < k * V + V && !art; ++r)
                        for (int dc = 0; dc < V && !art; ++dc)
                            art = on(c - side * dc, r);
                    occ[k][vx][vz] = radius[k] >= 0.0f && d <= radius[k] + V * 0.5f && (fillRows || art);
                }
        auto at = [&](int k, int vx, int vz) {
            return k >= 0 && k < G && vx >= 0 && vx < G && vz >= 0 && vz < G && occ[k][vx][vz];
        };
        const Uint32 params = baseParams(0, 0);
        const float x0 = x * 16.0f, z0 = y * 16.0f;
        for (int k = 0; k < G; ++k)
            for (int vx = 0; vx < G; ++vx)
                for (int vz = 0; vz < G; ++vz) {
                    if (!occ[k][vx][vz])
                        continue;
                    /* colour: the art pixel this far out on this side (upper row
                     * first), falling back toward the centre to one the art has */
                    int side;
                    int c = column(vx, vz, side);
                    int r = k * V;
                    if (!on(c, r) && on(c, r + 1))
                        ++r;
                    while (!on(c, r) && c != (side > 0 ? 7 : 8))
                        c -= side;
                    const float u = x0 + c + 0.5f, v = z0 + r + 0.5f;
                    const float px = x0 + vx * V, pz = z0 + vz * V, top = 16.0f - k * V, bot = top - V;
                    const float px1 = px + V, pz1 = pz + V;
                    auto face = [&](const float (&q)[4][3]) {
                        Quad(sMapVerts, kMaxMapVerts, n, q, u, v, u, v, 0, 0u, bChar, params);
                    };
                    if (!at(k - 1, vx, vz)) {
                        const float q[4][3] = { { px, top, pz }, { px1, top, pz }, { px, top, pz1 }, { px1, top, pz1 } };
                        face(q);
                    }
                    if (!at(k + 1, vx, vz) && k < G - 1) {
                        const float q[4][3] = { { px, bot, pz1 }, { px1, bot, pz1 }, { px, bot, pz }, { px1, bot, pz } };
                        face(q);
                    }
                    if (!at(k, vx, vz + 1)) {
                        const float q[4][3] = { { px, top, pz1 }, { px1, top, pz1 }, { px, bot, pz1 }, { px1, bot, pz1 } };
                        face(q);
                    }
                    if (!at(k, vx, vz - 1)) {
                        const float q[4][3] = { { px1, top, pz }, { px, top, pz }, { px1, bot, pz }, { px, bot, pz } };
                        face(q);
                    }
                    if (!at(k, vx - 1, vz)) {
                        const float q[4][3] = { { px, top, pz }, { px, top, pz1 }, { px, bot, pz }, { px, bot, pz1 } };
                        face(q);
                    }
                    if (!at(k, vx + 1, vz)) {
                        const float q[4][3] = { { px1, top, pz1 }, { px1, top, pz }, { px1, bot, pz1 }, { px1, bot, pz } };
                        face(q);
                    }
                }
    };
    /* Vertical lip of a sunk tile along one edge, heights d..0. */
    auto lip = [&](int x, int y, int edge, float d) {
        const float x0 = x * 16.0f, x1 = x0 + 16, z0 = y * 16.0f, z1 = z0 + 16;
        float c[4][3];
        float u0, v0, u1, v1;
        if (edge < 2) { /* 0 north, 1 south: runs along x */
            const float z = edge == 0 ? z0 : z1, v = edge == 0 ? z0 + 0.5f : z1 - 0.5f;
            const float d2[4][3] = { { x0, 0, z }, { x1, 0, z }, { x0, d, z }, { x1, d, z } };
            std::memcpy(c, d2, sizeof(c));
            u0 = x0, u1 = x1, v0 = v1 = v;
        } else { /* 2 west, 3 east: runs along z */
            const float xs = edge == 2 ? x0 : x1, u = edge == 2 ? x0 + 0.5f : x1 - 0.5f;
            const float d2[4][3] = { { xs, 0, z0 }, { xs, 0, z1 }, { xs, d, z0 }, { xs, d, z1 } };
            std::memcpy(c, d2, sizeof(c));
            u0 = u1 = u, v0 = z0, v1 = z1;
        }
        Quad(sMapVerts, kMaxMapVerts, n, c, u0, v0, u1, v1, 0, 0u, bChar, b8);
    };
    /* Ground under (x,y), from a walkable neighbour or the room's ground. */
    auto underlay = [&](int x, int y) {
        int ux, uy;
        if (groundFor(x, y, ux, uy))
            flatL(false, x, ux, uy, 0.0f, y * 16.0f, y * 16.0f + 16, 0);
    };
    /* A lone solid tile read as an outlined prop (bush, pot, sign, stump):
     * stands as a per-pixel card mid-tile over borrowed ground. On success the
     * mask lands in slot sPropCount - 1 (params carry sPropCount). */
    /* A solid neighbour of the same tile type continues this object (a fence,
     * a wall); one of a different type is a separate object it just touches,
     * like a sapling planted against a tree's trunk. */
    auto joins = [&](int x, int y, int nx, int ny) {
        return Geom(nx, ny) && BottomTileType(ny * 64 + nx) == BottomTileType(y * 64 + x);
    };
    auto isProp = [&](int x, int y) -> bool {
        const int ov = TileOverride(y * 64 + x);
        if (ov == PORT_VOXEL_SHAPE_FLOOR || ov == PORT_VOXEL_SHAPE_BLOCK || sPropCount >= kMaxProps ||
            Cover(x, y) == 2)
            return false;
        const bool alone = !Geom(x, y - 1) && !Geom(x, y + 1) && !Geom(x - 1, y) && !Geom(x + 1, y);
        if (ov != PORT_VOXEL_SHAPE_PROP && !alone) {
            /* Touching other solids of its own type, a bush in a bush patch
             * is still its own object: leafy art fully outlined inside the
             * tile. (Ledges, fences and doors outline too, so foliage only.) */
            if (joins(x, y, x, y - 1) || joins(x, y, x, y + 1) || joins(x, y, x - 1, y) || joins(x, y, x + 1, y)) {
                if (Cover(x, y) || !outdoors || !Foliage(x, y) || !BuildPropMask(x, y, sPropCount, bChar, b8 != 0))
                    return false;
                sPropOutlined[sPropCount] = true;
                ++sPropCount;
                return true;
            }
            /* Touching another kind of solid (a sapling against a trunk):
             * the tests below decide. */
        } else if (ov != PORT_VOXEL_SHAPE_PROP && Cover(x, y)) {
            return false;
        }
        sPropOutlined[sPropCount] = BuildPropMask(x, y, sPropCount, bChar, b8 != 0);
        if (!sPropOutlined[sPropCount]) {
            /* Open outline: the object is what differs from the ground it was
             * painted over. Grass comes in variants, so try every walkable
             * neighbour and the room's commonest ground. */
            static const int kNb[4][2] = { { 0, 1 }, { -1, 0 }, { 1, 0 }, { 0, -1 } };
            int cand[5][2], nc = 0;
            for (const auto& d : kNb) {
                const int nx = x + d[0], ny = y + d[1];
                if (inRoom(nx, ny) && !Geom(nx, ny) && Cover(nx, ny) == 0 && SinkDepth(nx, ny) == 0.0f)
                    cand[nc][0] = nx, cand[nc][1] = ny, ++nc;
            }
            if (groundX >= 0)
                cand[nc][0] = groundX, cand[nc][1] = groundY, ++nc;
            /* the ground that leaves the smallest object is the one it stands on */
            int best = -1, bestCount = 1 << 30;
            for (int c = 0; c < nc; ++c) {
                int count = 1 << 30;
                if (BuildDiffMask(x, y, cand[c][0], cand[c][1], sPropCount, bChar, b8 != 0, &count) &&
                    count < bestCount)
                    bestCount = count, best = c;
            }
            if (best < 0 || !BuildDiffMask(x, y, cand[best][0], cand[best][1], sPropCount, bChar, b8 != 0))
                return false;
        }
        ++sPropCount;
        return true;
    };

    /* ---- pass 1: classify columns into floor / prop / trunk / box runs and
     * record every box footprint's height per tile cell (hmap), so pass 2
     * can close a box's side wherever its neighbour is lower there. ---- */
    struct Run {
        int x, yt, yb, face, foot; /* foot: first footprint row (tile units) */
        float faceH, topH;
        bool ledge;
    };
    std::vector<Run> runs;
    /* Trunk rows: mostly dark art under foliage. A tree's middle trunk tile
     * is lighter (bark, not shadow), so a row flanked by trunk on both sides
     * under foliage counts too, or that one column stands out as a slab. */
    static Uint8 trunk[64 * 64];
    std::memset(trunk, 0, sizeof(trunk));
    if (outdoors) {
        for (int y = 1; y < H; ++y)
            for (int x = 0; x < W; ++x)
                trunk[y * 64 + x] = Geom(x, y) && Geom(x, y - 1) && DarkTile(x, y) && Foliage(x, y - 1);
        for (int y = 1; y < H; ++y)
            for (int x = 1; x + 1 < W; ++x) {
                const int t = y * 64 + x;
                if (!trunk[t] && trunk[t - 1] && trunk[t + 1] && Geom(x, y) && Geom(x, y - 1) && Foliage(x, y - 1))
                    trunk[t] = 2; /* 2 = flanked */
            }
    }
    static Uint8 kind[64 * 64]; /* 0 floor, 1 prop, 2 box, 3 trunk (flat) */
    static Uint32 propSlot[64 * 64];
    static float hmap[64 * 64];
    std::memset(kind, 0, sizeof(kind));
    std::memset(hmap, 0, sizeof(hmap));
    for (int x = 0; x < W; ++x) {
        int y = H - 1;
        while (y >= 0) {
            if (!Geom(x, y) || kind[y * 64 + x] == 1) { /* 1: a prop found on top of a run */
                --y;
                continue;
            }
            if (isProp(x, y)) {
                kind[y * 64 + x] = 1;
                propSlot[y * 64 + x] = (Uint32)sPropCount;
                --y;
                continue;
            }
            int yb = y;
            --y;
            /* Up the column while solid. A tile of another type standing on
             * top of the run (a sapling planted against a tree's trunk) may be
             * its own prop: the run ends below it (the scan skips it next).
             * Only on top of trees: walls change type row to row too. */
            while (y >= 0 && Geom(x, y)) {
                if (BottomTileType(y * 64 + x) != BottomTileType((y + 1) * 64 + x) && outdoors &&
                    (Cover(x, y + 1) || Foliage(x, y + 1)) && isProp(x, y)) {
                    kind[y * 64 + x] = 1;
                    propSlot[y * 64 + x] = (Uint32)sPropCount;
                    break;
                }
                --y;
            }
            const int yt = y + 1;
            /* Big trees' bottom rows are cast shadow and trunk: mostly dark
             * art under the canopy. They lie on the ground, the canopy box
             * stands above them. */
            while (outdoors && yb > yt && trunk[yb * 64 + x]) {
                kind[yb * 64 + x] = 3;
                --yb;
            }
            Run rn;
            rn.x = x, rn.yt = yt, rn.yb = yb;
            const int len = yb - yt + 1;
            /* A one-row run continuing sideways is a ledge / low fence line. */
            rn.ledge = len == 1 && (Geom(x - 1, yb) || Geom(x + 1, yb)) && !Cover(x, yb);
            rn.face = std::min(len, wallTiles);
            rn.faceH = rn.ledge ? 8.0f : 16.0f;
            rn.topH = rn.face * rn.faceH;
            /* Footprint: the top rows shifted south by the wall height (whole
             * tiles for full walls). A box on the room's north edge keeps its
             * top back to the edge so it meets the margin beyond. */
            rn.foot = len > rn.face ? (yt == 0 ? 0 : yt + rn.face) : yb;
            for (int r = yt; r <= yb; ++r)
                kind[r * 64 + x] = 2;
            for (int b = rn.foot; b <= yb; ++b)
                hmap[b * 64 + x] = rn.topH;
            runs.push_back(rn);
        }
    }
    /* ponytail: debug knob — TMC_VOXEL_DUMPMAP=<file>: per-tile classification
     * of each rebuilt room (kind, cover, bottom tile type), for tuning. */
    if (const char* dumpPath = std::getenv("TMC_VOXEL_DUMPMAP")) {
        if (FILE* f = std::fopen(dumpPath, "a")) {
            std::fprintf(f, "room area=0x%02x room=0x%02x %dx%d (kind: .floor P prop B box T trunk; "
                            "cover: space none, - partial, # opaque; type hex)\n",
                         gRoomControls.area, gRoomControls.room, W, H);
            for (int y = 0; y < H; ++y) {
                std::fprintf(f, "%2d ", y);
                for (int x = 0; x < W; ++x) {
                    const int t = y * 64 + x;
                    std::fprintf(f, "%c%c%03x ", ".PBT"[kind[t]], " -#"[cover[t]], BottomTileType(t) & 0xFFF);
                }
                std::fputc('\n', f);
            }
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    if (kind[y * 64 + x] == 1) {
                        const Uint32 slot = propSlot[y * 64 + x] - 1;
                        const int ox = (int)(slot % 16) * 16, oy = (int)(slot / 16) * 16;
                        std::fprintf(f, "prop %d,%d (%s)\n", x, y, sPropOutlined[slot] ? "outline" : "ground colours");
                        for (int r = 0; r < 16; ++r) {
                            for (int c = 0; c < 16; ++c)
                                std::fputc(sMaskPixels[(oy + r) * 256 + ox + c] ? '#' : '.', f);
                            std::fputc('\n', f);
                        }
                    }
            std::fclose(f);
        }
    }
    auto hAt = [&](int x, int b) { return inRoom(x, b) ? hmap[b * 64 + x] : 0.0f; };

    /* ---- pass 2: draw ---- */
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int t = y * 64 + x;
            if (kind[t] == 1) {
                underlay(x, y);
                voxelProp(x, y, propSlot[t]);
            } else if (kind[t] == 3) {
                underlay(x, y);
                flatV(x, y, 0.2f, y * 16.0f, y * 16.0f + 16, 0);
            } else if (kind[t] == 0) {
                /* Floor. Water/pits sink with lips. Overhead art here either
                 * paints the ground (see-through ground art: paths, grass
                 * edges) or floats above it (arches), placed by projection. */
                const float d = SinkDepth(x, y);
                flatL(false, x, x, y, d, y * 16.0f, y * 16.0f + 16, 0);
                if (d < 0.0f) {
                    static const int kEdge[4][2] = { { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 } };
                    for (int e = 0; e < 4; ++e) {
                        const int nx = x + kEdge[e][0], ny = y + kEdge[e][1];
                        if (!inRoom(nx, ny) || Geom(nx, ny) || SinkDepth(nx, ny) > d)
                            lip(x, y, e, d);
                    }
                }
                if (Cover(x, y)) {
                    int clear = 0;
                    for (int i = 0; i < 256; i += 3)
                        clear += BottomPixel(x, y, i & 15, i >> 4, bChar, b8 != 0) < 0;
                    if (clear >= 4)
                        flatL(true, x, x, y, d + 0.25f, y * 16.0f, y * 16.0f + 16, 0);
                    else
                        flatL(true, x, x, y, kTopLayerLift, y * 16.0f + kTopLayerLift,
                              y * 16.0f + 16 + kTopLayerLift, 0);
                }
            }
        }
    for (const Run& rn : runs) {
        const int x = rn.x, yt = rn.yt, yb = rn.yb, face = rn.face;
        const float topH = rn.topH, zFace = (yb + 1) * 16.0f;
        const bool thin = yb - yt + 1 <= face; /* all face: 8px-deep cap */

        Uint32 rowMask[64] = {};
        bool anyMask = false;
        if (outdoors && !rn.ledge)
            for (int r = yt; r <= yb; ++r)
                anyMask |= (rowMask[r] = geom[r * 64 + x] == 2 ? coverMask(x, r)
                                         : Foliage(x, r)       ? silhouette(x, r)
                                                               : 0u) != 0;
        /* Ground where it can be seen: behind the footprint, and wherever
         * silhouettes are cut. */
        for (int r = yt; r <= yb; ++r) {
            if (!anyMask && r >= rn.foot)
                continue;
            if (r < rn.foot) {
                /* Behind the footprint the 2D art has no ground (the object
                 * covered it): continue the ground north of the object. */
                int uy = yt - 1;
                while (uy >= 0 && Geom(x, uy))
                    --uy;
                if (uy >= 0 && Cover(x, uy) < 2 && SinkDepth(x, uy) == 0.0f) {
                    flatL(false, x, x, uy, 0.0f, r * 16.0f, r * 16.0f + 16, 0);
                    continue;
                }
            }
            underlay(x, r);
        }

        /* The run's dominant visible material: fill for see-through texels
         * on every box surface (top, front, sides). */
        std::map<int, int> counts;
        int fill = 0, fillN = 0;
        for (int r = yt; r <= yb; ++r)
            for (int i = 0; i < 256; i += 4) {
                int ci = Cover(x, r) ? TopIndex(x, r, i & 15, i >> 4, tChar, t8 != 0) : -1;
                if (ci < 0)
                    ci = BottomIndex(x, r, i & 15, i >> 4, bChar, b8 != 0);
                if (ci >= 0 && !Dark555(gBgPltt[ci]) && ++counts[ci] > fillN)
                    fillN = counts[ci], fill = ci;
            }
        const Uint32 fillP = fillN ? (Uint32)fill + 1 : 0;

        /* Front wall: the southmost `face` rows stood up. */
        for (int i = 0; i < face; ++i)
            wallV(x, yb - i, zFace, i * rn.faceH, rn.faceH, rowMask[yb - i], fillP);
        /* Top: footprint cell b shows art row b - face (1:1); cells behind
         * the art (north-edge extension) repeat the first row. */
        auto artRow = [&](int b) { return thin ? yt : std::max(yt, b - face); };
        if (thin) {
            flatV(x, yt, topH, zFace - 8.0f, zFace, rowMask[yt], fillP);
        } else {
            for (int b = rn.foot; b <= yb; ++b)
                flatV(x, artRow(b), topH, b * 16.0f, b * 16.0f + 16, rowMask[artRow(b)], fillP);
        }
        /* Sides wherever the neighbour column stands lower in that cell. */
        for (int b = thin ? yb : rn.foot; b <= yb; ++b) {
            const float z0 = thin ? zFace - 8.0f : b * 16.0f, z1 = thin ? zFace : b * 16.0f + 16;
            const int r = artRow(b);
            /* A canopy's side wears leaves from inside the same tree, not a
             * stretched copy of its outlined edge tile. */
            auto leafFrom = [&](int dir, int& sx) {
                sx = -1;
                if (!outdoors || !Foliage(x, r))
                    return;
                for (int k = 1; k <= 3; ++k)
                    if (Cover(x + dir * k, r) == 2 && Foliage(x + dir * k, r)) {
                        sx = x + dir * k;
                        return;
                    }
            };
            int sx;
            if (hAt(x - 1, b) < topH && x > 0) {
                leafFrom(1, sx);
                sideV(x, r, false, z0, z1, topH, (Uint32)fill, rowMask[r], -1.0f, sx, sx < 0 ? -1 : r);
            }
            if (hAt(x + 1, b) < topH && x < W - 1) {
                leafFrom(-1, sx);
                sideV(x, r, true, z0, z1, topH, (Uint32)fill, rowMask[r], -1.0f, sx, sx < 0 ? -1 : r);
            }
        }
    }
    if (topBelow) {
        constexpr float kBelow = 64.0f;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (!TopTileEmpty(x, y))
                    flatL(true, x, x, y, -kBelow, y * 16.0f - kBelow, y * 16.0f + 16 - kBelow, 0);
    }
    /* Outdoors, the perspective camera sees past the room edge the 2D camera
     * clamps to: continue the room outward with its edge tiles' visible art,
     * at the height they stand at. Interiors and dungeons keep the void. */
    constexpr int kMargin = 24;
    const int margin = outdoors ? kMargin : 0;
    for (int y = -margin; y < H + margin; ++y)
        for (int x = -margin; x < W + margin; ++x) {
            if (inRoom(x, y))
                continue;
            const int ex = std::clamp(x, 0, W - 1), ey = std::clamp(y, 0, H - 1);
            const float h = hAt(ex, ey); /* meet the edge cell exactly: no open step */
            flatL(false, x, ex, ey, h, y * 16.0f, y * 16.0f + 16, 0);
            if (Cover(ex, ey))
                flatL(true, x, ex, ey, h + 0.3f, y * 16.0f, y * 16.0f + 16, 0);
            /* Step to the next margin cell east: close it with a face wearing
             * the higher cell's art (see-through texels in its material). */
            const int nx = x + 1;
            if (nx < W + margin && !inRoom(nx, y)) {
                const int ex2 = std::clamp(nx, 0, W - 1);
                const float h2 = hAt(ex2, ey);
                if (h2 != h) {
                    const int sx = h > h2 ? ex : ex2;
                    const float lo = std::min(h, h2), hi = std::max(h, h2), xs = nx * 16.0f;
                    const bool topArt = Cover(sx, ey) != 0;
                    int dom = 0, domN = 0;
                    std::map<int, int> cnts;
                    for (int i = 0; i < 256; i += 4) {
                        const int ci = topArt ? TopIndex(sx, ey, i & 15, i >> 4, tChar, t8 != 0)
                                              : BottomIndex(sx, ey, i & 15, i >> 4, bChar, b8 != 0);
                        if (ci >= 0 && !Dark555(gBgPltt[ci]) && ++cnts[ci] > domN)
                            domN = cnts[ci], dom = ci;
                    }
                    const float c[4][3] = { { xs, hi, y * 16.0f }, { xs, hi, y * 16.0f + 16 }, { xs, lo, y * 16.0f },
                                            { xs, lo, y * 16.0f + 16 } };
                    Quad(sMapVerts, kMaxMapVerts, n, c, sx * 16.0f, ey * 16.0f, sx * 16.0f + 16, ey * 16.0f + 16, 0,
                         topArt ? 128u : 0u, topArt ? tChar : bChar, (topArt ? t8 : b8) | 2u | ((Uint32)dom << 20));
                }
            }
        }
    /* room geometry is shaded by the way each face points (voxel.frag) */
    for (int i = 0; i < n; ++i)
        sMapVerts[i].p[3] |= 0x80000000u;
    sMapVertCount = n;
    if (std::getenv("TMC_VOXEL_DEBUG") || std::getenv("TMC_VOXEL_DUMPMAP"))
        std::fprintf(stderr, "[voxel-dbg] room geometry: %d of %d vertices, %d props\n", n, kMaxMapVerts,
                     sPropCount);
    sBuildShapes = nullptr;
}

const u8 kObjSize[3][4][2] = {
    { { 8, 8 }, { 16, 16 }, { 32, 32 }, { 64, 64 } }, /* square */
    { { 16, 8 }, { 32, 8 }, { 32, 16 }, { 64, 32 } }, /* wide */
    { { 8, 16 }, { 8, 32 }, { 16, 32 }, { 32, 64 } }, /* tall */
};

struct ObjRect {
    int x, y, w, h;         /* on-screen box (double-size affine box included) */
    int sw, sh;             /* sprite pixel size */
    float uv[4][2];         /* sprite pixel at each box corner (flip / affine matrix applied) */
    Uint32 tile, pal, rowParam;
};

bool DecodeObj(int i, bool obj1d, ObjRect& o) {
    const u16 a0 = gOamMem[i * 4 + 0], a1 = gOamMem[i * 4 + 1], a2 = gOamMem[i * 4 + 2];
    const bool affine = (a0 & 0x100) != 0;
    if (!affine && (a0 & 0x200))
        return false; /* hidden */
    if (((a0 >> 10) & 3) == 2)
        return false; /* OBJ window: a mask, not a picture */
    const int shape = (a0 >> 14) & 3;
    if (shape == 3)
        return false;
    const int size = (a1 >> 14) & 3;
    o.sw = kObjSize[shape][size][0];
    o.sh = kObjSize[shape][size][1];
    const bool dbl = affine && (a0 & 0x200);
    o.w = dbl ? o.sw * 2 : o.sw;
    o.h = dbl ? o.sh * 2 : o.sh;
    o.y = a0 & 0xFF;
    if (o.y >= 160)
        o.y -= 256;
    o.x = a1 & 0x1FF;
    if (o.x >= Port_Widescreen_EffectiveViewWidth()) /* as the PPU: wraps to negative */
        o.x -= 512;
    if (affine) {
        /* texel = M * (box point - box centre) + sprite centre, M in 8.8 */
        const int g = (a1 >> 9) & 31;
        const float pa = (s16)gOamMem[g * 16 + 3] / 256.0f, pb = (s16)gOamMem[g * 16 + 7] / 256.0f;
        const float pc = (s16)gOamMem[g * 16 + 11] / 256.0f, pd = (s16)gOamMem[g * 16 + 15] / 256.0f;
        for (int k = 0; k < 4; ++k) {
            const float dx = (k & 1 ? 0.5f : -0.5f) * o.w, dy = (k & 2 ? 0.5f : -0.5f) * o.h;
            o.uv[k][0] = pa * dx + pb * dy + o.sw * 0.5f;
            o.uv[k][1] = pc * dx + pd * dy + o.sh * 0.5f;
        }
    } else {
        const bool hf = (a1 & 0x1000) != 0, vf = (a1 & 0x2000) != 0;
        for (int k = 0; k < 4; ++k) {
            o.uv[k][0] = ((k & 1) != 0) != hf ? (float)o.sw : 0.0f;
            o.uv[k][1] = ((k & 2) != 0) != vf ? (float)o.sh : 0.0f;
        }
    }
    const bool b8 = (a0 & 0x2000) != 0;
    o.tile = a2 & 0x3FF;
    o.pal = a2 >> 12;
    const Uint32 perRow = obj1d ? (Uint32)(o.sw / 8) * (b8 ? 2u : 1u) : 32u;
    /* bits 0-7 tiles per row, 8 8bpp, 9-11 / 12-14 sprite width / height in tiles - 1 */
    o.rowParam = perRow | (b8 ? 256u : 0u) | (Uint32)(o.sw / 8 - 1) << 9 | (Uint32)(o.sh / 8 - 1) << 12;
    return true;
}

/* ponytail: debug knob — TMC_LINK_FRAMES=<dir>: every distinct picture of
 * Link (his own OAM pieces, composed on a 64x64 canvas with his feet at
 * (32, 52)) is written once as <dir>/link_<n>.png, with its animation in
 * <dir>/frames.jsonl. Input for the voxel-Link builder (tools/voxel_link). */
constexpr int kRecW = 64, kRecH = 64, kRecFootX = 32, kRecFootY = 52;

bool WriteRgbaPng(const char* path, const Uint8* rgba, int w, int h) {
    FILE* fp = std::fopen(path, "wb");
    png_structp png = fp ? png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr) : nullptr;
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    bool ok = false;
    if (info && !setjmp(png_jmpbuf(png))) {
        png_init_io(png, fp);
        png_set_IHDR(png, info, w, h, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                     PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        for (int y = 0; y < h; ++y)
            png_write_row(png, rgba + (size_t)y * w * 4);
        png_write_end(png, nullptr);
        ok = true;
    }
    if (png)
        png_destroy_write_struct(&png, info ? &info : nullptr);
    if (fp)
        std::fclose(fp);
    return ok;
}

void RecordLinkFrame(bool obj1d) {
    static const char* dir = std::getenv("TMC_LINK_FRAMES");
    if (!dir || !*dir)
        return;
    static Uint8 canvas[kRecW * kRecH * 4];
    std::memset(canvas, 0, sizeof(canvas));
    int pieces = 0, anchorX = 0, groundY = 0;
    for (int i = 127; i >= 0; --i) { /* lower OAM index on top */
        const PortVoxelOamTag& tag = gPortVoxelOamTags[i];
        if (tag.kind != PORT_VOXEL_OAM_ENTITY || !tag.player)
            continue;
        ObjRect o;
        const u16 a0 = gOamMem[i * 4], a1 = gOamMem[i * 4 + 1];
        if ((a0 & 0x100) || !DecodeObj(i, obj1d, o)) /* affine pieces: skip */
            continue;
        if (tag.parked)
            o.x = tag.trueX, o.y = tag.trueY;
        anchorX = tag.anchorX, groundY = tag.groundY, ++pieces;
        const bool hf = (a1 & 0x1000) != 0, vf = (a1 & 0x2000) != 0, b8 = (o.rowParam & 256) != 0;
        const int perRow = (int)(o.rowParam & 255);
        for (int py = 0; py < o.sh; ++py)
            for (int px = 0; px < o.sw; ++px) {
                const int cx = o.x + px - tag.anchorX + kRecFootX, cy = o.y + py - tag.groundY + kRecFootY;
                if (cx < 0 || cy < 0 || cx >= kRecW || cy >= kRecH)
                    continue;
                const int tx = hf ? o.sw - 1 - px : px, ty = vf ? o.sh - 1 - py : py;
                const int tile = (int)o.tile + (ty / 8) * perRow + (tx / 8) * (b8 ? 2 : 1);
                const Uint32 addr = 0x10000u + (Uint32)(tile & 0x3FF) * 32u;
                int idx;
                if (b8) {
                    idx = gVram[(addr + (ty % 8) * 8 + tx % 8) % sizeof(gVram)];
                } else {
                    const Uint8 b = gVram[(addr + (ty % 8) * 4 + (tx % 8) / 2) % sizeof(gVram)];
                    idx = (tx & 1) ? b >> 4 : b & 15;
                    if (idx)
                        idx += (int)o.pal * 16;
                }
                if (!idx)
                    continue;
                const u16 c = gObjPltt[idx & 255];
                Uint8* d = &canvas[(cy * kRecW + cx) * 4];
                d[0] = (Uint8)((c & 31) * 255 / 31);
                d[1] = (Uint8)(((c >> 5) & 31) * 255 / 31);
                d[2] = (Uint8)(((c >> 10) & 31) * 255 / 31);
                d[3] = 255;
            }
    }
    if (!pieces)
        return;
    Uint64 h = 1469598103934665603ull;
    for (Uint8 b : canvas)
        h = (h ^ b) * 1099511628211ull;
    static std::map<Uint64, int> seen;
    if (seen.count(h))
        return;
    const int n = (int)seen.size();
    seen[h] = n;
    char path[512];
    std::snprintf(path, sizeof(path), "%s/link_%04d.png", dir, n);
    if (!WriteRgbaPng(path, canvas, kRecW, kRecH))
        return;
    const Entity& p = gPlayerEntity.base;
    const u8* anim = Port_GetSpriteAnimationData((u16)p.spriteIndex, p.animIndex);
    const int step = anim && p.animPtr ? (int)(((const u8*)p.animPtr - anim) / 4) - 1 : -1;
    std::snprintf(path, sizeof(path), "%s/frames.jsonl", dir);
    if (FILE* f = std::fopen(path, "a")) {
        std::fprintf(f,
                     "{\"file\": \"link_%04d.png\", \"sprite\": %d, \"anim\": %d, \"frame\": %d, \"step\": %d, "
                     "\"state\": %d, \"flip\": %d, \"player_anim\": %d, \"pieces\": %d}\n",
                     n, p.spriteIndex, p.animIndex, p.frameIndex, step, p.animationState, p.spriteSettings.flipX,
                     gPlayerState.animation, pieces);
        std::fclose(f);
    }
    (void)anchorX, (void)groundY;
}

} // namespace

int Port_Voxel_CurrentArea(void) {
    return SceneApplicable() ? gRoomControls.area : -1;
}

void Port_Voxel_RequestShot(const char* path) {
    std::snprintf(sShotPath, sizeof(sShotPath), "%s", path);
    sShotRequested = true;
}

/* Link and every enemy, feet in world space (X east, Y up, Z south, room
 * pixels): what the walls fade for. */
static void GatherFadeActors(PortVoxelFade& f) {
    int n = 0;
    auto add = [&](const Entity& e) {
        if (n >= kFadeMaxActors)
            return;
        f.actors[n][0] = (float)(e.x.HALF.HI - gRoomControls.origin_x);
        f.actors[n][1] = std::max(0.0f, -(float)e.z.HALF.HI); /* z is up-negative */
        f.actors[n][2] = (float)(e.y.HALF.HI - gRoomControls.origin_y);
        f.actors[n][3] = 0.0f;
        ++n;
    };
    add(gPlayerEntity.base);
    for (int l = 0; l < 9; ++l) {
        LinkedList* list = &gEntityLists[l];
        for (Entity* e = list->first; e && e != (Entity*)list; e = e->next)
            if (e->kind == ENEMY)
                add(*e);
    }
    f.count[0] = n;
}

PortVoxelTileAhead Port_Voxel_TileAhead(void) {
    PortVoxelTileAhead r = {};
    if (!SceneApplicable())
        return r;
    /* animationState >> 1: 0 up, 1 right, 2 down, 3 left */
    static const int kDx[4] = { 0, 1, 0, -1 }, kDy[4] = { -1, 0, 1, 0 };
    const Entity& p = gPlayerEntity.base;
    const int dir = (p.animationState >> 1) & 3;
    const int tx = ((p.x.HALF.HI - gRoomControls.origin_x) >> 4) + kDx[dir];
    const int ty = ((p.y.HALF.HI - gRoomControls.origin_y) >> 4) + kDy[dir];
    if (tx < 0 || ty < 0 || tx >= 64 || ty >= 64)
        return r;
    r.tileType = BottomTileType(ty * 64 + tx);
    if (r.tileType < 0)
        return r;
    r.valid = true;
    r.area = gRoomControls.area;
    const AreaShapes* s = CurrentShapes();
    const auto it = s ? s->tiles.find(r.tileType) : std::map<int, int>::const_iterator{};
    r.shape = (s && it != s->tiles.end()) ? it->second : PORT_VOXEL_SHAPE_AUTO;
    return r;
}

void Port_Voxel_SetTileShape(int area, int tileType, int shape) {
    if (!sShapesLoaded)
        LoadShapes();
    AreaShapes& s = sShapes[area];
    if (shape == PORT_VOXEL_SHAPE_AUTO)
        s.tiles.erase(tileType);
    else
        s.tiles[tileType] = shape;
    SaveShapes();
}

int Port_Voxel_AreaWallTiles(int area) {
    if (!sShapesLoaded)
        LoadShapes();
    const auto it = sShapes.find(area);
    return it == sShapes.end() ? kDefaultWallTiles : it->second.wall;
}

void Port_Voxel_SetAreaWallTiles(int area, int tiles) {
    if (!sShapesLoaded)
        LoadShapes();
    sShapes[area].wall = std::clamp(tiles, 1, 4);
    SaveShapes();
}

/* TMC_VOXEL_DEBUG=1: log each 2D fallback and geometry rebuild, to chase flicker. */
static bool VoxelDebug(void) {
    static int on = -1;
    if (on < 0) {
        const char* e = std::getenv("TMC_VOXEL_DEBUG");
        on = (e && *e && *e != '0') ? 1 : 0;
    }
    return on == 1;
}

static bool sDrewLastFrame = false;

static bool PresentImpl(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swap, int swapW, int swapH) {
    static unsigned sPresentFrame = 0;
    ++sPresentFrame;
    if (sShotPending)
        WriteShot();
    if (!Port_Config_GetVoxelView())
        return false;
    if (!SceneApplicable()) {
        if (VoxelDebug())
            fprintf(stderr, "[voxel-dbg] f=%u 2D fallback: task=%d state=%d substate=%d bg=%p w=%d h=%d bottomPct=%d\n",
                    sPresentFrame, (int)gMain.task, (int)gMain.state, (int)gMain.substate,
                    (void*)gMapBottom.bgSettings, (int)gRoomControls.width, (int)gRoomControls.height,
                    gMapBottom.bgSettings ? MapShownPct(gMapBottom, gMapDataBottomSpecial) : -1);
        return false;
    }
    if (!sInitTried)
        sReady = Init();
    if (!sReady || !EnsureDepth(swapW, swapH))
        return false;

    const int viewW = Port_Widescreen_EffectiveViewWidth();
    const float scrollX = (float)(gRoomControls.scroll_x - gRoomControls.origin_x);
    const float scrollY = (float)(gRoomControls.scroll_y - gRoomControls.origin_y);
    const bool obj1d = (gIoMem[0] & 0x40) != 0;

    /* ---- geometry ---- */
    int n = 0;
    const Uint64 mapKey = MapKey();
    /* Tile art (VRAM) streams in after the maps change, and the height
     * heuristics sample it, so rebuild once more after things settle. */
    static int sSettleFrames = 0;
    bool mapDirty = mapKey != sMapKey;
    if (mapDirty)
        sSettleFrames = 20;
    else if (sSettleFrames > 0 && --sSettleFrames == 0)
        mapDirty = true;
    if (mapDirty) {
        if (VoxelDebug())
            fprintf(stderr, "[voxel-dbg] f=%u rebuild (%s)\n", sPresentFrame,
                    mapKey != sMapKey ? "map changed" : "settle");
        BuildMap();
        sMapKey = mapKey;
    }

    UpdateCam();
    const float pitch = sCam.pitch * 3.14159265f / 180.0f;
    const float yaw = sCam.yaw * 3.14159265f / 180.0f;
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    /* Billboards face the camera: "right" across the view, "up" tilted back
     * by the pitch. At yaw 0 that is world x and the GBA's own rows. */
    const float rightX = cy, rightZ = -sy;
    const float upX = -sy * std::sin(pitch), upY = std::cos(pitch), upZ = -cy * std::sin(pitch);
    /* Each entity's lowest art row (its pieces share anchorX/groundY), so
     * the whole entity lifts together when its art reaches below its feet. */
    int artBottom[128];
    for (int i = 0; i < 128; ++i) {
        artBottom[i] = INT32_MIN;
        const PortVoxelOamTag& t = gPortVoxelOamTags[i];
        ObjRect o;
        if (t.kind == PORT_VOXEL_OAM_ENTITY && DecodeObj(i, obj1d, o))
            artBottom[i] = (t.parked ? t.trueY : o.y) + o.h;
    }
    int entBottom[128];
    for (int i = 0; i < 128; ++i) {
        entBottom[i] = artBottom[i];
        if (artBottom[i] == INT32_MIN)
            continue;
        const PortVoxelOamTag& t = gPortVoxelOamTags[i];
        for (int j = 0; j < 128; ++j) {
            const PortVoxelOamTag& u = gPortVoxelOamTags[j];
            if (artBottom[j] != INT32_MIN && u.anchorX == t.anchorX && u.groundY == t.groundY)
                entBottom[i] = std::max(entBottom[i], artBottom[j]);
        }
    }
    /* Lower OAM index wins on GBA; draw high -> low so it lands last (LEQUAL). */
    for (int i = 127; i >= 0; --i) {
        const PortVoxelOamTag tag = gPortVoxelOamTags[i];
        if (tag.kind == PORT_VOXEL_OAM_HUD)
            continue;
        ObjRect o;
        if (!DecodeObj(i, obj1d, o))
            continue;
        if (tag.parked) { /* beyond OAM's reach: port_draw.c parked it, real spot here */
            o.x = tag.trueX;
            o.y = tag.trueY;
        }
        const float x0 = o.x + scrollX, x1 = x0 + o.w;
        const int sy0 = o.y, sy1 = sy0 + o.h;
        const float elev = tag.layer == 2 ? kTopLayerLift : 0.0f;
        float c[4][3];
        if (tag.kind == PORT_VOXEL_OAM_DECAL) {
            const float y = elev + 0.25f, z0 = sy0 + scrollY, z1 = sy1 + scrollY;
            const float d[4][3] = { { x0, y, z0 }, { x1, y, z0 }, { x0, y, z1 }, { x1, y, z1 } };
            std::memcpy(c, d, sizeof(c));
        } else {
            /* Pixel row sy stands (groundY - sy) px up the billboard, which
             * stands on the entity's ground row, where its shadow is. Rows
             * below that row (feet, a sword swung down) fold flat onto the
             * floor toward the camera: at yaw 0 that reads exactly like the
             * GBA, and from any side they lie on the ground instead of
             * sinking into it or lifting the entity. Entities whose art hangs
             * far below their ground row (bosses anchored at their centre)
             * stand on their lowest row instead, moved toward the camera. */
            const int below = entBottom[i] - tag.groundY;
            const int foot = below > kBossSlack ? entBottom[i] : tag.groundY;
            const float toCam = (float)(foot - tag.groundY);
            const float footZ = tag.groundY + scrollY + cy * toCam, footY = elev + 0.5f;
            /* An entity's pieces turn round its anchor, so they stay together. */
            const float ax = tag.anchorX + scrollX, baseX = ax + sy * toCam;
            auto stand = [&](float a, int row, float* v) {
                const float h = (float)(foot - row);
                v[0] = baseX + rightX * a + upX * h;
                v[1] = footY + upY * h;
                v[2] = footZ + rightZ * a + upZ * h;
            };
            auto lie = [&](float a, int row, float* v) { /* row >= foot */
                const float t = (float)(row - foot);
                v[0] = baseX + rightX * a + sy * t;
                v[1] = footY;
                v[2] = footZ + rightZ * a + cy * t;
            };
            const float a0 = x0 - ax, a1 = x1 - ax;
            const int fold = std::clamp(foot, sy0, sy1);
            const float tf = (float)(fold - sy0) / (float)(sy1 - sy0);
            float uvFold[2][2];
            for (int k = 0; k < 2; ++k) {
                uvFold[0][k] = o.uv[0][k] + (o.uv[2][k] - o.uv[0][k]) * tf;
                uvFold[1][k] = o.uv[1][k] + (o.uv[3][k] - o.uv[1][k]) * tf;
            }
            if (fold > sy0) { /* standing part */
                float d[4][3];
                stand(a0, sy0, d[0]), stand(a1, sy0, d[1]), stand(a0, fold, d[2]), stand(a1, fold, d[3]);
                const float uv[4][2] = { { o.uv[0][0], o.uv[0][1] }, { o.uv[1][0], o.uv[1][1] },
                                         { uvFold[0][0], uvFold[0][1] }, { uvFold[1][0], uvFold[1][1] } };
                QuadUv(sVerts, kMaxVerts, n, d, uv, 1, o.tile, o.pal, o.rowParam);
            }
            if (fold < sy1) { /* lying part */
                float d[4][3];
                lie(a0, fold, d[0]), lie(a1, fold, d[1]), lie(a0, sy1, d[2]), lie(a1, sy1, d[3]);
                const float uv[4][2] = { { uvFold[0][0], uvFold[0][1] }, { uvFold[1][0], uvFold[1][1] },
                                         { o.uv[2][0], o.uv[2][1] }, { o.uv[3][0], o.uv[3][1] } };
                QuadUv(sVerts, kMaxVerts, n, d, uv, 1, o.tile, o.pal, o.rowParam);
            }
            continue;
        }
        QuadUv(sVerts, kMaxVerts, n, c, o.uv, 1, o.tile, o.pal, o.rowParam);
    }
    const int worldVerts = n;
    RecordLinkFrame(obj1d);

    /* HUD: BG0 (text boxes, banners) then HUD sprites, in GBA screen pixels.
     * BG0 goes through the real PPU line renderer so the widescreen HUD
     * anchoring and message-box centring match the 2D view. */
    const bool bg0 = (gIoMem[1] & 0x01) != 0; /* DISPCNT BG0 on */
    /* Screen-space BGs not showing a room map: BG3 (skies, clouds, far water,
     * dark-room veils) and any unbound BG1/BG2 (manager-drawn layers such as
     * the Minish paths' giant leaves or the house under the Minish rafters).
     * Ranked like the PPU (priority, then BG index); those above the bottom
     * map form the foreground overlay (rows 320-479), the rest plus BG3 the
     * backdrop (rows 160-319) drawn full-target at the far plane, so every
     * void shows it. */
    bool fg = false, back = false;
    {
        const void* bgs[4] = { nullptr, &gScreen.bg1, &gScreen.bg2, nullptr };
        auto key = [](int i) { return (gIoMem[8 + i * 2] & 3) * 4 + i; };
        const int bottomBg = LayerBg(gMapBottom.bgSettings);
        const int kb = bottomBg < 0 ? 0 : key(bottomBg);
        int order[3], cnt = 0;
        /* A translucent alpha-blend overlay above the room map (fog, cloud
         * shadows, the darkness veil: 1st target at EVA < 16/16) tints the 2D
         * frame; drawn opaque it would hide what's under it, so skip it. */
        const int bld = gIoMem[0x50], eva = gIoMem[0x52] & 31;
        for (int i = 1; i <= 3; ++i) {
            const bool on = (gIoMem[1] & (1 << i)) != 0 && ((gIoMem[0] & 7) == 0 || i == 1);
            const bool veil = ((bld >> 6) & 3) == 1 && (bld & (1 << i)) && eva < 16 && key(i) < kb;
            const bool bound = gMapBottom.bgSettings == bgs[i] || (gMapTop.bgSettings == bgs[i] && TopMapShown());
            if (on && !veil && (i == 3 || !bound))
                order[cnt++] = i;
        }
        std::sort(order, order + cnt, [&](int a, int b) { return key(a) < key(b); }); /* top-most first */
        /* WIN0/WIN1 masking as the PPU does it (OBJ window counts as outside). */
        const bool win = (gIoMem[1] & 0xE0) != 0;
        auto inWin = [](int reg, int x, int y) {
            const int x1 = gIoMem[reg + 1], x2 = gIoMem[reg], y1 = gIoMem[reg + 5], y2 = gIoMem[reg + 4];
            const bool ix = x1 <= x2 ? (x >= x1 && x < x2) : (x >= x1 || x < x2);
            const bool iy = y1 <= y2 ? (y >= y1 && y < y2) : (y >= y1 || y < y2);
            return ix && iy;
        };
        auto shown = [&](int i, int x, int y) {
            if ((gIoMem[1] & 0x20) && inWin(0x40, x, y))
                return (gIoMem[0x48] >> i) & 1;
            if ((gIoMem[1] & 0x40) && inWin(0x42, x, y))
                return (gIoMem[0x49] >> i) & 1;
            return (gIoMem[0x4A] >> i) & 1;
        };
        static uint32_t line[MODE1_GBA_WIDTH];
        for (int k = 0; k < cnt; ++k) {
            const int i = order[k];
            const bool front = key(i) < kb; /* incl. BG3 drawn over the room (Octorok boss) */
            bool& used = front ? fg : back;
            const int base = front ? 320 : 160;
            if (!used)
                std::memset(&sBg0Pixels[base * MODE1_GBA_WIDTH], 0, sizeof(uint32_t) * MODE1_GBA_WIDTH * 160);
            used = true;
            for (int y = 0; y < 160; ++y) {
                std::memset(line, 0, sizeof(line));
                virtuappu_mode1_render_text_bg_line(i, y, line, nullptr);
                uint32_t* dstRow = &sBg0Pixels[(base + y) * MODE1_GBA_WIDTH];
                for (int x = 0; x < MODE1_GBA_WIDTH; ++x)
                    if (!(dstRow[x] >> 24) && (line[x] >> 24) && (!win || shown(i, x, y)))
                        dstRow[x] = line[x];
            }
        }
    }
    if (fg) {
        const float w = (float)viewW;
        const float c[4][3] = { { 0, 0, 0 }, { w, 0, 0 }, { 0, 160, 0 }, { w, 160, 0 } };
        Quad(sVerts, kMaxVerts, n, c, 0, 0, w, 160, 2, 320, 0, 0);
    }
    if (bg0) {
        std::memset(sBg0Pixels, 0, sizeof(uint32_t) * MODE1_GBA_WIDTH * 160);
        for (int line = 0; line < 160; ++line)
            virtuappu_mode1_render_text_bg_line(0, line, &sBg0Pixels[line * MODE1_GBA_WIDTH], nullptr);
        const float w = (float)viewW;
        const float c[4][3] = { { 0, 0, 0 }, { w, 0, 0 }, { 0, 160, 0 }, { w, 160, 0 } };
        Quad(sVerts, kMaxVerts, n, c, 0, 0, w, 160, 2, 0, 0, 0);
    }
    for (int i = 127; i >= 0; --i) {
        if (gPortVoxelOamTags[i].kind != PORT_VOXEL_OAM_HUD)
            continue;
        ObjRect o;
        if (!DecodeObj(i, obj1d, o))
            continue;
        const float x0 = (float)o.x, y0 = (float)o.y, x1 = x0 + o.w, y1 = y0 + o.h;
        const float c[4][3] = { { x0, y0, 0 }, { x1, y0, 0 }, { x0, y1, 0 }, { x1, y1, 0 } };
        QuadUv(sVerts, kMaxVerts, n, c, o.uv, 1, o.tile, o.pal, o.rowParam);
    }
    const int hudEnd = n;
    if (back) {
        const float w = (float)viewW;
        const float c[4][3] = { { 0, 0, 0 }, { w, 0, 0 }, { 0, 160, 0 }, { w, 160, 0 } };
        Quad(sVerts, kMaxVerts, n, c, 0, 0, w, 160, 2, 160, 0, 0);
    }

    /* ---- upload ---- */
    auto* dst = static_cast<Uint8*>(SDL_MapGPUTransferBuffer(sDev, sXfer, true));
    if (!dst)
        return false;
    std::memcpy(dst, gVram, kVramBytes);
    std::memcpy(dst + kVramBytes, gMapDataBottomSpecial, kMapBytes);
    std::memcpy(dst + kVramBytes + kMapBytes, gMapDataTopSpecial, kMapBytes);
    /* Same colour correction the 2D present applies (F8 -> Display). */
    static Uint32 pal[512];
    for (int i = 0; i < 512; ++i) {
        const u16 c = i < 256 ? gBgPltt[i] : gObjPltt[i - 256];
        const Uint32 r = (c & 31u) << 3, g = ((c >> 5) & 31u) << 3, b = ((c >> 10) & 31u) << 3;
        pal[i] = r | (g << 8) | (b << 16) | 0xFF000000u;
    }
    Port_PPU_ColorCorrectBuffer(pal, 512);
    std::memcpy(dst + kVramBytes + 2 * kMapBytes, pal, sizeof(pal));
    const Uint32 bg0Off = kVramBytes + 2 * kMapBytes + kPalBytes;
    if (bg0 || back || fg) {
        Port_PPU_ColorCorrectBuffer(sBg0Pixels, MODE1_GBA_WIDTH * 480);
        std::memcpy(dst + bg0Off, sBg0Pixels, sizeof(sBg0Pixels));
    }
    const Uint32 vertOff = bg0Off + kBg0Bytes;
    std::memcpy(dst + vertOff, sVerts, (size_t)n * sizeof(Vert));
    SDL_UnmapGPUTransferBuffer(sDev, sXfer);

    SDL_GPUCopyPass* cp = SDL_BeginGPUCopyPass(cmd);
    auto upTex = [&](SDL_GPUTexture* t, Uint32 off, Uint32 w, Uint32 h) {
        SDL_GPUTextureTransferInfo src = {};
        src.transfer_buffer = sXfer;
        src.offset = off;
        src.pixels_per_row = w;
        src.rows_per_layer = h;
        SDL_GPUTextureRegion r = {};
        r.texture = t;
        r.w = w;
        r.h = h;
        r.d = 1;
        SDL_UploadToGPUTexture(cp, &src, &r, false);
    };
    upTex(sVramTex, 0, 256, 384);
    upTex(sMapTex, kVramBytes, 128, 256);
    upTex(sPalTex, kVramBytes + 2 * kMapBytes, 512, 1);
    if (bg0 || back || fg)
        upTex(sBg0Tex, bg0Off, MODE1_GBA_WIDTH, 480);
    const Uint32 dynFirst = (Uint32)kMaxMapVerts; /* per-frame quads live after the room geometry */
    SDL_GPUTransferBufferLocation vsrc = { sXfer, vertOff };
    SDL_GPUBufferRegion vdst = { sVertBuf, dynFirst * (Uint32)sizeof(Vert), (Uint32)(n * sizeof(Vert)) };
    if (n > 0)
        SDL_UploadToGPUBuffer(cp, &vsrc, &vdst, false);
    if (mapDirty) {
        auto* m = static_cast<Uint8*>(SDL_MapGPUTransferBuffer(sDev, sMapXfer, true));
        if (m) {
            std::memcpy(m, sMapVerts, (size_t)sMapVertCount * sizeof(Vert));
            std::memcpy(m + sizeof(sMapVerts), sMaskPixels, sizeof(sMaskPixels));
            SDL_UnmapGPUTransferBuffer(sDev, sMapXfer);
            if (sMapVertCount > 0) {
                SDL_GPUTransferBufferLocation msrc = { sMapXfer, 0 };
                SDL_GPUBufferRegion mdst = { sVertBuf, 0, (Uint32)(sMapVertCount * sizeof(Vert)) };
                SDL_UploadToGPUBuffer(cp, &msrc, &mdst, false);
            }
            SDL_GPUTextureTransferInfo ksrc = {};
            ksrc.transfer_buffer = sMapXfer;
            ksrc.offset = (Uint32)sizeof(sMapVerts);
            ksrc.pixels_per_row = 256;
            ksrc.rows_per_layer = 256;
            SDL_GPUTextureRegion kdst = {};
            kdst.texture = sMaskTex;
            kdst.w = 256;
            kdst.h = 256;
            kdst.d = 1;
            SDL_UploadToGPUTexture(cp, &ksrc, &kdst, false);
        }
    }
    SDL_EndGPUCopyPass(cp);

    /* ---- draw ---- */
    const u16 bd = gBgPltt[0]; /* GBA backdrop colour as the sky */
    uint32_t bdc = ((bd & 31u) << 3) | (((bd >> 5) & 31u) << 11) | (((bd >> 10) & 31u) << 19);
    Port_PPU_ColorCorrectBuffer(&bdc, 1);
    auto drawTo = [&](SDL_GPUTexture* target, SDL_GPUTexture* depth, int tw, int th) {
        SDL_GPUColorTargetInfo col = {};
        col.texture = target;
        col.clear_color = { (bdc & 255) / 255.0f, ((bdc >> 8) & 255) / 255.0f, ((bdc >> 16) & 255) / 255.0f, 1.0f };
        col.load_op = SDL_GPU_LOADOP_CLEAR;
        col.store_op = SDL_GPU_STOREOP_STORE;
        SDL_GPUDepthStencilTargetInfo dep = {};
        dep.texture = depth;
        dep.clear_depth = 1.0f;
        dep.load_op = SDL_GPU_LOADOP_CLEAR;
        dep.store_op = SDL_GPU_STOREOP_DONT_CARE;
        dep.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
        dep.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
        SDL_GPURenderPass* rp = SDL_BeginGPURenderPass(cmd, &col, 1, &dep);
        SDL_BindGPUGraphicsPipeline(rp, sPipeline);
        SDL_GPUTextureSamplerBinding tsb[5] = { { sVramTex, sSampler },
                                               { sMapTex, sSampler },
                                               { sPalTex, sSampler },
                                               { sBg0Tex, sSampler },
                                               { sMaskTex, sSampler } };
        SDL_BindGPUFragmentSamplers(rp, 0, tsb, 5);
        SDL_GPUBufferBinding vb = { sVertBuf, 0 };
        SDL_BindGPUVertexBuffers(rp, 0, &vb, 1);

        /* The camera, for the fade (the 3D pass below); the shader thins
         * room geometry (kind 0) only, never the backdrop, HUD or sprites. */
        const float target3[3] = { scrollX + viewW * 0.5f, 0.0f, scrollY + 80.0f };
        const float eye[3] = { target3[0] + sCam.dist * sy * std::cos(pitch), sCam.dist * std::sin(pitch),
                               target3[2] + sCam.dist * cy * std::cos(pitch) };
        static PortVoxelFade fade;
        fade.cam[0] = eye[0], fade.cam[1] = eye[1], fade.cam[2] = eye[2];
        fade.cam[3] = Port_Config_GetVoxelWallFade() ? 1.0f : 0.0f;
        fade.fade[0] = kFadeKeep, fade.fade[1] = kFadeRadius;
        fade.fade[2] = kFadeFeather, fade.fade[3] = kFadeAim;
        GatherFadeActors(fade);
        SDL_PushGPUFragmentUniformData(cmd, 0, &fade, sizeof(fade));

        /* Backdrop: BG3 stretched over the whole target at the far plane. */
        SDL_GPUViewport vp = { 0, 0, (float)tw, (float)th, 0, 1 };
        SDL_SetGPUViewport(rp, &vp);
        if (n > hudEnd) {
            Mat4 back = Ortho((float)viewW, 160.0f);
            back.m[14] = 0.99999f; /* depth: behind everything */
            SDL_PushGPUVertexUniformData(cmd, 0, back.m, sizeof(back.m));
            SDL_DrawGPUPrimitives(rp, (Uint32)(n - hudEnd), 1, dynFirst + (Uint32)hudEnd, 0);
        }

        /* 3D pass over the whole target. */
        const Mat4 mvp = Mul(Perspective(kFovYDeg * 3.14159265f / 180.0f, (float)tw / (float)th, 32.0f, 4000.0f),
                             LookAt(eye, target3));
        SDL_PushGPUVertexUniformData(cmd, 0, mvp.m, sizeof(mvp.m));
        if (sMapVertCount > 0)
            SDL_DrawGPUPrimitives(rp, (Uint32)sMapVertCount, 1, 0, 0);
        if (worldVerts > 0) {
            SDL_BindGPUGraphicsPipeline(rp, sSpritePipeline);
            SDL_BindGPUFragmentSamplers(rp, 0, tsb, 5);
            SDL_BindGPUVertexBuffers(rp, 0, &vb, 1);
            SDL_PushGPUVertexUniformData(cmd, 0, mvp.m, sizeof(mvp.m));
            SDL_DrawGPUPrimitives(rp, (Uint32)worldVerts, 1, dynFirst, 0);
            SDL_BindGPUGraphicsPipeline(rp, sPipeline);
            SDL_BindGPUFragmentSamplers(rp, 0, tsb, 5);
            SDL_BindGPUVertexBuffers(rp, 0, &vb, 1);
        }

        /* HUD pass in a centred GBA-aspect rect. */
        if (hudEnd > worldVerts) {
            float hw = (float)tw, hh = hw * 160.0f / (float)viewW;
            if (hh > th) {
                hh = (float)th;
                hw = hh * (float)viewW / 160.0f;
            }
            SDL_GPUViewport hvp = { (tw - hw) * 0.5f, (th - hh) * 0.5f, hw, hh, 0, 1 };
            SDL_SetGPUViewport(rp, &hvp);
            const Mat4 ortho = Ortho((float)viewW, 160.0f);
            SDL_PushGPUVertexUniformData(cmd, 0, ortho.m, sizeof(ortho.m));
            SDL_DrawGPUPrimitives(rp, (Uint32)(hudEnd - worldVerts), 1, dynFirst + (Uint32)worldVerts, 0);
        }
        SDL_EndGPURenderPass(rp);
    };
    drawTo(swap, sDepthTex, swapW, swapH);

    /* Debug shot (repro tour): the same frame rendered offscreen and read
     * back, independent of whether the compositor shows the window. */
    if (sShotRequested && !sShotPending && EnsureShotTargets()) {
        drawTo(sShotTex, sShotDepth, kShotW, kShotH);
        SDL_GPUCopyPass* dcp = SDL_BeginGPUCopyPass(cmd);
        SDL_GPUTextureRegion src = {};
        src.texture = sShotTex;
        src.w = kShotW;
        src.h = kShotH;
        src.d = 1;
        SDL_GPUTextureTransferInfo dst = {};
        dst.transfer_buffer = sShotXfer;
        dst.pixels_per_row = kShotW;
        dst.rows_per_layer = kShotH;
        SDL_DownloadFromGPUTexture(dcp, &src, &dst);
        SDL_EndGPUCopyPass(dcp);
        sShotRequested = false;
        sShotPending = true; /* written next frame, after the GPU finished */
    }
    return true;
}

bool Port_Voxel_Present(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swap, int swapW, int swapH) {
    sDrewLastFrame = PresentImpl(cmd, swap, swapW, swapH);
    return sDrewLastFrame;
}

bool Port_Voxel_IsDrawing(void) {
    return sDrewLastFrame;
}

void Port_Voxel_HandleEvent(const SDL_Event* e) {
    if (!sDrewLastFrame)
        return;
    switch (e->type) {
    case SDL_EVENT_MOUSE_WHEEL:
        if (!Port_ImGui_WantsMouse())
            sCam.distGoal = std::clamp(sCam.distGoal * std::pow(0.88f, e->wheel.y), kCamMinDist, kCamMaxDist);
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (e->button.button == SDL_BUTTON_RIGHT)
            sCam.dragging = e->type == SDL_EVENT_MOUSE_BUTTON_DOWN && !Port_ImGui_WantsMouse();
        break;
    case SDL_EVENT_MOUSE_MOTION:
        if (sCam.dragging) {
            sCam.yaw -= e->motion.xrel * kCamMouseDeg;
            sCam.pitch = std::clamp(sCam.pitch + e->motion.yrel * kCamMouseDeg, kCamMinPitch, kCamMaxPitch);
        }
        break;
    case SDL_EVENT_KEY_DOWN:
        if (e->key.scancode == SDL_SCANCODE_H && !e->key.repeat && !Port_DebugMenu_IsOpen() &&
            !Port_ImGui_WantsTextInput())
            ResetCam();
        break;
    default:
        break;
    }
}

int Port_Voxel_ViewTurn(void) {
    if (!sDrewLastFrame || !SceneApplicable())
        return 0;
    return ((int)std::lround(SnappedYaw() / 45.0f) % 8 + 8) % 8;
}

void Port_Voxel_RemapDpad(uint16_t* keyinput) {
    /* Only while walking round a room: menus and text-box choices keep the
     * plain D-pad. */
    if (gMessage.state & MESSAGE_ACTIVE)
        return;
    const int step = Port_Voxel_ViewTurn();
    if (step == 0)
        return;
    const uint16_t k = *keyinput; /* GBA KEYINPUT: 0 = pressed */
    const int ix = (int)!(k & DPAD_RIGHT) - (int)!(k & DPAD_LEFT);
    const int iz = (int)!(k & DPAD_DOWN) - (int)!(k & DPAD_UP);
    if (ix == 0 && iz == 0)
        return;
    /* Screen-space press -> world direction: "up" heads away from the eye. */
    const float a = (float)step * 45.0f * 3.14159265f / 180.0f;
    const float wx = ix * std::cos(a) + iz * std::sin(a), wz = -ix * std::sin(a) + iz * std::cos(a);
    uint16_t out = k | DPAD_RIGHT | DPAD_LEFT | DPAD_UP | DPAD_DOWN;
    if (wx > 0.38f)
        out &= ~DPAD_RIGHT;
    if (wx < -0.38f)
        out &= ~DPAD_LEFT;
    if (wz > 0.38f)
        out &= ~DPAD_DOWN;
    if (wz < -0.38f)
        out &= ~DPAD_UP;
    *keyinput = out;
}

void Port_Voxel_Shutdown(void) {
    if (!sDev)
        return;
    SDL_ReleaseGPUTexture(sDev, sShotTex);
    SDL_ReleaseGPUTexture(sDev, sShotDepth);
    SDL_ReleaseGPUTransferBuffer(sDev, sShotXfer);
    sShotTex = sShotDepth = nullptr;
    sShotXfer = nullptr;
    sShotPending = sShotRequested = false;
    SDL_ReleaseGPUTransferBuffer(sDev, sXfer);
    SDL_ReleaseGPUTransferBuffer(sDev, sMapXfer);
    SDL_ReleaseGPUBuffer(sDev, sVertBuf);
    SDL_ReleaseGPUTexture(sDev, sDepthTex);
    SDL_ReleaseGPUTexture(sDev, sBg0Tex);
    SDL_ReleaseGPUTexture(sDev, sMaskTex);
    SDL_ReleaseGPUTexture(sDev, sPalTex);
    SDL_ReleaseGPUTexture(sDev, sMapTex);
    SDL_ReleaseGPUTexture(sDev, sVramTex);
    SDL_ReleaseGPUSampler(sDev, sSampler);
    SDL_ReleaseGPUGraphicsPipeline(sDev, sPipeline);
    SDL_ReleaseGPUGraphicsPipeline(sDev, sSpritePipeline);
    SDL_ReleaseGPUShader(sDev, sFs);
    SDL_ReleaseGPUShader(sDev, sVs);
    sXfer = nullptr;
    sMapXfer = nullptr;
    sVertBuf = nullptr;
    sMapKey = 0;
    sMapVertCount = 0;
    sDepthTex = nullptr;
    sBg0Tex = sMaskTex = sPalTex = sMapTex = sVramTex = nullptr;
    sSampler = nullptr;
    sPipeline = sSpritePipeline = nullptr;
    sFs = sVs = nullptr;
    sDev = nullptr;
    sReady = false;
    sInitTried = false;
    sDepthW = sDepthH = 0;
}

#endif
