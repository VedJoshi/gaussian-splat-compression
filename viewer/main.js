// Upstream's default intrinsics, used when no ?camera= is given.
let camera = { fx: 1159.5880733038064, fy: 1164.6601287484507 };

const viewerStats = {
    phases: {},
    sortMs: [],
    frameMs: [],
    firstFrameMs: null,
    renderer: null,
    error: null,
};
window.viewerStats = viewerStats;

function fail(message) {
    if (viewerStats.error) return;
    viewerStats.error = message;
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = message;
    console.error(message);
}

function pushCapped(list, value) {
    list.push(value);
    if (list.length > 10000) list.shift();
}

let pendingCapture = null;
window.captureFrame = () =>
    new Promise((resolve) => {
        pendingCapture = resolve;
    });

function encodeBase64(bytes) {
    let text = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
        text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(text);
}

// The frame loop re-inverts the view matrix every frame, so a static view drifts in the last bits.
const nearView = (a, b) => !!a && a.every((v, i) => Math.abs(v - b[i]) <= 1e-6 * (1 + Math.abs(v)));

// gsplat culls means nearer than this; the vertex shader applies it explicitly
// because the clip test alone would only cull below about ZNEAR / 2.2.
const ZNEAR = 0.01;

function getProjectionMatrix(fx, fy, width, height, cx = width / 2, cy = height / 2) {
    const znear = ZNEAR;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [(2 * cx) / width - 1, 1 - (2 * cy) / height, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;
uniform highp sampler2D u_codebook;
uniform bool u_sh;
uniform vec3 campos;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

#define C(k) vec3(s[3 * (k)], s[3 * (k) + 1], s[3 * (k) + 2])

// Degree 1-3 terms in gsplat's order and constants (gsplat/cuda/_torch_impl.py).
// Codebook row coefficient k is gsplat's coefficient k + 1.
vec3 shRest(vec3 dir, uint label) {
    float s[48];
    for (int t = 0; t < 12; t++) {
        vec4 v = texelFetch(u_codebook, ivec2(t, int(label)), 0);
        s[4 * t] = v.x; s[4 * t + 1] = v.y; s[4 * t + 2] = v.z; s[4 * t + 3] = v.w;
    }
    float x = dir.x, y = dir.y, z = dir.z;
    vec3 r = 0.48860251190292 * (-y * C(0) + z * C(1) - x * C(2));

    float z2 = z * z;
    float fTmpB = -1.092548430592079 * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.0 * x * y;
    r += 0.5462742152960395 * fS1 * C(3)
       + fTmpB * y * C(4)
       + (0.9461746957575601 * z2 - 0.3153915652525201) * C(5)
       + fTmpB * x * C(6)
       + 0.5462742152960395 * fC1 * C(7);

    float fTmpC = -2.285228997322329 * z2 + 0.4570457994644658;
    float fTmpB3 = 1.445305721320277 * z;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    r += -0.5900435899266435 * fS2 * C(8)
       + fTmpB3 * fS1 * C(9)
       + fTmpC * y * C(10)
       + z * (1.865881662950577 * z2 - 1.119528997770346) * C(11)
       + fTmpC * x * C(12)
       + fTmpB3 * fC1 * C(13)
       - 0.5900435899266435 * fC2 * C(14);
    return r;
}

void main () {
    ivec2 base = ivec2((uint(index) & 0x3ffu) * 3u, uint(index) >> 10);
    uvec4 cen = texelFetch(u_texture, base, 0);
    vec3 world = uintBitsToFloat(cen.xyz);
    vec4 cam = view * vec4(world, 1);
    vec4 pos2d = projection * cam;

    vec4 appearance = uintBitsToFloat(texelFetch(u_texture, base + ivec2(2, 0), 0));
    if (cam.z < ${ZNEAR} || appearance.a < 1.0 / 255.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, base + ivec2(1, 0), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = 0.25 * mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    // gsplat's persp_proj clamps the Jacobian's view ray to 0.3 tan-fov beyond the image edges.
    vec2 principal = vec2(projection[2][0] + 1.0, 1.0 - projection[2][1]) * 0.5 * viewport;
    vec2 margin = 0.15 * viewport / focal;
    vec2 limNeg = principal / focal + margin, limPos = (viewport - principal) / focal + margin;
    vec2 t = cam.z * clamp(cam.xy / cam.z, -limNeg, limPos);
    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * t.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * t.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;
    // gsplat's eps2d low-pass, in pixels squared.
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;

    // gsplat culls on the opacity-aware bounding box, not the centre.
    vec2 vCenter = vec2(pos2d) / pos2d.w;
    vec2 pixel = (vCenter + 1.0) * 0.5 * viewport;
    float extend = min(3.33, sqrt(2.0 * log(appearance.a * 255.0)));
    vec2 bbox = ceil(extend * sqrt(vec2(cov2d[0][0], cov2d[1][1])));
    if (any(lessThanEqual(pixel + bbox, vec2(0.0))) || any(greaterThanEqual(pixel - bbox, viewport))) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    // One standard deviation in pixels along each principal axis.
    vec2 majorAxis = min(sqrt(lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vec3 rgb = appearance.rgb;
    // gsplat takes view directions from the camera centre to each mean.
    if (u_sh) rgb += shRest(normalize(world - campos), cen.w);
    vColor = vec4(max(rgb, 0.0), appearance.a);
    vPosition = position;

    gl_Position = vec4(
        vCenter + 2.0 * (position.x * majorAxis + position.y * minorAxis) / viewport,
        0.0, 1.0);
}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    // gsplat's rasteriser: alpha capped at 0.999 and skipped below 1/255.
    float alpha = min(0.999, vColor.a * exp(-0.5 * dot(vPosition, vPosition)));
    if (alpha < 1.0 / 255.0) discard;
    fragColor = vec4(alpha * vColor.rgb, alpha);
}
`.trim();

let viewMatrix = [
    0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,
    0.03, 6.55, 1,
];
async function main() {
    let carousel = true;
    const params = new URLSearchParams(location.search);
    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        carousel = false;
    } catch (err) {}
    if (params.get("camera")) {
        const c = JSON.parse(params.get("camera"));
        viewMatrix = c.view;
        camera = { fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy };
        carousel = params.get("orbit") === "1";
    }
    const homeViewMatrix = viewMatrix;
    const scene = params.get("url");
    if (!scene) throw new Error("no scene: pass ?url=<scene.splatc>");
    const sceneName = scene.split("/").pop().replace(/\.splatc$/, "");
    const title = sceneName.charAt(0).toUpperCase() + sceneName.slice(1);
    document.getElementById("scene-name").innerText = title;
    document.title = `${title} · Compressing 3D Gaussian Splats`;

    const worker = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });
    worker.onerror = (e) => fail(`viewer worker failed to start: ${e.message}`);

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        throw new Error(`vertex shader failed to compile: ${gl.getShaderInfoLog(vertexShader)}`);

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        throw new Error(`fragment shader failed to compile: ${gl.getShaderInfoLog(fragmentShader)}`);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw new Error(`shader program failed to link: ${gl.getProgramInfoLog(program)}`);
    gl.useProgram(program);

    const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
    viewerStats.renderer = debugInfo
        ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
        : gl.getParameter(gl.RENDERER);

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");
    const u_campos = gl.getUniformLocation(program, "campos");
    const u_sh = gl.getUniformLocation(program, "u_sh");
    const shRequested = params.get("sh") !== "0";
    const codebookTexture = gl.createTexture();
    gl.uniform1i(gl.getUniformLocation(program, "u_codebook"), 1);
    gl.uniform1i(u_sh, 0);

    // sqrt(2 ln 255): beyond this many standard deviations alpha is below 1/255.
    const extent = 3.33;
    const triangleVertices = new Float32Array([-extent, -extent, extent, -extent, extent, extent, -extent, extent]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const resize = () => {
        gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));
        projectionMatrix = getProjectionMatrix(camera.fx, camera.fy, innerWidth, innerHeight, camera.cx, camera.cy);
        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));
        gl.canvas.width = innerWidth;
        gl.canvas.height = innerHeight;
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();

    let sortedView = null;
    let sceneInfo = null;
    worker.onmessage = (e) => {
        if (e.data.error) {
            fail(e.data.error);
        } else if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            const uploadStart = performance.now();
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
            const { codebook, hasCodebook } = e.data;
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, codebookTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, codebook.width, codebook.height, 0, gl.RGBA, gl.FLOAT, codebook.data);
            gl.activeTexture(gl.TEXTURE0);
            gl.uniform1i(u_sh, shRequested && hasCodebook ? 1 : 0);
            viewerStats.phases = { ...e.data.phases, upload: performance.now() - uploadStart };
            sceneInfo = { count: e.data.count, bytes: e.data.bytes };
        } else if (e.data.depthIndex) {
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, e.data.depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.count;
            sortedView = e.data.viewProj;
            pushCapped(viewerStats.sortMs, e.data.sortMs);
        }
    };
    worker.postMessage({ url: new URL(scene, location.href).href });

    let activeKeys = [];

    window.addEventListener("keydown", (e) => {
        // if (document.activeElement != document.body) return;
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (e.code == "KeyV") {
            location.hash =
                "#" +
                JSON.stringify(
                    viewMatrix.map((k) => Math.round(k * 100) / 100),
                );
        } else if (e.code === "KeyP") {
            carousel = true;
        }
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener(
        "wheel",
        (e) => {
            carousel = false;
            e.preventDefault();
            const lineHeight = 10;
            const scale =
                e.deltaMode == 1
                    ? lineHeight
                    : e.deltaMode == 2
                      ? innerHeight
                      : 1;
            let inv = invert4(viewMatrix);
            if (e.shiftKey) {
                inv = translate4(
                    inv,
                    (e.deltaX * scale) / innerWidth,
                    (e.deltaY * scale) / innerHeight,
                    0,
                );
            } else if (e.ctrlKey || e.metaKey) {
                // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
                // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
                // let preY = inv[13];
                inv = translate4(
                    inv,
                    0,
                    0,
                    (-10 * (e.deltaY * scale)) / innerHeight,
                );
                // inv[13] = preY;
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }

            viewMatrix = invert4(inv);
        },
        { passive: false },
    );

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / innerWidth;
            let dy = (5 * (e.clientY - startY)) / innerHeight;
            let d = 4;

            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            // inv = rotateY(inv, );
            // let preY = inv[13];
            inv = translate4(
                inv,
                (-10 * (e.clientX - startX)) / innerWidth,
                0,
                (10 * (e.clientY - startY)) / innerHeight,
            );
            // inv[13] = preY;
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    let altX = 0,
        altY = 0;
    canvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                carousel = false;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                carousel = false;
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                // inv = translate4(inv,  -x, -y, -z);
                // inv = translate4(inv,  x, y, z);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                // alert('beep')
                const dtheta =
                    Math.atan2(startY - altY, startX - altX) -
                    Math.atan2(
                        e.touches[0].clientY - e.touches[1].clientY,
                        e.touches[0].clientX - e.touches[1].clientX,
                    );
                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                    );
                const dx =
                    (e.touches[0].clientX +
                        e.touches[1].clientX -
                        (startX + altX)) /
                    2;
                const dy =
                    (e.touches[0].clientY +
                        e.touches[1].clientY -
                        (startY + altY)) /
                    2;
                let inv = invert4(viewMatrix);
                // inv = translate4(inv,  0, 0, d);
                inv = rotate4(inv, dtheta, 0, 0, 1);

                inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);

                // let preY = inv[13];
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));
                // inv[13] = preY;

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );

    let jumpDelta = 0;
    let vertexCount = 0;

    let lastFrame = 0;
    let avgFps = 0;
    let start = 0;

    const frame = (now) => {
        let inv = invert4(viewMatrix);
        let shiftKey =
            activeKeys.includes("Shift") ||
            activeKeys.includes("ShiftLeft") ||
            activeKeys.includes("ShiftRight");

        if (activeKeys.includes("ArrowUp")) {
            if (shiftKey) {
                inv = translate4(inv, 0, -0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, 0.1);
            }
        }
        if (activeKeys.includes("ArrowDown")) {
            if (shiftKey) {
                inv = translate4(inv, 0, 0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, -0.1);
            }
        }
        if (activeKeys.includes("ArrowLeft"))
            inv = translate4(inv, -0.03, 0, 0);
        //
        if (activeKeys.includes("ArrowRight"))
            inv = translate4(inv, 0.03, 0, 0);
        // inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
        if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
        if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
        if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
        if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

        const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
        let isJumping = activeKeys.includes("Space");
        for (let gamepad of gamepads) {
            if (!gamepad) continue;

            const axisThreshold = 0.1; // Threshold to detect when the axis is intentionally moved
            const moveSpeed = 0.06;
            const rotateSpeed = 0.02;

            // Assuming the left stick controls translation (axes 0 and 1)
            if (Math.abs(gamepad.axes[0]) > axisThreshold) {
                inv = translate4(inv, moveSpeed * gamepad.axes[0], 0, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[1]) > axisThreshold) {
                inv = translate4(inv, 0, 0, -moveSpeed * gamepad.axes[1]);
                carousel = false;
            }
            if (gamepad.buttons[12].pressed || gamepad.buttons[13].pressed) {
                inv = translate4(
                    inv,
                    0,
                    -moveSpeed *
                        (gamepad.buttons[12].pressed -
                            gamepad.buttons[13].pressed),
                    0,
                );
                carousel = false;
            }

            if (gamepad.buttons[14].pressed || gamepad.buttons[15].pressed) {
                inv = translate4(
                    inv,
                    -moveSpeed *
                        (gamepad.buttons[14].pressed -
                            gamepad.buttons[15].pressed),
                    0,
                    0,
                );
                carousel = false;
            }

            // Assuming the right stick controls rotation (axes 2 and 3)
            if (Math.abs(gamepad.axes[2]) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * gamepad.axes[2], 0, 1, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[3]) > axisThreshold) {
                inv = rotate4(inv, -rotateSpeed * gamepad.axes[3], 1, 0, 0);
                carousel = false;
            }

            let tiltAxis = gamepad.buttons[6].value - gamepad.buttons[7].value;
            if (Math.abs(tiltAxis) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * tiltAxis, 0, 0, 1);
                carousel = false;
            }
            if (gamepad.buttons[0].pressed) {
                isJumping = true;
                carousel = false;
            }
            if (gamepad.buttons[3].pressed) {
                carousel = true;
            }
        }

        if (
            ["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))
        ) {
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(
                inv,
                activeKeys.includes("KeyJ")
                    ? -0.05
                    : activeKeys.includes("KeyL")
                      ? 0.05
                      : 0,
                0,
                1,
                0,
            );
            inv = rotate4(
                inv,
                activeKeys.includes("KeyI")
                    ? 0.05
                    : activeKeys.includes("KeyK")
                      ? -0.05
                      : 0,
                1,
                0,
                0,
            );
            inv = translate4(inv, 0, 0, -d);
        }

        viewMatrix = invert4(inv);

        if (carousel) {
            let inv = invert4(homeViewMatrix);

            const t = Math.sin((Date.now() - start) / 5000);
            inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
            inv = rotate4(inv, -0.6 * t, 0, 1, 0);

            viewMatrix = invert4(inv);
        }

        if (isJumping) {
            jumpDelta = Math.min(1, jumpDelta + 0.05);
        } else {
            jumpDelta = Math.max(0, jumpDelta - 0.05);
        }

        let inv2 = invert4(viewMatrix);
        inv2 = translate4(inv2, 0, -jumpDelta, 0);
        inv2 = rotate4(inv2, -0.1 * jumpDelta, 1, 0, 0);
        let actualViewMatrix = invert4(inv2);

        const viewProj = multiply4(projectionMatrix, actualViewMatrix);
        worker.postMessage({ view: viewProj });

        if (lastFrame) pushCapped(viewerStats.frameMs, now - lastFrame);
        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            const camToWorld = invert4(actualViewMatrix);
            gl.uniform3fv(u_campos, new Float32Array([camToWorld[12], camToWorld[13], camToWorld[14]]));
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            if (viewerStats.firstFrameMs === null) {
                viewerStats.firstFrameMs = performance.now();
                document.getElementById("scene-stats").innerText =
                    `${sceneInfo.count.toLocaleString("en-US")} Gaussians · ${(sceneInfo.bytes / 1e6).toFixed(1)} MB · ` +
                    `loaded in ${(viewerStats.firstFrameMs / 1000).toFixed(1)} s`;
            }
            if (pendingCapture && nearView(sortedView, viewProj)) {
                const { width, height } = gl.canvas;
                const pixels = new Uint8Array(width * height * 4);
                gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
                pendingCapture({ width, height, pixels: encodeBase64(pixels) });
                pendingCapture = null;
            }
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            if (!viewerStats.error) document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }
        fps.innerText = Math.round(avgFps) + " fps";
        lastFrame = now;
        requestAnimationFrame(frame);
    };

    requestAnimationFrame(frame);

    window.addEventListener("hashchange", (e) => {
        try {
            viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
            carousel = false;
        } catch (err) {}
    });
}

main().catch((err) => fail(String(err && err.message ? err.message : err)));
