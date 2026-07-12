// SPDX-License-Identifier: MIT
// Copyright (c) 2026-present K. S. Ernest (iFire) Lee
//
// core/calib/tic_calib — implementation of the TIC calibrator (mount/drift) and
// the deploy correction.  Extracted from the old in-driver path; now a core lib
// the server links.  Input layout and the device-quaternion frame match
// core/calib/train/device_apply.py exactly: per sensor per frame
//   [ normalize(R_device·accel)(3), normalize(R_device·mag)(3), R_device row(9) ].
#include "tic_calib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "netfwd.h"

// ── Orientation math (Vec3/Mat3 + quaternion/rotation helpers) ───────────────
typedef struct {
	float x, y, z;
} Vec3;
typedef struct {
	Vec3 r0, r1, r2;
} Mat3;

static inline Vec3 v_sub(Vec3 a, Vec3 b) {
	return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}
static inline Vec3 v_add(Vec3 a, Vec3 b) {
	return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}
static inline Vec3 v_scale(float s, Vec3 a) {
	return (Vec3){s * a.x, s * a.y, s * a.z};
}
static inline float v_dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline Vec3 v_cross(Vec3 a, Vec3 b) {
	return (Vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline Vec3 v_normalize(Vec3 a) {
	float n = sqrtf(v_dot(a, a));
	if (n < 1e-9f) {
		return (Vec3){0.f, 0.f, 0.f};
	}
	return v_scale(1.0f / n, a);
}

// Quaternion → rotation matrix (rows), matching normalise_quat's w,x,y,z order.
static Mat3 m_quat(Quat q) {
	float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
	if (n < 1e-9f) {
		return (Mat3){{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	}
	float w = q.w / n, x = q.x / n, y = q.y / n, z = q.z / n;
	return (Mat3){{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
	              {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
	              {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
}

// Rotation matrix (rows) → quaternion.  Inverse of m_quat.
static Quat m_to_quat(Mat3 m) {
	float tr = m.r0.x + m.r1.y + m.r2.z;
	Quat q;
	if (tr > 0.f) {
		float s = sqrtf(tr + 1.0f) * 2.0f;
		q = (Quat){0.25f * s, (m.r2.y - m.r1.z) / s, (m.r0.z - m.r2.x) / s, (m.r1.x - m.r0.y) / s};
	} else if (m.r0.x > m.r1.y && m.r0.x > m.r2.z) {
		float s = sqrtf(1.0f + m.r0.x - m.r1.y - m.r2.z) * 2.0f;
		q = (Quat){(m.r2.y - m.r1.z) / s, 0.25f * s, (m.r0.y + m.r1.x) / s, (m.r0.z + m.r2.x) / s};
	} else if (m.r1.y > m.r2.z) {
		float s = sqrtf(1.0f + m.r1.y - m.r0.x - m.r2.z) * 2.0f;
		q = (Quat){(m.r0.z - m.r2.x) / s, (m.r0.y + m.r1.x) / s, 0.25f * s, (m.r1.z + m.r2.y) / s};
	} else {
		float s = sqrtf(1.0f + m.r2.z - m.r0.x - m.r1.y) * 2.0f;
		q = (Quat){(m.r1.x - m.r0.y) / s, (m.r0.z + m.r2.x) / s, (m.r1.z + m.r2.y) / s, 0.25f * s};
	}
	float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
	if (n < 1e-9f) {
		return (Quat){1, 0, 0, 0};
	}
	return (Quat){q.w / n, q.x / n, q.y / n, q.z / n};
}

static Mat3 m_transpose(Mat3 m) {
	return (Mat3){{m.r0.x, m.r1.x, m.r2.x}, {m.r0.y, m.r1.y, m.r2.y}, {m.r0.z, m.r1.z, m.r2.z}};
}

// Matrix product: (m_mul a b) applied to v == a applied to (b applied to v).
static Mat3 m_mul(Mat3 a, Mat3 b) {
	Mat3 r;
	r.r0 = v_add(v_scale(a.r0.x, b.r0), v_add(v_scale(a.r0.y, b.r1), v_scale(a.r0.z, b.r2)));
	r.r1 = v_add(v_scale(a.r1.x, b.r0), v_add(v_scale(a.r1.y, b.r1), v_scale(a.r1.z, b.r2)));
	r.r2 = v_add(v_scale(a.r2.x, b.r0), v_add(v_scale(a.r2.y, b.r1), v_scale(a.r2.z, b.r2)));
	return r;
}

// Gram–Schmidt rows back into a proper rotation (3rd = 1st×2nd).
static Mat3 m_orthonormalize(Mat3 m) {
	Vec3 a = v_normalize(m.r0);
	Vec3 b = v_normalize(v_sub(m.r1, v_scale(v_dot(a, m.r1), a)));
	Vec3 c = v_cross(a, b);
	return (Mat3){a, b, c};
}

static inline Vec3 m_apply(Mat3 m, Vec3 v) {  // R · v
	return (Vec3){v_dot(m.r0, v), v_dot(m.r1, v), v_dot(m.r2, v)};
}

// ── TIC calibrator state ─────────────────────────────────────────────────────
#define TIC_S 32
#define TIC_NSENS 15                       // device body-part count
#define TIC_FEAT 15                        // accel(3) + mag(3) + rot(9)
#define TIC_NIN (TIC_NSENS * TIC_FEAT)     // 225
#define TIC_NOUT 90                        // 15 sensors × 6D, per head

static NetfwdCtx *g_tic;
static int g_tic_init_tried;
static float g_tic_x[TIC_S * TIC_NIN];     // current window, frame-major
static int g_tic_fill;                     // snapshots written into the window
static uint64_t g_tic_snap_ms;             // timestamp of the last snapshot
static Vec3 g_tic_a[TIC_NSENS], g_tic_m[TIC_NSENS];  // latest sample, carried forward
static Quat g_tic_q[TIC_NSENS];
static int g_tic_live[TIC_NSENS], g_tic_nlive;
static Mat3 g_tic_mount[TIC_NSENS], g_tic_drift[TIC_NSENS];  // predicted R_BS, R_DG
static int g_tic_ready;                    // a window has produced a calibration

// 6D (two 3-vecs) → rotation, columns [a b c] (Zhou et al.) — matches
// device_apply.sixd_angle's np.stack([a, b, c], 1).
static Mat3 sixd_to_m(const float *v) {
	Vec3 a = v_normalize((Vec3){v[0], v[1], v[2]});
	Vec3 braw = {v[3], v[4], v[5]};
	Vec3 b = v_normalize(v_sub(braw, v_scale(v_dot(a, braw), a)));
	Vec3 c = v_cross(a, b);
	return (Mat3){{a.x, b.x, c.x}, {a.y, b.y, c.y}, {a.z, b.z, c.z}};
}

// Load wtrained.bin + the 5 SPIR-V kernels from `dir` and build a netfwd context for
// cfg `c`.  Returns null if the weights file is missing/short or the GPU/kernels are
// unavailable, so callers degrade gracefully.  Shared by the live calibrator and the
// self-test.
static NetfwdCtx *tic_create_from_dir(const NetfwdCfg *c, const char *dir) {
	char path[1024];
	snprintf(path, sizeof(path), "%s/wtrained.bin", dir);
	FILE *f = fopen(path, "rb");
	if (!f) {
		return NULL;
	}
	size_t nw = netfwd_weight_count(c);
	float *weights = (float *)malloc(nw * sizeof(float));
	size_t got = weights ? fread(weights, sizeof(float), nw, f) : 0;
	fclose(f);
	if (got != nw) {
		free(weights);
		return NULL;
	}
	char gemm[1024], lin[1024], ln[1024], attn[1024], ew[1024];
	snprintf(gemm, sizeof(gemm), "%s/gemm.spv", dir);
	snprintf(lin, sizeof(lin), "%s/lin.spv", dir);
	snprintf(ln, sizeof(ln), "%s/ln.spv", dir);
	snprintf(attn, sizeof(attn), "%s/attn.spv", dir);
	snprintf(ew, sizeof(ew), "%s/ew.spv", dir);
	const char *spv[5] = {gemm, lin, ln, attn, ew};
	NetfwdCtx *ctx = netfwd_create(c, spv, weights, nw);  // null if Vulkan/kernels unavailable
	free(weights);
	return ctx;
}

void tic_calib_init(void) {
	g_tic_init_tried = 1;
	const char *dir = getenv("SINEW_TIC_DIR");
	if (!dir) {
		dir = ".";
	}
	NetfwdCfg c = {TIC_S, TIC_NIN, 64, 4, 128, TIC_NOUT, 2};  // D,H,F,STACK = device_apply.py
	g_tic = tic_create_from_dir(&c, dir);
}

// One frame in.  Carries the sample forward per sensor and, once all 15 are live,
// snapshots them whenever the timestamp advances (matching device_apply.py); every
// TIC_S snapshots it runs one forward and refreshes the calibration.
void tic_calib_push(int s, Accel a, Accel m, Quat q, uint64_t ms) {
	if (s < 0 || s >= TIC_NSENS) {
		return;
	}
	if (!g_tic && !g_tic_init_tried) {
		tic_calib_init();
	}
	if (!g_tic) {
		return;
	}
	if (!g_tic_live[s]) {
		g_tic_live[s] = 1;
		g_tic_nlive++;
	}
	g_tic_a[s] = (Vec3){a.x, a.y, a.z};
	g_tic_m[s] = (Vec3){m.x, m.y, m.z};
	g_tic_q[s] = q;
	if (g_tic_nlive < TIC_NSENS || ms == g_tic_snap_ms) {
		g_tic_snap_ms = ms;
		return;  // wait for a full sensor set and a fresh timestamp
	}
	g_tic_snap_ms = ms;
	float *row = g_tic_x + (size_t)g_tic_fill * TIC_NIN;
	for (int i = 0; i < TIC_NSENS; i++) {
		Mat3 R = m_quat(g_tic_q[i]);          // device fused orientation (drifts)
		Vec3 ag = v_normalize(m_apply(R, g_tic_a[i]));  // global accel dir (grav-norm)
		Vec3 mg = v_normalize(m_apply(R, g_tic_m[i]));  // R_DG · m_world
		float *o = row + (size_t)i * TIC_FEAT;
		o[0] = ag.x; o[1] = ag.y; o[2] = ag.z;
		o[3] = mg.x; o[4] = mg.y; o[5] = mg.z;
		o[6] = R.r0.x; o[7] = R.r0.y; o[8] = R.r0.z;
		o[9] = R.r1.x; o[10] = R.r1.y; o[11] = R.r1.z;
		o[12] = R.r2.x; o[13] = R.r2.y; o[14] = R.r2.z;
	}
	if (++g_tic_fill < TIC_S) {
		return;
	}
	g_tic_fill = 0;
	float out[2 * TIC_NOUT];  // [global 90, local 90]
	if (netfwd_forward(g_tic, g_tic_x, out)) {
		for (int i = 0; i < TIC_NSENS; i++) {
			g_tic_drift[i] = sixd_to_m(out + (size_t)i * 6);             // global head = R_DG
			g_tic_mount[i] = sixd_to_m(out + TIC_NOUT + (size_t)i * 6);  // local head  = R_BS
		}
		g_tic_ready = 1;
	}
}

// R_clean = R_DGᵀ · R_device · R_BSᵀ.  Returns 1 and writes *out when a calibration
// is ready; 0 otherwise (caller passes the uncalibrated device quaternion through).
int tic_calib_apply(int s, Quat device, Quat *out) {
	if (!g_tic_ready || s < 0 || s >= TIC_NSENS) {
		return 0;
	}
	Mat3 rc = m_mul(m_transpose(g_tic_drift[s]), m_mul(m_quat(device), m_transpose(g_tic_mount[s])));
	*out = m_to_quat(m_orthonormalize(rc));
	return 1;
}

// Geodesic angle between two rotations, degrees: arccos((tr(AᵀB) − 1) / 2).
static double geo_deg(Mat3 a, Mat3 b) {
	Mat3 m = m_mul(m_transpose(a), b);
	double c = (m.r0.x + m.r1.y + m.r2.z - 1.0) / 2.0;
	c = c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
	return acos(c) * (180.0 / 3.14159265358979323846);
}

// On-device accuracy check: run labeled caldata windows (export_selftest.py) through
// the SAME deploy path the server uses — netfwd_forward + the 6D decode (sixd_to_m) +
// the R_DGᵀ·R_sensor·R_BSᵀ correction — and report OME against ground truth, the number
// train_tic.py's pose_ome prints offline.  Confirms the C deploy reproduces the offline
// result rather than adding error.  Needs wtrained.bin + the 5 .spv in spv_dir (run.sh)
// and a GPU; returns 0 on success, nonzero on any setup/read failure.
int tic_selftest(const char *window_bin, const char *spv_dir) {
	FILE *f = fopen(window_bin, "rb");
	if (!f) {
		fprintf(stderr, "tic-selftest: cannot open %s\n", window_bin);
		return 1;
	}
	int hdr[4];  // K, S, NIN, NOUT
	if (fread(hdr, sizeof(int), 4, f) != 4) {
		fclose(f);
		fprintf(stderr, "tic-selftest: bad header in %s\n", window_bin);
		return 1;
	}
	int K = hdr[0], S = hdr[1], NIN = hdr[2], NOUT = hdr[3];
	int nsens = TIC_NSENS, cps = NIN / nsens, mid = S / 2;
	if (NOUT != nsens * 6) {
		fclose(f);
		fprintf(stderr, "tic-selftest: NOUT %d != nsens*6 (%d)\n", NOUT, nsens * 6);
		return 1;
	}
	NetfwdCfg c = {S, NIN, 64, 4, 128, NOUT, 2};
	NetfwdCtx *ctx = tic_create_from_dir(&c, spv_dir ? spv_dir : ".");
	if (!ctx) {
		fclose(f);
		fprintf(stderr, "tic-selftest: netfwd init failed (wtrained.bin/.spv/GPU in %s?)\n",
		        spv_dir ? spv_dir : ".");
		return 1;
	}
	float *x = (float *)malloc((size_t)S * NIN * sizeof(float));
	float *y = (float *)malloc((size_t)2 * NOUT * sizeof(float));
	float *out = (float *)malloc((size_t)2 * NOUT * sizeof(float));
	double ome_sum = 0.0, omer_sum = 0.0, id_sum = 0.0, st_sum = 0.0;
	double ome_bone[TIC_NSENS] = {0};
	long n = 0;
	int evaluated = 0;
	for (int w = 0; w < K; w++) {
		if (fread(x, sizeof(float), (size_t)S * NIN, f) != (size_t)S * NIN ||
		    fread(y, sizeof(float), (size_t)2 * NOUT, f) != (size_t)2 * NOUT) {
			fprintf(stderr, "tic-selftest: short read at window %d\n", w);
			break;
		}
		if (!netfwd_forward(ctx, x, out)) {
			fprintf(stderr, "tic-selftest: forward failed at window %d\n", w);
			break;
		}
		Mat3 rec[TIC_NSENS], tru[TIC_NSENS], rsa[TIC_NSENS], rota[TIC_NSENS];
		for (int s = 0; s < nsens; s++) {
			const float *r = x + (size_t)mid * NIN + (size_t)s * cps + (cps - 9);  // R_sensor, row-major
			Mat3 rs = {{r[0], r[1], r[2]}, {r[3], r[4], r[5]}, {r[6], r[7], r[8]}};
			Mat3 rdp = sixd_to_m(out + s * 6), rop = sixd_to_m(out + NOUT + s * 6);  // pred drift, mount
			Mat3 rdt = sixd_to_m(y + s * 6), rot = sixd_to_m(y + NOUT + s * 6);      // true drift, mount
			rec[s] = m_mul(m_transpose(rdp), m_mul(rs, m_transpose(rop)));  // = tic_calib_apply (TIC)
			tru[s] = m_mul(m_transpose(rdt), m_mul(rs, m_transpose(rot)));  // ground-truth clean bone
			rsa[s] = rs;
			rota[s] = rot;
		}
		for (int s = 0; s < nsens; s++) {
			double e = geo_deg(rec[s], tru[s]);
			ome_sum += e;                                                                  // TIC absolute
			ome_bone[s] += e;                                                              // per-bone
			omer_sum += geo_deg(m_mul(m_transpose(rec[0]), rec[s]),                         // TIC root-relative
			                    m_mul(m_transpose(tru[0]), tru[s]));
			id_sum += geo_deg(rsa[s], tru[s]);                                  // no calibration (identity)
			st_sum += geo_deg(m_mul(rsa[s], m_transpose(rota[s])), tru[s]);     // static: oracle mount, no drift
			n++;
		}
		evaluated++;
	}
	fclose(f);
	free(x);
	free(y);
	free(out);
	netfwd_destroy(ctx);
	if (n == 0) {
		fprintf(stderr, "tic-selftest: no windows evaluated\n");
		return 1;
	}
	printf("tic-selftest: %d windows × %d sensors\n", evaluated, nsens);
	printf("  no-calibration (identity)        OME %6.2f°\n", id_sum / n);
	printf("  static calibration (mount only)  OME %6.2f°\n", st_sum / n);
	printf("  TIC (ours: mount + drift)        OME %6.2f°   (root-relative %5.2f°)\n",
	       ome_sum / n, omer_sum / n);
	printf("  reference: transformerimucalib2025 §5.4  TIC 15.20° vs static 49.98°;  "
	       "our offline ≈18.5° cross-study held-out.\n");
	static const char *bone[TIC_NSENS] = {
	    "Hips",       "LUpperLeg", "RUpperLeg", "LLowerLeg", "RLowerLeg",
	    "LFoot",      "RFoot",     "Chest",     "Head",      "LUpperArm",
	    "RUpperArm",  "LLowerArm", "RLowerArm", "LHand",     "RHand"};
	printf("  per-bone TIC OME:\n");
	for (int s = 0; s < nsens && s < TIC_NSENS; s++) {
		printf("    %-10s %6.2f°\n", bone[s], ome_bone[s] / evaluated);
	}
	return 0;
}
