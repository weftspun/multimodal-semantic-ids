"""TIC trainer — two modes, one file.

Default (Stage 1): lift the reference-free TIC net toward its ~15deg floor on SYNTHETIC continuous
normal motion — geodesic loss, ACCA, and robust temporal aggregation over the constant per-session
mount.  Compares MSE-on-6D vs geodesic loss; single-window vs aggregated (the mount is constant per
session, so a robust SO(3) mean over W windows cuts variance ~sqrt(W)).
  pixi run -e gpu python slangtrain/torch/train_tic.py [STEPS]

`real` mode: train the SAME net on REAL AddBiomechanics normal-motion windows (b3d_to_caldata.py),
with an honest subject-level split, and report the in-sample FLOOR (train==eval, no generalization)
alongside held-out recovery + a per-bone breakdown, then save netfwd-compatible weights.  Confirms on
real data what Stage 1 shows on synthetic: the ~8deg per-window floor, and that the held-out gap is
distribution match (cross-study domain shift), not capacity.  SINEW_ACCEL=raw|zero|norm ablates accel.
  pixi run -e gpu python slangtrain/torch/train_tic.py real <parquet> [--test <p>] [D H Fd STACK STEPS BATCH LR]
Paths resolve from env: SINEW_TIC_REF (reference TIC dir), SINEW_CALDATA[_TEST] (parquets).
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.environ.get("SINEW_TIC_REF", "/mnt/e/tmp/tic-calib"))
from my_model import TIC  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from calib_rank import aa_to_R, build_frame, target_dir, geo_deg  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)
S, NSENS = 32, 15
NIN, NOUT = NSENS * 12, NSENS * 6
D, H, Fd, STACK = 64, 4, 128, 2
G = np.array([0.0, 9.81, 0.0])
OFFSET_DEG, DRIFT_DEG = 30, 40
BATCH = 64
BONE = ["Hips", "LLeg", "RLeg", "LShin", "RShin", "LFoot", "RFoot", "Chest", "Head",
        "LArm", "RArm", "LForeA", "RForeA", "LHand", "RHand"]   # 15 sensors (calib_rank order)


def r6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def walk(base, n, step_deg=15.0):  # continuous normal motion: cumulative 3-axis small rotations
    C, R = [], base.copy()
    axes = [np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), np.array([0.0, 0, 1])]
    for _ in range(n):
        ax = axes[rng.integers(3)]
        R = aa_to_R(ax * np.radians(step_deg) * (rng.random() * 2 - 1)) @ R
        C.append(R.copy())
    return np.stack(C)


def gen_window(mount=None, drift=None):
    mount_basis = mount if mount is not None else np.stack(
        [aa_to_R((rng.random(3) - 0.5) * 2 * np.radians(OFFSET_DEG)) for _ in range(NSENS)])
    if drift is None:
        yaw = (rng.random(NSENS) - 0.5) * 2 * np.radians(DRIFT_DEG)
        tilt = (rng.random((NSENS, 2)) - 0.5) * 2 * np.radians(10)
        drift_basis = np.stack([aa_to_R(np.array([tilt[s, 0], yaw[s], tilt[s, 1]])) for s in range(NSENS)])
    else:
        drift_basis = drift
    x = np.zeros((S, NSENS, 12), np.float32)
    for s in range(NSENS):
        C = walk(build_frame(target_dir(0, s)), S)
        cr = np.einsum("ij,fjk,kl->fil", drift_basis[s], C, mount_basis[s])  # drift · clean · mount
        x[:, s, :3] = (drift_basis[s] @ G)[None, :]                          # ACCA: global accel = drift · gravity
        x[:, s, 3:] = cr.reshape(S, 9)
    y = np.concatenate([np.concatenate([r6d(drift_basis[s]) for s in range(NSENS)]),
                        np.concatenate([r6d(mount_basis[s]) for s in range(NSENS)])])
    return x.reshape(S, NIN).astype(np.float32), y.astype(np.float32), drift_basis, mount_basis


def decode6d(v):  # (...,6)->(...,3,3) Gram-Schmidt, torch, batched
    a = v[..., :3] / (v[..., :3].norm(dim=-1, keepdim=True) + 1e-9)
    b = v[..., 3:] - (a * v[..., 3:]).sum(-1, keepdim=True) * a
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], dim=-1)


def geo_t(Ra, Rb):  # geodesic angle (deg), batched
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(torch.clamp((tr - 1) / 2, -1, 1)))


def chordal_loss(pred6, tgt6):  # geodesic-aligned: ||R_pred - R_tgt||_F^2 on decoded rotations
    Rp = decode6d(pred6.view(-1, NSENS, 6))
    Rt = decode6d(tgt6.view(-1, NSENS, 6))
    return ((Rp - Rt) ** 2).sum(dim=(-1, -2)).mean()


def make_set(n):
    xs, ys = [], []
    for _ in range(n):
        x, y, _, _ = gen_window()
        xs.append(x); ys.append(y)
    return (torch.tensor(np.stack(xs), device=dev), torch.tensor(np.stack(ys), device=dev))


def train(loss_kind, steps):
    model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xtr, Ytr = make_set(2000)
    for step in range(steps):
        idx = torch.randint(0, len(Xtr), (BATCH,), device=dev)
        g, l = model(Xtr[idx])
        out = torch.cat([g, l], dim=1)
        tgt = Ytr[idx]
        if loss_kind == "mse":
            loss = ((out - tgt) ** 2).mean()
        else:
            loss = chordal_loss(out[:, :NOUT], tgt[:, :NOUT]) + chordal_loss(out[:, NOUT:], tgt[:, NOUT:])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def eval_single(model, ntest=200):
    errs = []
    with torch.no_grad():
        for _ in range(ntest):
            x, _, drift_basis, mount_basis = gen_window()
            g, l = model(torch.tensor(x[None], device=dev))
            Bp = decode6d(l.view(NSENS, 6))
            errs.append(geo_t(Bp, torch.tensor(np.stack(mount_basis), device=dev, dtype=torch.float32)).cpu().numpy())
    return float(np.mean(errs))


def chordal_mean(Rs):  # robust SO(3) mean: average then project to SO(3) (bounded, no explosion)
    M = Rs.mean(dim=0)
    U, _, Vt = torch.linalg.svd(M)
    d = torch.sign(torch.linalg.det(U @ Vt))
    return U @ torch.diag(torch.tensor([1.0, 1.0, d.item()], device=Rs.device)) @ Vt


def eval_aggregated(model, W, nsess=60):
    errs = []
    with torch.no_grad():
        for _ in range(nsess):
            mount = np.stack([aa_to_R((rng.random(3) - 0.5) * 2 * np.radians(OFFSET_DEG)) for _ in range(NSENS)])
            yaw = (rng.random(NSENS) - 0.5) * 2 * np.radians(DRIFT_DEG)
            tilt = (rng.random((NSENS, 2)) - 0.5) * 2 * np.radians(10)
            drift = np.stack([aa_to_R(np.array([tilt[s, 0], yaw[s], tilt[s, 1]])) for s in range(NSENS)])
            est = [[] for _ in range(NSENS)]
            for _ in range(W):
                x, _, _, _ = gen_window(mount=mount, drift=drift)
                _, l = model(torch.tensor(x[None], device=dev))
                Bp = decode6d(l.view(NSENS, 6))
                for s in range(NSENS):
                    est[s].append(Bp[s])
            for s in range(NSENS):
                Bm = chordal_mean(torch.stack(est[s]))
                errs.append(geo_t(Bm, torch.tensor(mount[s], device=dev, dtype=torch.float32)).item())
    return float(np.mean(errs))


def synthetic_main(steps):
    print(f"Stage 1: TIC on continuous normal motion (ACCA on)   steps={steps}")
    for kind in ["mse", "geodesic"]:
        m = train(kind, steps)
        s1 = eval_single(m)
        print(f"\n  loss={kind:9s}  single-window mount = {s1:5.1f}deg")
        for W in [1, 4, 16, 64]:
            print(f"      aggregated over W={W:3d} windows: {eval_aggregated(m, W):5.1f}deg")


# ── Real-data mode: train on AddBiomechanics parquet, report the in-sample FLOOR + held-out + accel ──
def decode6d_np(v6):  # 6D -> R (Gram-Schmidt), columns [a, b, a×b]  (matches real_recover.py)
    a = v6[:3] / (np.linalg.norm(v6[:3]) + 1e-9)
    b = v6[3:] - a * (a @ v6[3:])
    b /= (np.linalg.norm(b) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=1)


UP = np.array([0.0, 1.0, 0.0])     # world up = gravity axis (b3d_to_caldata world: up=+Y)


def swing_twist_up(R):
    """Swing-twist of R about world up ŷ, matrix/6D form (no quaternion — keeps the continuous 6D rep
    the pipeline uses): R = swing · twist, with twist a yaw about ŷ and swing the gravity tilt.  Since
    twist·ŷ = ŷ, swing carries ŷ → R·ŷ by the shortest arc; twist = swingᵀ·R.  Returns (swing, twist)."""
    v = R @ UP
    axis = np.cross(UP, v)
    s = np.linalg.norm(axis)
    c = float(np.clip(UP @ v, -1.0, 1.0))
    if s < 1e-8:                                  # ŷ already (anti)aligned with R·ŷ
        swing = np.eye(3) if c > 0 else aa_to_R(np.array([np.pi, 0.0, 0.0]))   # 180° tilt: pick X axis
    else:
        swing = aa_to_R(axis / s * np.arccos(c))
    return swing, swing.T @ R                     # twist = swingᵀ·R (a pure yaw about ŷ)


def drift_tilt(Y6, nsens, nout):
    # Drift R_DG is yaw-dominant, but gravity (up=+Y) is yaw-symmetric, so per-window drift YAW is
    # unobservable; the net would memorise subject facing to guess it (the dominant domain-shift
    # carrier, domain_probe rot_abs 0.70).  Replace each window's drift 6D with its SWING (gravity tilt)
    # — the observable part — and drop the twist (yaw): TRIAD/mag owns world yaw at runtime.
    Yt = Y6.copy()
    for ti in range(len(Y6)):
        for s in range(nsens):
            Yt[ti][s * 6:s * 6 + 6] = r6d(swing_twist_up(decode6d_np(Y6[ti][s * 6:s * 6 + 6]))[0])
    return Yt


def real_main(argv):
    import pyarrow.parquet as pq
    from pack_weights import pack_winit  # slangtrain/ already on sys.path (calib_rank import above)

    here = os.path.dirname(os.path.abspath(__file__))
    test_parquet = os.environ.get("SINEW_CALDATA_TEST")
    if "--test" in argv:
        i = argv.index("--test"); test_parquet = argv[i + 1]; del argv[i:i + 2]
    parquet = argv[0] if argv else os.environ.get("SINEW_CALDATA", "/mnt/e/tmp/caldata_addb.parquet")
    TEST_SUBJECTS = {"Subject5", "subject_3"}      # within-parquet fallback split (unseen people)
    dflt = [64, 4, 128, 2, 3000, 256, 0.003]       # d h fd stack steps batch lr
    a = argv[1:8]
    d, h, fd, stack, steps, batch = (int(a[i]) if i < len(a) else dflt[i] for i in range(6))
    lr = float(a[6]) if len(a) > 6 else dflt[6]
    EVAL_EVERY = 250
    IDENTITY_FLOOR = 28.6                           # held A/T/S/B identity baseline ("doing nothing")
    # Default = grav: gravity-normalize accel.  Cross-study held-out (tr24/te10): raw 43.8°, zero 35.5°,
    # norm 36.4°, grav 36.0° — grav recovers the ~8° raw accel-magnitude costs AND keeps the best floor
    # of the accel-retaining modes (9.1° vs zero's 10.9°), since the direction cue survives.
    accel_mode = os.environ.get("SINEW_ACCEL", "grav")
    # SINEW_DRIFT=full|tilt — tilt replaces the drift target R_DG with its gravity-observable swing
    # (yaw dropped), so the net is not asked to estimate the gravity-unobservable yaw and has no reason
    # to memorise subject facing (the dominant domain-shift carrier).  Mount R_BS is unchanged.
    drift_mode = os.environ.get("SINEW_DRIFT", "full")
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device — running on CPU (use `pixi run -e gpu`).")

    def load(path):
        t = pq.read_table(path)
        md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
        s, n, o = int(md["S"]), int(md["NIN"]), int(md["NOUT"])
        # Read the list<float32> columns straight from Arrow's flat value buffer (no Python-level
        # iteration): to_pydict + np.array(list-of-lists) walks every element in Python and dominates
        # the run (minutes on the 38k-window mag set); the flat-buffer reshape is ~65x faster.
        x = t["x"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, s, n).astype(np.float32)
        y = t["y"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)
        return np.array(t["subject"].to_pylist()), x, y, s, n, o

    sub, Xn, Yn, s_, nin, nout = load(parquet)
    nsens = nout // 6
    if drift_mode == "tilt":
        Yn = drift_tilt(Yn, nsens, nout)

    per = nin // nsens  # input channels per sensor: 12 = [accel(3), rot(9)], 15 = +emulated mag(3)

    nomag = per == 15 and os.environ.get("SINEW_NOMAG") == "1"  # A/B: zero the mag channel on mag data
    if nomag:
        print("ablating magnetometer channel (SINEW_NOMAG=1): mag-off baseline on the mag dataset")

    def accel_xform(Z):  # per-sensor [accel(3), (mag(3),) rot(9)]; raw|zero|norm|grav on the accel channel
        if accel_mode == "raw" and not nomag:
            return Z
        z = Z.reshape(Z.shape[0], Z.shape[1], nsens, per).copy()  # mag/rot pass through untouched
        if nomag:
            z[..., 3:6] = 0.0                          # ablate emulated magnetometer (drift-yaw cue)
        if accel_mode == "zero":
            z[..., :3] = 0.0
        elif accel_mode == "norm":                 # standardize accel over the S frames of each window
            aa = z[..., :3]
            z[..., :3] = (aa - aa.mean(1, keepdims=True)) / (aa.std(1, keepdims=True) + 1e-6)
        elif accel_mode == "grav":                 # gravity normalization: unit-length per frame/sensor
            # The ACCA cue is the GRAVITY DIRECTION rotated by drift; |g - p̈| carries the study-
            # dependent linear-accel magnitude AND local-g variation (9.7803 equator..9.8322 poles by
            # study location, references.bib moritz1980grs80) — both cross-study carriers.  Unit-
            # normalizing keeps the direction cue and drops the magnitude — lossless for synthetic
            # (pure gravity, |a|=9.81).
            aa = z[..., :3]
            z[..., :3] = aa / (np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-6)
        else:
            sys.exit(f"unknown SINEW_ACCEL={accel_mode} (raw|zero|norm|grav)")
        return z.reshape(Z.shape)

    Xn = accel_xform(Xn)
    print(f"train parquet {parquet}  [accel={accel_mode}  drift={drift_mode}]")
    if test_parquet:                               # separate held-out file (addb train/test dir split)
        teSub, teX, teY, s2, _, _ = load(test_parquet)
        teX = accel_xform(teX)
        if drift_mode == "tilt":
            teY = drift_tilt(teY, nsens, nout)
        assert s2 == s_, "S mismatch between train and test parquet"
        overlap = set(sub) & set(teSub)
        assert not overlap, f"subject leak across train/test parquets: {sorted(overlap)[:8]}"
        trX, trY, tr_sub = Xn, Yn, sub
        print(f"test  parquet {test_parquet}")
        print(f"split: train {len(trX)} windows / {len(set(sub))} subjects, "
              f"test {len(teX)} windows / {len(set(teSub))} subjects (separate files)")
    else:                                          # within-parquet subject-level split
        allsubs = sorted(set(sub))
        test_set = TEST_SUBJECTS & set(allsubs) or set(allsubs[::6])   # auto: hold out ~1/6 of subjects
        is_test = np.array([x in test_set for x in sub])
        trX, trY, teX, teY = Xn[~is_test], Yn[~is_test], Xn[is_test], Yn[is_test]
        tr_sub = sub[~is_test]
        print(f"subject-level split: train {len(trX)} windows / {len(allsubs) - len(test_set)} subjects, "
              f"test {len(teX)} windows / {len(test_set)} subjects {sorted(test_set)[:6]}")
    ntrain, ntest = len(trX), len(teX)
    if ntrain == 0 or ntest == 0:
        sys.exit("empty split — check the parquet(s) / TEST_SUBJECTS")
    print(f"config D={d} H={h} Fd={fd} STACK={stack} STEPS={steps} BATCH={batch} LR={lr}  "
          f"S={s_} NIN={nin} NOUT={nout}  device={device.type}")

    trX_t = torch.from_numpy(trX).to(device)
    trY_t = torch.from_numpy(trY).to(device)
    teX_t = torch.from_numpy(teX).to(device)
    # in-sample floor: same windows trained and evaluated, so no generalization gap — the residual is
    # pure per-window observability + capacity, the limit held-out cannot beat.
    flo_idx = np.arange(ntrain) if ntrain <= 1024 else np.random.RandomState(0).choice(ntrain, 1024, False)
    flX_t = trX_t[torch.as_tensor(flo_idx, device=device)]
    flY = trY[flo_idx]

    model = TIC(stack=stack, n_input=nin, n_output=nout, multi_head=h, d_model=d, d_ff=fd).to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    def evaluate(Xt, Yv):  # recovered-vs-true geodesic deg for R_BS (offset) and R_DG (drift)
        model.eval()
        with torch.no_grad():
            g, l = model(Xt); pred = torch.cat([g, l], dim=1).cpu().numpy()
        model.train()
        oe, de, ob, db = [], [], [], []
        for ti in range(len(Yv)):
            for s in range(nsens):
                Rop, Rdp = decode6d_np(pred[ti][nout + s * 6:nout + s * 6 + 6]), decode6d_np(pred[ti][s * 6:s * 6 + 6])
                Rot, Rdt = decode6d_np(Yv[ti][nout + s * 6:nout + s * 6 + 6]), decode6d_np(Yv[ti][s * 6:s * 6 + 6])
                oe.append(geo_deg(Rop, Rot)); de.append(geo_deg(Rdp, Rdt))
                ob.append(geo_deg(np.eye(3), Rot)); db.append(geo_deg(np.eye(3), Rdt))
        return np.mean(oe), np.median(oe), np.mean(ob), np.mean(de), np.median(de), np.mean(db)

    def floor_per_bone(Xt, Yv):  # in-sample R_BS error per sensor — which bones floor high vs low
        model.eval()
        with torch.no_grad():
            pred = model(Xt)[1].cpu().numpy()      # local head = offset R_BS
        model.train()
        per = np.zeros(nsens)
        for s in range(nsens):
            per[s] = np.mean([geo_deg(decode6d_np(pred[ti][s * 6:s * 6 + 6]),
                                      decode6d_np(Yv[ti][nout + s * 6:nout + s * 6 + 6])) for ti in range(len(Yv))])
        return per

    def pose_ome(Xt, Yv):  # downstream POSE error (transformerimucalib2025 OME): the per-bone
        # orientation error of the calibration-recovered clean bone vs the true clean bone.  Applying
        # the predicted calibration to the sensor gives recovered = R_DG_predᵀ·R_sensor·R_BS_predᵀ; the
        # truth is R_DG_trueᵀ·R_sensor·R_BS_trueᵀ.  Their geodesic is how the R_BS/R_DG head errors
        # propagate into the reconstructed pose — the metric the paper reports at 15.2° average.
        # Also reports the ROOT-RELATIVE OME (each bone relative to the pelvis, sensor 0) — the
        # deployment-relevant pose error, since the runtime roots the body at the hips and re-anchors
        # the global orientation from the HMD/foot-lock, so the pelvis's own mount error is corrected.
        model.eval()
        with torch.no_grad():
            g, l = model(Xt); pred = torch.cat([g, l], dim=1).cpu().numpy()
        model.train()
        cps = nin // nsens                                 # channels/sensor; rot9 is the last 9
        Xmid = Xt[:, Xt.shape[1] // 2].cpu().numpy()       # window middle frame: (N, NIN), the live pose
        ome = np.zeros(nsens); omer = np.zeros(nsens)
        for ti in range(len(Yv)):
            rec, tru = [], []
            for s in range(nsens):
                Rs = Xmid[ti, s * cps + cps - 9: s * cps + cps].reshape(3, 3)  # R_sensor (corrupted bone)
                Rdp, Rop = decode6d_np(pred[ti][s * 6:s * 6 + 6]), decode6d_np(pred[ti][nout + s * 6:nout + s * 6 + 6])
                Rdt, Rot = decode6d_np(Yv[ti][s * 6:s * 6 + 6]), decode6d_np(Yv[ti][nout + s * 6:nout + s * 6 + 6])
                rec.append(Rdp.T @ Rs @ Rop.T); tru.append(Rdt.T @ Rs @ Rot.T)  # recovered / true clean bone
            for s in range(nsens):
                ome[s] += geo_deg(rec[s], tru[s])                          # absolute world orientation
                omer[s] += geo_deg(rec[0].T @ rec[s], tru[0].T @ tru[s])   # relative to the pelvis (root)
        return ome / len(Yv), omer / len(Yv)

    # SINEW_WEIGHTS=pheno_weights.json (from pheno_equity.py): draw each train window with prob ∝ its
    # subject's equity weight, so training hits the representation floors (sex 50/50, hardest BMI band
    # maximised) instead of the convenience distribution — the torch analog of sample_caldata.py.
    samp_prob = None
    if os.environ.get("SINEW_WEIGHTS"):
        import json
        wmap = {x["subject"]: x["weight"] for x in json.load(open(os.environ["SINEW_WEIGHTS"]))}
        w = np.array([wmap.get(x, 0.0) for x in tr_sub], dtype=np.float64)
        if w.sum() == 0:
            sys.exit("SINEW_WEIGHTS: no train window's subject is in the weights file")
        samp_prob = torch.tensor(w / w.sum(), device=device)
        print(f"equity-weighted sampling [{os.environ['SINEW_WEIGHTS']}]: "
              f"{len(set(tr_sub[w > 0]))} weighted subjects, {int((w == 0).sum())} windows zero-weighted")

    print("\ntraining on real normal-motion windows (GPU PyTorch)...")
    model.train()
    loss0 = None
    for step in range(1, steps + 1):
        idx = (torch.multinomial(samp_prob, min(batch, ntrain), replacement=True)
               if samp_prob is not None else torch.randint(0, ntrain, (min(batch, ntrain),), device=device))
        g, l = model(trX_t[idx])
        loss = mse(torch.cat([g, l], dim=1), trY_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        loss0 = loss.item() if loss0 is None else loss0
        if step % EVAL_EVERY == 0 or step == steps:
            fm, fmed = evaluate(flX_t, flY)[:2]
            offm, offmed, offb, dgm, dgmed, dgb = evaluate(teX_t, teY)
            print(f"step {step:5d}  loss {loss.item():.4f}   floor mount {fm:5.1f}°/{fmed:5.1f}°   "
                  f"held-out mount {offm:5.1f}°/{offmed:5.1f}° (id {offb:.1f}°)   "
                  f"drift {dgm:5.1f}°/{dgmed:5.1f}° (id {dgb:.1f}°)")

    print(f"\ntrain loss {loss0:.4f} -> {loss.item():.4f}")
    fm, fmed = evaluate(flX_t, flY)[:2]
    offm, offmed, offb, dgm, dgmed, dgb = evaluate(teX_t, teY)
    pack_winit(model.state_dict(), stack).tofile(os.path.join(here, "wtrained.bin"))
    torch.save(model.state_dict(), os.path.join(here, "tic_trained.pth"))
    print("\nsaved wtrained.bin (netfwd layout) + tic_trained.pth")
    per = floor_per_bone(flX_t, flY)
    order = np.argsort(per)
    print(f"\nFLOOR (in-sample, {len(flY)} train windows — no generalization gap):")
    print(f"  mount : {fm:5.1f}° mean / {fmed:5.1f}° median   <- best the net can do on this data")
    print("  per-bone (best→worst): " + "  ".join(f"{BONE[s]}:{per[s]:.0f}°" for s in order))
    print(f"HELD-OUT RECOVERY ({ntest} unseen-subject windows, {nsens} sensors each):")
    print(f"  mount : {offm:5.1f}° mean / {offmed:5.1f}° median   (identity baseline {offb:.1f}°)")
    print(f"  drift : {dgm:5.1f}° mean / {dgmed:5.1f}° median   (identity baseline {dgb:.1f}°)")
    ome, omer = pose_ome(teX_t, teY)
    oorder = np.argsort(ome)
    print(f"  pose OME    : {ome.mean():5.1f}° mean / {np.median(ome):5.1f}° median   "
          f"(transformerimucalib2025 OME 15.2°)  — absolute world orientation")
    print("  per-bone (best→worst): " + "  ".join(f"{BONE[s]}:{ome[s]:.0f}°" for s in oorder))
    rel = np.delete(omer, 0)  # the pelvis is the reference (relative-to-self = 0); average the other 14
    print(f"  pose OME root-rel : {rel.mean():5.1f}° mean / {np.median(rel):5.1f}° median   "
          f"— relative to the pelvis (the deployment metric; HMD/foot-lock anchor the global frame)")
    print(f"\nfloor {fm:.1f}°  |  generalization gap {offm - fm:+.1f}°  |  held-out {offm:.1f}°  "
          f"(identity {offb:.1f}°, static A/T/S/B ~{IDENTITY_FLOOR}°)")
    print("read: the floor is the per-window observability limit (more steps/capacity); the gap is what "
          "more IN-DISTRIBUTION subjects close. Cross-study held-out (different markers/motion/units) can "
          "exceed identity — that is domain shift, not capacity. Held-out cannot beat the floor.")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "real":
        real_main(a[1:])
    else:
        synthetic_main(int(a[0]) if a else 4000)
