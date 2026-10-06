import numpy as np
import torch
import mdtraj
from sklearn.decomposition import PCA
import scipy.stats
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import time
import shutil
import random
import py3Dmol
import plotly.graph_objects as go
import wandb
import MDAnalysis as mda
import os

from cryoflow import paths


def log_py3Dmol_to_wandb(view, name='structure_visualization'):
    temp_html_path = Path("temp_protein_view.html")
    # Write HTML content using a file handle
    with temp_html_path.open('w', encoding='utf-8') as f:
        view.write_html(f, fullpage=True)

    # Read the HTML content
    with temp_html_path.open('r', encoding='utf-8') as f:
        html_content = f.read()

    # Log to W&B as HTML
    wandb.log({
        name: wandb.Html(html_content)
    })

    # Clean up temporary file
    temp_html_path.unlink()

def pca_plot(af_coords_joint_pca, ref_coords_joint_pca, seed_coords_joint_pca, label):
    return _embedding_plot(af_coords_joint_pca, ref_coords_joint_pca, seed_coords_joint_pca, label, method='PCA')

def _embedding_plot(af, ref, seed, label, method='PCA'):
    trace_af = go.Scatter(x=af[:, 0], y=af[:, 1], mode='markers', marker=dict(color='red'), name='Generated')
    trace_ref = go.Scatter(x=ref[:, 0], y=ref[:, 1], mode='markers', marker=dict(color='blue'), name='Reference')
    trace_seed = go.Scatter(x=seed[:, 0], y=seed[:, 1], mode='markers',
                            marker=dict(color='green', size=12, symbol='diamond'), name='Seed Coordinate')
    axis = {'PCA': 'PC', 't-SNE': 't-SNE', 'UMAP': 'UMAP'}.get(method, method)
    fig = go.Figure(data=[trace_ref, trace_af, trace_seed])
    fig.update_layout(
        title=f'{method} trained on reference {label} trajectory',
        xaxis_title=f'{axis} 1',
        yaxis_title=f'{axis} 2',
    )
    return fig

def visualize_structures(model, dataset, pdb_file, save_dir="saved_pdbs", device=None, N=20):
    print("Generating a model sample...")
    model.eval()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    with torch.no_grad():
        sample = model.inference(batch_size=1, n_residues=model.n_residue, N=N)[0]
        sample = dataset.reverse_transform(sample)
    
    print("Computing RMSD with ground truth samples...")
    
    sample = sample.to(device)
    
    # Load all ground truth samples
    ground_truth_samples = []
    for i in range(len(dataset)):
        gt_sample,_ = dataset[i]
        gt_sample = gt_sample.to(device)
        gt_sample = dataset.reverse_transform(gt_sample).to(device)
        ground_truth_samples.append(gt_sample)
    
    # Compute RMSD with all ground truth samples
    rmsd_values = []
    for gt_sample in ground_truth_samples:
        aligned_sample = rmsdalign(gt_sample, sample)
        rmsd = torch.sqrt(torch.mean(torch.sum((aligned_sample - gt_sample) ** 2, dim=-1))).item()
        rmsd_values.append(rmsd)
    
    # Find the most similar ground truth sample
    best_match_idx = int(torch.argmin(torch.tensor(rmsd_values)))
    best_rmsd_value = rmsd_values[best_match_idx]
    best_match_gt = ground_truth_samples[best_match_idx]
    
    # Align the generated sample with the most similar ground truth
    aligned_sample = rmsdalign(best_match_gt, sample)
    print("Best RMSD value:", best_rmsd_value)
    
    # Convert to numpy for visualization
    gt_coords = best_match_gt.detach().cpu().numpy()
    aligned_coords = aligned_sample.detach().cpu().numpy()
    
    gt_u = mda.Universe(pdb_file)
    gt_backbone = gt_u.select_atoms("backbone")
    
    os.makedirs(save_dir, exist_ok=True)
    gt_backbone_pdb_path = os.path.join(save_dir, f"ground_truth_backbone.pdb")
    gen_backbone_pdb_path = os.path.join(save_dir, f"generated_backbone.pdb")
    
    # Save only backbone atoms
    gt_backbone.positions = gt_coords
    gt_backbone.write(gt_backbone_pdb_path)
    gt_backbone.positions = aligned_coords
    gt_backbone.write(gen_backbone_pdb_path)
    
    print(f"Ground truth backbone PDB saved to: {gt_backbone_pdb_path}")
    print(f"Generated backbone PDB saved to: {gen_backbone_pdb_path}")

    wandb.save(gt_backbone_pdb_path)
    wandb.save(gen_backbone_pdb_path)
    
    # Read PDB content for Py3DMol visualization
    with open(gt_backbone_pdb_path, "r") as f:
        gt_pdb_data = f.read()
    with open(gen_backbone_pdb_path, "r") as f:
        gen_pdb_data = f.read()
    
    # Aligned visualization
    viewer = py3Dmol.view(width=1000, height=500)
    viewer.addLabel(f"RMSD = {best_rmsd_value:.3f} Å, Green-Generated, Gray-GT", {
        "fontSize": 16,
        "fontColor": "black",
        "backgroundColor": "white",
        "position": {"x": 0, "y": 0, "z": 0}
    })
    viewer.addModel(gt_pdb_data, "pdb")
    viewer.addModel(gen_pdb_data, "pdb")
    viewer.setStyle({'model': 0}, {"cartoon": {"color": "lightgrey"}})
    viewer.setStyle({'model': 1}, {"cartoon": {"color": "lightgreen"}})
    viewer.zoomTo()
    log_py3Dmol_to_wandb(viewer, "structure_comparison")
    
    # Side-by-side visualization
    viewer_side_by_side = py3Dmol.view(width=1000, height=500, viewergrid=(1, 2))
    viewer_side_by_side.addModel(gt_pdb_data, "pdb", viewer=(0, 0))
    viewer_side_by_side.addModel(gen_pdb_data, "pdb", viewer=(0, 1))
    viewer_side_by_side.setStyle({"cartoon": {"color": "lightgrey"}}, viewer=(0, 0))
    viewer_side_by_side.setStyle({"cartoon": {"color": "lightgreen"}}, viewer=(0, 1))
    viewer_side_by_side.zoomTo()

    log_py3Dmol_to_wandb(viewer_side_by_side, "structure_side_by_side")
    
    return best_rmsd_value

def wait_for_pymol_frames(frame_dir: Path, expected_frame_count: int, timeout: float = 100, poll_interval: float = 0.5):
    start = time.time()
    while True:
        frame_files = sorted(frame_dir.glob("frame*.png"))
        if len(frame_files) >= expected_frame_count:
            try:
                with open(frame_files[-1], 'rb') as f:
                    f.read(10)
                print(f"All {len(frame_files)} frames saved and ready.")
                return frame_files
            except (PermissionError, OSError):
                pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for {expected_frame_count} frames.")
        time.sleep(poll_interval)

def save_gif(gt_pdb, gt_xtc, gen_pdb, gen_xtc, gif_path=None, fps=5, max_frames=500, frame_dir=None):
    gif_path = gif_path or os.path.join(paths.ANALYSIS, "saved_gifs", "comparison.gif")
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    # Create a temporary directory for PNG frames
    frame_dir = Path(frame_dir or "temp_frames").resolve()
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Start PyMOL instance
    from pymolPy3 import pymolPy3
    pm = pymolPy3(1) # change to 1 for head

    # Load structures and trajectories
    pm(f'load {gt_pdb}, gt')
    pm(f'load_traj {gt_xtc}, gt, state=1')

    pm(f'load {gen_pdb}, gen')
    pm(f'load_traj {gen_xtc}, gen, state=1')

    pm("viewport 1280, 960")

    pm('translate [40, 0, 0], gen, state=0') # generated on the right
    pm('zoom all, buffer=5')
    pm(f'set movie_fps, {fps}')

    traj = mdtraj.load_xtc(gt_xtc, top=gt_pdb)
    expected_frames = min(traj.n_frames, max_frames)

    pm(f'mpng {frame_dir / "frame"}, 1, {expected_frames}')
    pm('quit')

    # Wait until frames are ready
    frame_files = wait_for_pymol_frames(frame_dir, expected_frame_count=expected_frames)

    # Create GIF
    gif_path = Path(gif_path).resolve()
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    if gif_path.exists():
        gif_path.unlink()

    # Load images
    images = [Image.open(f) for f in frame_files]

    # Save new GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:], 
        duration=int(1000 / fps), 
        loop=0
    )

    # Clean up temporary frames
    shutil.rmtree(frame_dir)
    
    return gif_path


def get_pca(xyz):
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords


def get_tsne_joint(ref_xyz, gen_xyz, seed_xyz, perplexity=50, random_state=0):
    from sklearn.manifold import TSNE
    ref_f = ref_xyz.reshape(ref_xyz.shape[0], -1)
    gen_f = gen_xyz.reshape(gen_xyz.shape[0], -1)
    seed_f = seed_xyz.reshape(seed_xyz.shape[0], -1)
    X = np.concatenate([ref_f, gen_f, seed_f], axis=0)
    perp = min(perplexity, max(5, X.shape[0] // 3 - 1))
    emb = TSNE(n_components=2, perplexity=perp, random_state=random_state, init='pca').fit_transform(X)
    n_ref, n_gen = ref_f.shape[0], gen_f.shape[0]
    return emb[n_ref:n_ref + n_gen], emb[:n_ref], emb[n_ref + n_gen:]


def get_umap_joint(ref_xyz, gen_xyz, seed_xyz, n_neighbors=15, random_state=0):
    import umap
    ref_f = ref_xyz.reshape(ref_xyz.shape[0], -1)
    gen_f = gen_xyz.reshape(gen_xyz.shape[0], -1)
    seed_f = seed_xyz.reshape(seed_xyz.shape[0], -1)
    nn = min(n_neighbors, max(2, ref_f.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=nn, random_state=random_state)
    reducer.fit(ref_f)
    return reducer.transform(gen_f), reducer.transform(ref_f), reducer.transform(seed_f)

def get_rmsds(traj1, traj2, broadcast=False, chunk_size=128):
    n_atoms = traj1.shape[1]
    flat1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    flat2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if not broadcast:
        return np.sqrt(np.square(flat1 - flat2).sum(-1)) / np.sqrt(n_atoms) * 10

    M, N = flat1.shape[0], flat2.shape[0]
    out = np.empty((M, N), dtype=np.float32)
    sq2 = np.square(flat2).sum(-1)
    for i in range(0, M, chunk_size):
        a = flat1[i:i + chunk_size]
        sq1 = np.square(a).sum(-1)[:, None]
        cross = a @ flat2.T
        sq = np.maximum(sq1 + sq2[None] - 2.0 * cross, 0.0)
        out[i:i + chunk_size] = np.sqrt(sq) / np.sqrt(n_atoms) * 10
    return out

def get_mean_covar(xyz):
    mean = xyz.mean(0)
    xyz_centered = xyz - mean
    covar = np.mean(xyz_centered[..., None] * xyz_centered[..., None, :], axis=0)
    return mean, covar

def sqrtm(M):
    D, P = np.linalg.eig(M)
    # PSD-similar inputs -> real sqrt; drop numerical imaginary noise so downstream
    # metrics (emd_var/RMWD) stay JSON/wandb serializable.
    return ((P * np.sqrt(D[:, None])) @ np.linalg.inv(P)).real

def align_tops(top1, top2):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]
    intersection = [name for name in names1 if name in names2]
    mask1 = [names1.index(name) for name in intersection]
    mask2 = [names2.index(name) for name in intersection]
    return mask1, mask2

def generate_samples(model, dataset, num_samples=1000, device=None, batch_size=100, N=20, sample_scaler=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()

    reverse = sample_scaler.reverse_transform if sample_scaler is not None else dataset.reverse_transform

    generated_samples = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            batch_samples = model.inference(batch_size=current_batch, n_residues=model.n_residue, N=N)
            batch_samples = reverse(batch_samples).to(device)
            generated_samples.append(batch_samples)
    
    generated_samples = torch.cat(generated_samples, dim=0)
    return generated_samples

def generate_guided_samples(model, dataset, num_samples=1000, device=None, n_candidates=5, N=10, warmup_steps=3, guidance_strength=0.01, batch_size=128):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    n_frames = len(dataset)
    n_residues = model.flow_model.n_residue

    samples_per_frame = max(1, num_samples // n_frames)
    frame_idx_list = []
    for f in range(n_frames):
        frame_idx_list.extend([f] * samples_per_frame)
    frame_idx_list = frame_idx_list[:num_samples]

    use_defocus = getattr(model, "_use_defocus", False) and hasattr(dataset, "defocus")
    frame_gt_imgs = {}
    frame_angles_map = {}
    frame_defocus_map = {}
    for f in range(n_frames):
        gt_images, _, angles, _ = dataset[f]
        frame_gt_imgs[f] = gt_images[:1].to(device)
        frame_angles_map[f] = angles[:1].to(device)
        if use_defocus:
            frame_defocus_map[f] = dataset.defocus[f, :1]

    all_samples = []
    for start in tqdm(range(0, len(frame_idx_list), batch_size), desc="Generating guided samples"):
        chunk = frame_idx_list[start:start + batch_size]
        B = len(chunk)
        gt_batch = torch.cat([frame_gt_imgs[f] for f in chunk], dim=0)
        angles_batch = torch.cat([frame_angles_map[f] for f in chunk], dim=0)
        defocus_batch = (torch.cat([frame_defocus_map[f] for f in chunk], dim=0)
                         if use_defocus else None)

        with torch.no_grad():
            candidates = model.flow_model.inference(
                batch_size=B * n_candidates, n_residues=n_residues, N=warmup_steps
            ).reshape(B, n_candidates, n_residues, 3)

            flat_cands = candidates.reshape(B * n_candidates, n_residues, 3)
            flat_cands_denorm = model.coords_scaler.reverse_transform(flat_cands)
            angles_expanded = angles_batch.unsqueeze(1).expand(-1, n_candidates, -1).reshape(B * n_candidates, 3)
            defocus_expanded = (None if defocus_batch is None
                                else defocus_batch.repeat_interleave(n_candidates))
            pred_imgs = model.renderer.render_particle_batch(flat_cands_denorm, angles_expanded, sigma=model.sigma, defocus=defocus_expanded)
            pred_imgs = pred_imgs.reshape(B, n_candidates, *pred_imgs.shape[-2:])

            gt_exp = gt_batch.unsqueeze(1).expand(-1, n_candidates, -1, -1)
            losses = torch.nn.functional.l1_loss(pred_imgs, gt_exp, reduction='none').mean(dim=(-2, -1))

            best_idx = losses.argmin(dim=1)
            best_candidates = candidates[torch.arange(B, device=device), best_idx]

        guided = model.guided_sampling_batched(
            best_candidates, gt_batch, angles_batch,
            N=N, warmup_steps=0, guidance_strength=guidance_strength,
            defocus=defocus_batch
        )
        guided_denorm = model.coords_scaler.reverse_transform(guided)
        all_samples.append(guided_denorm.detach().to(device))

    return torch.cat(all_samples, dim=0).to(device)

def match_generated_to_md(dataset, generated_samples, device='cuda', total_matches=500, start_frame=0, end_frame=None):
    used_indices = set()
    matched_samples = []
    best_rmsd_values = []

    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.to(device)
    else:
        generated_samples = torch.stack([g.to(device) for g in generated_samples])

    subset = dataset[start_frame:end_frame] if end_frame is not None else dataset
    for gt_sample in tqdm(subset, desc="Matching frames"):
        gt_sample,_ = gt_sample
        gt_sample = dataset.reverse_transform(gt_sample).to(device)
        valid_indices = [i for i in range(len(generated_samples)) if i not in used_indices]
        if not valid_indices:
            break
        valid_idx_tensor = torch.tensor(valid_indices, device=device)
        candidates = generated_samples.index_select(0, valid_idx_tensor)
        gt_expanded = gt_sample.unsqueeze(0).expand_as(candidates)
        aligned = rmsdalign(gt_expanded, candidates)
        rmsd_vec = torch.sqrt(((aligned - gt_expanded) ** 2).sum(-1).mean(-1))
        best_relative_idx = int(rmsd_vec.argmin().item())
        best_idx = valid_indices[best_relative_idx]
        used_indices.add(best_idx)
        matched_samples.append(aligned[best_relative_idx].detach().cpu())
        best_rmsd_values.append(rmsd_vec[best_relative_idx].item())
        if len(matched_samples) >= total_matches:
            break
    return matched_samples, best_rmsd_values

def save_matched_samples_as_trajectory(matched_samples, dataset, output_dir="generated_trajectory", label="start", start_frame=0, end_frame=None):

    os.makedirs(output_dir, exist_ok=True)
    template_u = dataset.universe
    backbone_atoms = template_u.select_atoms("backbone")
    
    gt_pdb_path = os.path.join(output_dir, f"gt_backbone_only_{label}.pdb")
    template_u.trajectory[0]
    backbone_atoms.write(gt_pdb_path)
    
    backbone_u = mda.Universe(gt_pdb_path)
    
    gen_traj_path = os.path.join(output_dir, f"generated_trajectory_{label}.xtc")
    with mda.coordinates.XTC.XTCWriter(gen_traj_path, backbone_atoms.n_atoms) as W:
        for coords in matched_samples:
            backbone_u.atoms.positions = coords.numpy()
            W.write(backbone_u.atoms)
    
    gt_subset = dataset[start_frame:end_frame] if end_frame is not None else dataset
    n_frames = len(matched_samples)
    ground_truth_coords = []
    count = 0
    for gt_sample in gt_subset:
        gt_sample,_ = gt_sample
        gt_sample = dataset.reverse_transform(gt_sample)
        ground_truth_coords.append(gt_sample.cpu())
        count += 1
        if count >= n_frames:
            break
             
    gt_traj_path = os.path.join(output_dir, f"ground_truth_trajectory_{label}.xtc")
    with mda.coordinates.XTC.XTCWriter(gt_traj_path, backbone_atoms.n_atoms) as W:
        for coords in ground_truth_coords:
            backbone_u.atoms.positions = coords.numpy()
            W.write(backbone_u.atoms)
    
    gen_pdb_path = os.path.join(output_dir, f"generated_backbone_only_{label}.pdb")
    backbone_u.atoms.positions = matched_samples[0].numpy()
    backbone_u.atoms.write(gen_pdb_path)
    
    return gt_pdb_path, gen_pdb_path, gen_traj_path, gt_traj_path

def save_paths(gt_pdb_path, gen_pdb_path, gen_traj_path, gt_traj_path):
    wandb.save(os.path.abspath(gt_pdb_path))
    wandb.save(os.path.abspath(gen_pdb_path))
    wandb.save(os.path.abspath(gen_traj_path))
    wandb.save(os.path.abspath(gt_traj_path))

def analyze_trajectories(ref_pdb_path, ref_xtc_path, gen_pdb_path, gen_xtc_path, bb_only=True, ca_only=False, label="", wandb=None):
    # Load reference structure and trajectory
    ref_aa = mdtraj.load(ref_pdb_path)
    traj_aa = mdtraj.load(ref_xtc_path, top=ref_pdb_path)
    
    # Load generated (model prediction) trajectory
    aftraj_aa = mdtraj.load(gen_xtc_path, top=gen_pdb_path)
    
    # Remove hydrogens from all trajectories
    traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.element.symbol != 'H'], True)
    ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.element.symbol != 'H'], True)
    aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.element.symbol != 'H'], True)
    
    # If backbone-only is requested, remove sidechains
    if bb_only:
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
        traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
    
    elif ca_only:
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name == 'CA'], True)

    # Align the topologies between the reference trajectory and the generated trajectory
    refmask, afmask = align_tops(traj_aa.top, aftraj_aa.top)
    traj_aa = traj_aa.atom_slice(refmask, True)
    ref_aa = ref_aa.atom_slice(refmask, True)
    aftraj_aa = aftraj_aa.atom_slice(afmask, True)
    
    # Superpose the trajectories onto the reference structure
    traj_aa.superpose(ref_aa)
    aftraj_aa.superpose(ref_aa)
    
    # For computing pairwise RMSDs, further restrict to CA atoms
    ca_mask = [a.index for a in traj_aa.top.atoms if a.name == 'CA']
    traj = traj_aa.atom_slice(ca_mask, False)
    ref  = ref_aa.atom_slice(ca_mask, False)
    aftraj = aftraj_aa.atom_slice(ca_mask, False)
    
    traj.superpose(ref)
    aftraj.superpose(ref)
    
    # Generate random indices for pairwise RMSD and EMD calculations
    np.random.seed(137)
    RAND1 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND2 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND1K = np.random.randint(0, traj_aa.n_frames, min(1000, traj_aa.n_frames))
    
    results = {}
    ref_pairwise = get_rmsds(traj_aa[RAND1].xyz, traj_aa[RAND2].xyz, broadcast=True)
    pred_pairwise = get_rmsds(aftraj_aa[RAND1].xyz, aftraj_aa[RAND2].xyz, broadcast=True)
    results['ref_mean_pairwise_rmsd'] = ref_pairwise.mean()
    results['predicted_mean_pairwise_rmsd'] = pred_pairwise.mean()
    results['pairwise_rmsd r'] = scipy.stats.pearsonr(
        np.asarray(ref_pairwise).ravel(), np.asarray(pred_pairwise).ravel()
    )[0]

    results['rmsd_between_predicted_and_reference'] = get_rmsds(aftraj_aa.xyz, traj_aa.xyz, broadcast=True).mean()
    
    # Compute Earth Mover's Distance (EMD) metrics using full-atom trajectories (without CA slicing)
    ref_mean, ref_covar = get_mean_covar(traj_aa[RAND1K].xyz)
    af_mean, af_covar = get_mean_covar(aftraj_aa[RAND1K].xyz)
    results['emd_mean'] = np.sqrt(np.square(ref_mean - af_mean).sum(-1)) * 10
    results['emd_mean'] = np.square(results['emd_mean']).mean() ** 0.5
    try:
        results['emd_var'] = np.sqrt(np.trace(ref_covar + af_covar - 2*sqrtm(ref_covar @ af_covar), axis1=1, axis2=2)) * 10
    except Exception:
        results['emd_var'] = np.sqrt(np.trace(ref_covar)) * 10

    results['emd_var'] = np.square(results['emd_var']).mean() ** 0.5

    results['RMWD'] = np.sqrt(results['emd_mean']**2 + results['emd_var']**2)
    
    # Compute RMSF (Root Mean Square Fluctuations) for generated and reference trajectories
    results['predicted_rmsf'] = mdtraj.rmsf(aftraj_aa, ref_aa) * 10
    results['ref_rmsf'] = mdtraj.rmsf(traj_aa, ref_aa) * 10
    results['global_rmsf r'] = scipy.stats.pearsonr(results['ref_rmsf'], results['predicted_rmsf'])[0]
    results['predicted_rmsf'] = np.median(results['predicted_rmsf'])
    results['ref_rmsf'] = np.median(results['ref_rmsf'])
    
    ref_pca, ref_coords = get_pca(traj.xyz) # PCA on reference trajectory
    af_coords_ref_pca = ref_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    seed_coords_ref_pca = ref_pca.transform(ref.xyz.reshape(1, -1)) # first frame of the trajectory
    fig = pca_plot(af_coords_ref_pca, ref_coords, seed_coords_ref_pca, label)
    wandb.log({f"PCA Plot": fig})

    try:
        af_tsne, ref_tsne, seed_tsne = get_tsne_joint(traj.xyz, aftraj.xyz, ref.xyz.reshape(1, *ref.xyz.shape[1:]))
        wandb.log({"tSNE Plot": _embedding_plot(af_tsne, ref_tsne, seed_tsne, label, method='t-SNE')})
    except Exception as e:
        print(f"t-SNE plot skipped: {e}")

    try:
        af_umap, ref_umap, seed_umap = get_umap_joint(traj.xyz, aftraj.xyz, ref.xyz.reshape(1, *ref.xyz.shape[1:]))
        wandb.log({"UMAP Plot": _embedding_plot(af_umap, ref_umap, seed_umap, label, method='UMAP')})
    except Exception as e:
        print(f"UMAP plot skipped: {e}")

    # log final_result to wandb
    for key, value in results.items():
        print(f"{label}_{key}: {value}")
        wandb.log({label+"_"+key: value})
        
    return results


def calculate_lDDT(
    pred_pos,  # (B, L, 3) or (L, 3)
    true_pos,  # (B, L, 3) or (L, 3)
    per_atom=False,  # if true, return lDDT for each atom (retain L axis)
    true_dist_cutoff=15.0,
    dist_diff_cutoffs=[0.5, 1.0, 2.0, 4.0],
):
    # Calculate predicted and true distance matrices
    def pos_to_dist(pos):
        return ((pos[..., None, :, :] - pos[..., None, :]) ** 2).sum(-1) ** 0.5

    pred_dist, true_dist = pos_to_dist(pred_pos), pos_to_dist(true_pos)

    # Get mask for distances to be scored (remove self distances)
    L = pred_pos.shape[-2]
    dist_mask = (true_dist < true_dist_cutoff) * (1.0 - np.eye(L))

    # Calculate contribution of each distance to lDDT
    dist_diff = np.abs(pred_dist - true_dist)
    score = (dist_diff[..., None] < np.array(dist_diff_cutoffs)).mean(-1)

    # Return calculated lDDT
    axis = (-1,) if per_atom else (-2, -1)
    return (score * dist_mask).sum(axis) / dist_mask.sum(axis)

def _lddt_pairs(pred_d, true_d, eye, cutoffs, cutoff=15.0):
    # pred_d, true_d: (..., L, L) precomputed distance matrices
    mask = (true_d < cutoff) * (1.0 - eye)
    diff = torch.abs(pred_d - true_d)
    s = (diff.unsqueeze(-1) < cutoffs).float().mean(-1)
    denom = mask.sum(dim=(-1, -2)).clamp(min=1.0)
    return (s * mask).sum(dim=(-1, -2)) / denom


def calculate_lddt_metrics(generated, dataset, device='cuda', n_gt_samples=None, gen_chunk=64, gt_chunk=64):
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    gen_cpu = torch.stack([g.detach().float().cpu() for g in generated])  # (Ng, L, 3)

    if n_gt_samples is None:
        n_gt_samples = len(generated)
    n_gt_samples = min(n_gt_samples, len(dataset))
    random_indices = random.sample(range(len(dataset)), k=n_gt_samples)
    gt_list = []
    for i in random_indices:
        gt_sample, _ = dataset[i]
        gt_list.append(dataset.reverse_transform(gt_sample).float().cpu())
    gt_cpu = torch.stack(gt_list)  # (Ngt, L, 3)

    Ng, L, _ = gen_cpu.shape
    Ngt = gt_cpu.shape[0]
    eye = torch.eye(L, device=dev)
    cutoffs = torch.tensor([0.5, 1.0, 2.0, 4.0], device=dev)

    scores = torch.zeros(Ng, Ngt)
    pbar = tqdm(total=((Ng + gen_chunk - 1) // gen_chunk) * ((Ngt + gt_chunk - 1) // gt_chunk),
                desc="Calculating lDDT matrix")
    for gi in range(0, Ng, gen_chunk):
        gen_block = gen_cpu[gi:gi + gen_chunk].to(dev)
        gen_d = torch.cdist(gen_block, gen_block)  # (cg, L, L)
        for gj in range(0, Ngt, gt_chunk):
            gt_block = gt_cpu[gj:gj + gt_chunk].to(dev)
            gt_d = torch.cdist(gt_block, gt_block)  # (cgt, L, L)
            # Pairwise: (cg, 1, L, L) vs (1, cgt, L, L) -> (cg, cgt)
            block = _lddt_pairs(gen_d.unsqueeze(1), gt_d.unsqueeze(0), eye, cutoffs)
            scores[gi:gi + gen_block.shape[0], gj:gj + gt_block.shape[0]] = block.cpu()
            pbar.update(1)
        del gen_d, gen_block
    pbar.close()

    recall = scores.max(dim=0).values.mean().item()
    precision = scores.max(dim=1).values.mean().item()

    rng = np.random.default_rng(seed=42)
    n_pair_samples = min(Ng * Ngt, 1000)

    def _pairwise_diversity(coords_cpu, n_items):
        i = rng.integers(0, n_items, size=n_pair_samples * 2)
        j = rng.integers(0, n_items, size=n_pair_samples * 2)
        keep = i != j
        i, j = i[keep][:n_pair_samples], j[keep][:n_pair_samples]
        scores_div = []
        for k in range(0, len(i), gen_chunk):
            ii = torch.from_numpy(i[k:k + gen_chunk]).long()
            jj = torch.from_numpy(j[k:k + gen_chunk]).long()
            d_i = torch.cdist(coords_cpu[ii].to(dev), coords_cpu[ii].to(dev))
            d_j = torch.cdist(coords_cpu[jj].to(dev), coords_cpu[jj].to(dev))
            scores_div.append(_lddt_pairs(d_i, d_j, eye, cutoffs).cpu())
        s = torch.cat(scores_div)
        return 1.0 - s.mean().item()

    diversity_predicted = _pairwise_diversity(gen_cpu, Ng)
    diversity_gt = _pairwise_diversity(gt_cpu, Ngt)

    return recall, precision, diversity_predicted, diversity_gt


def _rmsd_matrix_batched(gt, gen, chunk_size=256):
    L = gt.shape[1]
    gen_c = gen - gen.mean(dim=1, keepdim=True)
    gen_norm2 = (gen_c ** 2).sum(dim=(1, 2))
    gt_c = gt - gt.mean(dim=1, keepdim=True)
    gt_norm2 = (gt_c ** 2).sum(dim=(1, 2))

    Ngt = gt.shape[0]
    out = torch.empty(Ngt, gen.shape[0], device=gt.device, dtype=gt.dtype)
    for start in range(0, Ngt, chunk_size):
        end = min(start + chunk_size, Ngt)
        B = torch.einsum('ika,jkb->ijab', gt_c[start:end], gen_c)
        u, s, vh = torch.linalg.svd(B)
        sgn = torch.sign(torch.linalg.det(u @ vh))
        s_corrected = s.clone()
        s_corrected[..., -1] = s[..., -1] * sgn
        trace = s_corrected.sum(dim=-1)
        rmsd2 = (gt_norm2[start:end, None] + gen_norm2[None, :] - 2 * trace) / L
        out[start:end] = rmsd2.clamp(min=0).sqrt()
    return out


def precision_rmsd(generated=None, ground_truth=None, chunk_size=256, rmsd_matrix=None):
    # generated: (N, atoms, 3), ground_truth: (M, atoms, 3)
    # returns mean over generated of min RMSD to any ground truth
    if rmsd_matrix is None:
        rmsd_matrix = _rmsd_matrix_batched(ground_truth, generated, chunk_size)  # (M, N)
    return float(rmsd_matrix.min(dim=0).values.mean())


def recall_rmsd(generated=None, ground_truth=None, chunk_size=256, rmsd_matrix=None):
    # returns mean over ground truth of min RMSD to any generated
    if rmsd_matrix is None:
        rmsd_matrix = _rmsd_matrix_batched(ground_truth, generated, chunk_size)  # (M, N)
    return float(rmsd_matrix.min(dim=1).values.mean())


def emd_rmsd(generated=None, ground_truth=None, chunk_size=256, rmsd_matrix=None):
    # Hungarian min-cost matching between GT and generated; mean RMSD over matched pairs
    from scipy.optimize import linear_sum_assignment
    if rmsd_matrix is None:
        rmsd_matrix = _rmsd_matrix_batched(ground_truth, generated, chunk_size)  # (M, N)
    cost = rmsd_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def plot_pairwise_distance_distributions(gen_coords, gt_coords, atom_pair_1=None, atom_pair_2=None,
                                          cutoff_x=None, cutoff_y=None, label_x=None, label_y=None,
                                          ca_indices=None, save_path=None, min_seq_sep=5,
                                          max_abs_corr=0.5):
    # Figure-8-style plot: each 3D structure -> one point defined by two Cα-Cα pairwise
    # distances. Dashed cutoffs stratify samples into A/B/C macrostates; right panel
    # shows fraction in each state for GT vs model. Pairs and cutoffs auto-derived from
    # GT unless caller overrides.
    import matplotlib.pyplot as plt
    from scipy.stats import skew, kurtosis

    if isinstance(gen_coords, torch.Tensor):
        gen_coords = gen_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    # Restrict to Cα atoms. Backbone PDB ordering per residue is N, CA, C, O -> Cα at
    # indices 1, 5, 9, ... unless caller supplies explicit indices.
    n_atoms = gt_coords.shape[1]
    if ca_indices is None:
        if n_atoms % 4 == 0:
            ca_indices = np.arange(1, n_atoms, 4)
        else:
            ca_indices = np.arange(n_atoms)
    ca_indices = np.asarray(ca_indices, dtype=int)
    gt_ca = gt_coords[:, ca_indices]
    gen_ca = gen_coords[:, ca_indices]

    def pair_dist(coords, p):
        return np.linalg.norm(coords[:, p[0]] - coords[:, p[1]], axis=-1)

    if atom_pair_1 is None or atom_pair_2 is None:
        # Per-Cα positional std (RMSF-like) across the full GT trajectory. Assumes
        # GT is rigid-body aligned (ATLAS *_fit.xtc files are).
        gt_mean = gt_ca.mean(axis=0)  # (A, 3)
        pos_rmsf = np.sqrt(((gt_ca - gt_mean) ** 2).sum(axis=2).mean(axis=0))  # (A,)

        # Restrict pair search to the top-K most mobile Cα atoms.
        top_k = min(40, gt_ca.shape[1])
        top_atoms = np.sort(np.argsort(pos_rmsf)[::-1][:top_k])

        pairs = []
        for a_i in range(len(top_atoms)):
            for b_i in range(a_i + 1, len(top_atoms)):
                if abs(int(top_atoms[a_i]) - int(top_atoms[b_i])) >= min_seq_sep:
                    pairs.append((int(top_atoms[a_i]), int(top_atoms[b_i])))
        pairs = np.array(pairs, dtype=int)
        ii = pairs[:, 0]
        jj = pairs[:, 1]

        n_sub = min(500, gt_ca.shape[0])
        rng = np.random.default_rng(0)
        idx = rng.choice(gt_ca.shape[0], n_sub, replace=False)
        sub = gt_ca[idx]
        diffs = sub[:, ii] - sub[:, jj]
        dist_mat = np.linalg.norm(diffs, axis=-1)  # (S, P)
        mean_d = dist_mat.mean(axis=0)
        var = dist_mat.var(axis=0)
        n = dist_mat.shape[0]
        sk = skew(dist_mat, axis=0)
        ku = kurtosis(dist_mat, axis=0, fisher=False)
        denom = ku + 3 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
        bc = (sk ** 2 + 1) / np.where(denom > 0, denom, np.nan)
        v_n = var / (var.max() + 1e-9)
        bc_n = np.nan_to_num(bc, nan=0.0)
        # Also reward pairs whose endpoints are themselves highly mobile.
        mob = (pos_rmsf[ii] + pos_rmsf[jj]) / (pos_rmsf.max() + 1e-9)
        score = v_n * bc_n * mob

        q_short = np.quantile(mean_d, 0.25)
        q_long = np.quantile(mean_d, 0.75)
        short_mask = mean_d <= q_short
        long_mask = mean_d >= q_long

        def _top_masked(mask):
            return np.argsort(np.where(mask, score, -np.inf))[::-1]

        long_order = _top_masked(long_mask)
        p1_k = int(long_order[0])
        atom_pair_1 = (int(ii[p1_k]), int(jj[p1_k]))
        d1 = dist_mat[:, p1_k]

        short_order = _top_masked(short_mask)
        atom_pair_2 = None
        for k in short_order:
            k = int(k)
            if not np.isfinite(score[k]):
                break
            p = (int(ii[k]), int(jj[k]))
            sep = min(abs(p[0] - atom_pair_1[0]), abs(p[0] - atom_pair_1[1]),
                      abs(p[1] - atom_pair_1[0]), abs(p[1] - atom_pair_1[1]))
            if sep < min_seq_sep:
                continue
            corr = float(np.corrcoef(dist_mat[:, k], d1)[0, 1])
            if abs(corr) > max_abs_corr:
                continue
            atom_pair_2 = p
            break
        if atom_pair_2 is None:
            atom_pair_2 = (int(ii[int(short_order[0])]), int(jj[int(short_order[0])]))

    gt_x = pair_dist(gt_ca, atom_pair_2)
    gt_y = pair_dist(gt_ca, atom_pair_1)
    gen_x = pair_dist(gen_ca, atom_pair_2)
    gen_y = pair_dist(gen_ca, atom_pair_1)

    # Auto cutoffs: midpoint between the two modes of the GT distribution via
    # 2-component 1-D Gaussian mixture (falls back to median on failure).
    def _auto_cutoff(values):
        try:
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=2, random_state=0).fit(values.reshape(-1, 1))
            mu = np.sort(gm.means_.ravel())
            return float(mu.mean())
        except Exception:
            return float(np.median(values))

    if cutoff_x is None:
        cutoff_x = _auto_cutoff(gt_x)
    if cutoff_y is None:
        cutoff_y = _auto_cutoff(gt_y)

    if label_x is None:
        label_x = f"Cα{atom_pair_2[0]}-Cα{atom_pair_2[1]} distance (Å)"
    if label_y is None:
        label_y = f"Cα{atom_pair_1[0]}-Cα{atom_pair_1[1]} distance (Å)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(gt_x, gt_y, s=6, c='blue', label='Ground truth samples', alpha=0.6)
    ax.scatter(gen_x, gen_y, s=6, c='orange', label='Model samples', alpha=0.6)
    ax.axhline(cutoff_y, ls='--', c='gray', lw=1)
    ax.axvline(cutoff_x, ls='--', c='gray', lw=1)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend(loc='upper right')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin + 0.03*(xmax-xmin), cutoff_y + 0.03*(ymax-cutoff_y), 'A', fontsize=14, fontweight='bold')
    ax.text(xmin + 0.03*(xmax-xmin), ymin + 0.03*(cutoff_y-ymin), 'B', fontsize=14, fontweight='bold')
    ax.text(cutoff_x + 0.02*(xmax-cutoff_x), ymin + 0.03*(cutoff_y-ymin), 'C', fontsize=14, fontweight='bold')

    def fractions(x, y):
        fA = float((y > cutoff_y).mean())
        fB = float(((y <= cutoff_y) & (x <= cutoff_x)).mean())
        fC = float(((y <= cutoff_y) & (x > cutoff_x)).mean())
        return [fA, fB, fC]

    gt_frac = fractions(gt_x, gt_y)
    gen_frac = fractions(gen_x, gen_y)

    ax2 = axes[1]
    xs = np.arange(3)
    w = 0.4
    ax2.bar(xs - w/2, gt_frac, width=w, color='blue', label='Ground truth')
    ax2.bar(xs + w/2, gen_frac, width=w, color='orange', label='Model')
    ax2.set_xticks(xs)
    ax2.set_xticklabels(['A', 'B', 'C'])
    ax2.set_xlabel('macrostate')
    ax2.set_ylabel('Fraction of samples')
    ax2.legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, {'atom_pair_1': atom_pair_1, 'atom_pair_2': atom_pair_2,
                 'cutoff_x': cutoff_x, 'cutoff_y': cutoff_y,
                 'gt_fractions': gt_frac, 'gen_fractions': gen_frac}


def _run_analysis_for_samples(generated_samples, dataset, device, prefix="", skip_matching=False, compute_lddt=True, output_dir=None):
    segments = {
        "start": (0, 10000)
    }
    tag = f"{prefix}_" if prefix else ""
    if output_dir is None:
        output_dir = paths.results(prefix) if prefix else paths.results("_default")

    if isinstance(generated_samples, torch.Tensor):
        gen_tensor = generated_samples.to(device).float()
    else:
        gen_tensor = torch.stack([g.to(device) for g in generated_samples]).float()

    gt_per_segment = {}
    gt_offsets = {}
    gt_all_list = []
    cursor = 0
    for label, (sf, ef) in segments.items():
        subset = dataset[sf:ef]
        seg_gt = []
        for item in subset:
            gt_sample, _ = item
            seg_gt.append(dataset.reverse_transform(gt_sample).to(device).float())
        if not seg_gt:
            gt_per_segment[label] = None
            continue
        gt_seg = torch.stack(seg_gt)
        gt_per_segment[label] = gt_seg
        gt_offsets[label] = (cursor, cursor + gt_seg.shape[0])
        gt_all_list.append(gt_seg)
        cursor += gt_seg.shape[0]

    if not gt_all_list:
        print(f"{tag}No GT frames found across segments, aborting analysis.")
        return

    gt_all = torch.cat(gt_all_list, dim=0)

    try:
        dist_plot_path = os.path.join(output_dir, f"{tag}pairwise_distance_distribution.png")
        fig_dist, dist_info = plot_pairwise_distance_distributions(
            gen_coords=gen_tensor, gt_coords=gt_all, save_path=dist_plot_path
        )
        wandb.log({
            f"{tag}pairwise_distance_distribution": wandb.Image(dist_plot_path),
            f"{tag}pairwise_distance_gt_fractions": dist_info['gt_fractions'],
            f"{tag}pairwise_distance_gen_fractions": dist_info['gen_fractions'],
        })
        import matplotlib.pyplot as _plt
        _plt.close(fig_dist)
    except Exception as e:
        print(f"{tag}pairwise distance plot failed: {e}")

    print(f"\n{tag}Computing full RMSD matrix ({gt_all.shape[0]} GT × {gen_tensor.shape[0]} gen)...")
    rmsd_full = _rmsd_matrix_batched(gt_all, gen_tensor)
    rmsd_full_cpu = rmsd_full.detach().cpu().numpy()

    precision_rmsd_val = precision_rmsd(rmsd_matrix=rmsd_full)
    recall_rmsd_val = recall_rmsd(rmsd_matrix=rmsd_full)

    global_row_ind = global_col_ind = None
    if skip_matching:
        emd_rmsd_val = float('nan')
    else:
        from scipy.optimize import linear_sum_assignment
        print(f"{tag}Hungarian matching {rmsd_full_cpu.shape[0]} GT x {rmsd_full_cpu.shape[1]} gen...")
        global_row_ind, global_col_ind = linear_sum_assignment(rmsd_full_cpu)
        emd_rmsd_val = float(rmsd_full_cpu[global_row_ind, global_col_ind].mean())
    print(f"{tag}precision_rmsd: {precision_rmsd_val}, recall_rmsd: {recall_rmsd_val}, emd_rmsd: {emd_rmsd_val}")
    wandb.log({
        f"{tag}precision_rmsd": precision_rmsd_val,
        f"{tag}recall_rmsd": recall_rmsd_val,
        f"{tag}emd_rmsd": emd_rmsd_val,
    })

    all_rmsd_values = []
    for label, (start_frame, end_frame) in segments.items():
        gt_seg = gt_per_segment.get(label)
        if gt_seg is None:
            continue
        a, b = gt_offsets[label]
        rmsd_seg = rmsd_full_cpu[a:b]

        print(f"\nProcessing {tag}{label} segment ({start_frame}-{end_frame})...")
        seg_len = rmsd_seg.shape[0]
        n_pick = min(seg_len, gen_tensor.shape[0])
        if skip_matching:
            winner_indices = list(range(n_pick))
            rmsd_best = [float(rmsd_seg[k, k]) for k in range(n_pick)]
        elif a == 0 and b == rmsd_full_cpu.shape[0]:
            order = np.argsort(global_row_ind)
            winner_indices = global_col_ind[order].tolist()
            rmsd_best = [float(rmsd_full_cpu[r, c])
                         for r, c in zip(global_row_ind[order], global_col_ind[order])]
        else:
            from scipy.optimize import linear_sum_assignment
            print(f"Hungarian matching {rmsd_seg.shape[0]} GT x {rmsd_seg.shape[1]} gen...")
            row_ind, col_ind = linear_sum_assignment(rmsd_seg)
            order = np.argsort(row_ind)
            winner_indices = col_ind[order].tolist()
            rmsd_best = [float(rmsd_seg[r, c]) for r, c in zip(row_ind[order], col_ind[order])]

        if not winner_indices:
            print(f"No matches found for {tag}{label} segment, skipping...")
            continue

        winner_tensor = torch.tensor(winner_indices, device=device, dtype=torch.long)
        gt_batch = gt_seg[:len(winner_indices)]
        gen_batch = gen_tensor.index_select(0, winner_tensor)
        aligned_batch = rmsdalign(gt_batch, gen_batch)
        matched = [aligned_batch[i].detach().cpu() for i in range(aligned_batch.shape[0])]
        rmsd = rmsd_best

        gt_pdb_path, gen_pdb_path, gen_traj_path, gt_traj_path = save_matched_samples_as_trajectory(
            matched,
            dataset,
            output_dir=output_dir,
            label=label,
            start_frame=start_frame,
            end_frame=end_frame
        )

        save_paths(gt_pdb_path, gen_pdb_path, gen_traj_path, gt_traj_path)

        results = analyze_trajectories(
            ref_pdb_path=gt_pdb_path,
            ref_xtc_path=gt_traj_path,
            gen_pdb_path=gen_pdb_path,
            gen_xtc_path=gen_traj_path,
            bb_only=True,
            label=f"{tag}{label}",
            wandb=wandb
        )

        all_rmsd_values.append(results['rmsd_between_predicted_and_reference'])

        # Keep gifs under output_dir so concurrent runs don't share saved_gifs/temp_frames.
        gif_path = os.path.join(output_dir, "gifs", f"{tag}{label}_compare_trajectory.gif")
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)

        try:
            gif_file = save_gif(
                gt_pdb=gt_pdb_path,
                gt_xtc=gt_traj_path,
                gen_pdb=gen_pdb_path,
                gen_xtc=gen_traj_path,
                gif_path=gif_path,
                frame_dir=os.path.join(output_dir, "temp_frames", f"{tag}{label}"),
            )
            wandb.log({f"{tag}{label}_compare_trajectory": wandb.Image(str(gif_file))})
        except Exception as e:
            print(f"{tag}{label} trajectory gif failed (metrics unaffected): {e}")

        wandb.log({f"{tag}{label}_rmsd": results['rmsd_between_predicted_and_reference']})

    if all_rmsd_values:
        average_rmsd = np.mean(all_rmsd_values)
        print(f"\n{tag}Average RMSD across all segments: {average_rmsd}")
        wandb.log({f"{tag}average_rmsd_all_segments": average_rmsd})

    if compute_lddt:
        print(f"\n{tag}Calculating LDDT metrics...")
        recall, precision, diversity_pred, diversity_gt = calculate_lddt_metrics(
            generated=generated_samples,
            dataset=dataset,
            n_gt_samples=len(dataset),
        )
        print(f"{tag}Recall: {recall}, Precision: {precision}, Diversity Pred: {diversity_pred}, Diversity GT: {diversity_gt}")
        wandb.log({
            f"{tag}recall": recall,
            f"{tag}precision": precision,
            f"{tag}diversity_pred": diversity_pred,
            f"{tag}diversity_gt": diversity_gt
        })
    else:
        print(f"\n{tag}Skipping LDDT metrics (compute_lddt=False)")

def run_analysis(model, dataset, pdb_file, device=None, N=20, num_samples=10000, sample_scaler=None, samples=None, skip_matching=False, prefix="", compute_lddt=True, output_dir=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if samples is None:
        model.to(device)

        try:
            visualize_structures(
                model=model,
                dataset=dataset,
                pdb_file=pdb_file,
                save_dir=output_dir or paths.results("_default"),
                device=device,
                N=N
            )
        except Exception as e:
            print(f"Error in structure visualization: {e}")
            import traceback
            traceback.print_exc()

        try:
            print("Generating samples and matching to MD trajectory...")
            generated_samples = generate_samples(
                model=model,
                dataset=dataset,
                num_samples=num_samples,
                device=device,
                N=N,
                sample_scaler=sample_scaler,
            )
        except Exception as e:
            print(f"Error generating samples: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        generated_samples = samples

    _run_analysis_for_samples(generated_samples, dataset, device, prefix=prefix, skip_matching=skip_matching, compute_lddt=compute_lddt, output_dir=output_dir)

def run_guided_analysis(model, dataset, device=None, N=None, warmup_steps=None, guidance_strength=None, n_candidates=None, num_samples=10000, compute_lddt=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    N = N if N is not None else model.guided_N
    warmup_steps = warmup_steps if warmup_steps is not None else model.guided_warmup_steps
    guidance_strength = guidance_strength if guidance_strength is not None else model.guided_strength
    n_candidates = n_candidates if n_candidates is not None else model.guided_n_candidates

    try:
        print("Generating guided samples...")
        guided_samples = generate_guided_samples(
            model=model, dataset=model.dataset, num_samples=num_samples, device=device,
            n_candidates=n_candidates, N=N,
            warmup_steps=warmup_steps, guidance_strength=guidance_strength
        )
    except Exception as e:
        print(f"Error generating guided samples: {e}")
        import traceback
        traceback.print_exc()
        return

    _run_analysis_for_samples(guided_samples, dataset, device, prefix="guided", compute_lddt=compute_lddt)

def create_latent_gif(latent_dataset, md_dataset, output_dir=None, gif_name="latent_vs_gt.gif", fps=5):
    output_dir = output_dir or os.path.join(paths.ANALYSIS, "latent_comparison")
    os.makedirs(output_dir, exist_ok=True)

    template_u = md_dataset.universe
    backbone_atoms = template_u.select_atoms("backbone")

    pdb_path = os.path.join(output_dir, "backbone_template.pdb")
    template_u.trajectory[0]
    backbone_atoms.write(pdb_path)

    backbone_u = mda.Universe(pdb_path)

    n_frames = min(len(latent_dataset), len(md_dataset))

    # align latent to gt for each frame
    aligned_latent = []
    gt_coords_list = []
    for i in range(n_frames):
        lat, _ = latent_dataset[i]
        gt, _ = md_dataset[i]
        if isinstance(lat, torch.Tensor):
            lat = lat.detach().cpu()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu()
        aligned = rmsdalign(gt, lat)
        aligned_latent.append(aligned.numpy() if isinstance(aligned, torch.Tensor) else aligned)
        gt_coords_list.append(gt.numpy() if isinstance(gt, torch.Tensor) else gt)

    # save latent trajectory
    latent_xtc = os.path.join(output_dir, "latent_trajectory.xtc")
    with mda.coordinates.XTC.XTCWriter(latent_xtc, backbone_atoms.n_atoms) as W:
        for coords in aligned_latent:
            backbone_u.atoms.positions = coords
            W.write(backbone_u.atoms)

    # save gt trajectory
    gt_xtc = os.path.join(output_dir, "gt_trajectory.xtc")
    with mda.coordinates.XTC.XTCWriter(gt_xtc, backbone_atoms.n_atoms) as W:
        for coords in gt_coords_list:
            backbone_u.atoms.positions = coords
            W.write(backbone_u.atoms)

    # create gif
    gif_path = os.path.join(output_dir, gif_name)
    gif_file = save_gif(
        gt_pdb=pdb_path,
        gt_xtc=gt_xtc,
        gen_pdb=pdb_path,
        gen_xtc=latent_xtc,
        gif_path=gif_path,
        fps=fps
    )

    wandb.log({f"latent_compare_trajectory": wandb.Image(str(gif_file))})

    return gif_file

def rmsdalign(a, b, weights=None): # aligns B to A  # [*, N, 3]
    # Ensure tensors are on the same device
    if a.device != b.device:
        b = b.to(a.device)
    if weights is not None and weights.device != a.device:
        weights = weights.to(a.device)
    
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights is None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    
    # Compute means and centered coordinates (avoid inplace operations)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a_centered = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b_centered = b - b_mean
    
    # Compute covariance matrix
    B_matrix = torch.einsum('...ji,...jk->...ik', weights * a_centered, b_centered)
    u, s, vh = torch.linalg.svd(B_matrix)

    # Correct improper rotation if necessary
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s_corrected = s.clone()
    s_corrected[...,-1] = s[...,-1] * sgn
    u_corrected = u.clone()
    u_corrected[...,:,-1] = u[...,:,-1] * sgn.unsqueeze(-1)
    
    C = u_corrected @ vh
    return b_centered @ C.mT + a_mean