import numpy as np
import torch


def gram_schmidt(v1, v2):
    # v1, v2: [..., 3] -> R [..., 3, 3] with columns (e1, e2, e3)
    e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u2 = v2 - e1 * (e1 * v2).sum(-1, keepdim=True)
    e2 = u2 / u2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def build_base_frames(base_coords, resindices, names, enm_cutoff=10.0, enm_min_sep=3):
    # per-residue rigid frames (origin CA, axes from C and N) of the base conformation
    base_coords = np.asarray(base_coords, dtype=np.float32)
    resindices = np.asarray(resindices)
    names = list(names)

    res2idx, order = {}, []
    for r in resindices:
        if r not in res2idx:
            res2idx[r] = len(order)
            order.append(r)
    n_res = len(order)
    atom2res = np.array([res2idx[r] for r in resindices], dtype=np.int64)

    R_base = np.zeros((n_res, 3, 3), dtype=np.float32)
    t_base = np.zeros((n_res, 3), dtype=np.float32)
    per_res_atoms = []
    for ridx, r in enumerate(order):
        atom_ids = np.where(resindices == r)[0]
        res_names = {names[a]: a for a in atom_ids}
        per_res_atoms.append(res_names)
        if not {"N", "CA", "C"} <= set(res_names):
            raise ValueError(f"Residue {r} missing N/CA/C backbone atoms: {list(res_names)}")
        ca = base_coords[res_names["CA"]]
        n = base_coords[res_names["N"]]
        c = base_coords[res_names["C"]]
        e1 = c - ca
        e1 = e1 / (np.linalg.norm(e1) + 1e-8)
        v = n - ca
        u2 = v - e1 * float(e1 @ v)
        e2 = u2 / (np.linalg.norm(u2) + 1e-8)
        e3 = np.cross(e1, e2)
        R_base[ridx] = np.stack([e1, e2, e3], axis=-1)
        t_base[ridx] = ca

    R_base_t = torch.from_numpy(R_base)
    t_base_t = torch.from_numpy(t_base)
    atom2res_t = torch.from_numpy(atom2res)

    # atom coords in their residue frame: local = R_base^T (pos - t_base)
    pos = torch.from_numpy(base_coords)
    centered = (pos - t_base_t[atom2res_t]).unsqueeze(-1)
    local = torch.matmul(R_base_t[atom2res_t].transpose(-1, -2), centered).squeeze(-1)

    # backbone-continuity pairs: only inter-residue ones (intra-residue distances are
    # invariant, each residue is rigid): the C(i)-N(i+1) peptide bond and CA(i)-CA(i+1)
    bond_i, bond_j = [], []
    for ridx in range(n_res - 1):
        a, b = per_res_atoms[ridx], per_res_atoms[ridx + 1]
        for na, nb in (("C", "N"), ("CA", "CA")):
            if na in a and nb in b:
                bond_i.append(a[na]); bond_j.append(b[nb])
    bond_i_t = torch.tensor(bond_i, dtype=torch.long)
    bond_j_t = torch.tensor(bond_j, dtype=torch.long)
    bond_target = (pos[bond_i_t] - pos[bond_j_t]).norm(dim=-1)

    # elastic-network pairs: CA-CA contacts of the base conformation (close in space,
    # separated in sequence)
    ca_idx = torch.tensor([per_res_atoms[r]["CA"] for r in range(n_res)], dtype=torch.long)
    dmat = torch.cdist(pos[ca_idx], pos[ca_idx])
    iu = torch.triu_indices(n_res, n_res, offset=1)
    keep = (dmat[iu[0], iu[1]] < enm_cutoff) & ((iu[0] - iu[1]).abs() >= enm_min_sep)
    enm_i = ca_idx[iu[0][keep]]
    enm_j = ca_idx[iu[1][keep]]
    enm_target = dmat[iu[0][keep], iu[1][keep]]

    return {
        "R_base": R_base_t,          # [n_res, 3, 3]
        "t_base": t_base_t,          # [n_res, 3]
        "local_atoms": local,        # [n_atoms, 3]
        "atom2res": atom2res_t,      # [n_atoms]
        "bond_i": bond_i_t, "bond_j": bond_j_t, "bond_target": bond_target,
        "enm_i": enm_i, "enm_j": enm_j, "enm_target": enm_target,
        "n_res": n_res,
        "n_atoms": base_coords.shape[0],
    }


def base_frames_from_universe(universe, device):
    # base conformation = trajectory frame 0, centred
    universe.trajectory[0]
    bb = universe.select_atoms("backbone")
    base = torch.tensor(bb.positions.copy(), dtype=torch.float32)
    base = base - base.mean(0, keepdim=True)
    bf = build_base_frames(base.numpy(), bb.resindices, bb.names)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bf.items()}


def apply_delta_frames(delta9, R_base, t_base, local_atoms, atom2res):
    # delta9 [B, n_res, 9] = (translation, rot_v1, rot_v2) per residue; zero -> base conformation
    dt = delta9[..., 0:3]
    r1 = delta9[..., 3:6] + torch.tensor([1.0, 0.0, 0.0], device=delta9.device)
    r2 = delta9[..., 6:9] + torch.tensor([0.0, 1.0, 0.0], device=delta9.device)
    dR = gram_schmidt(r1, r2)                                # [B, n_res, 3, 3]

    R_final = torch.matmul(R_base.unsqueeze(0), dR)
    t_final = t_base.unsqueeze(0) + dt

    R_atom = R_final[:, atom2res]                            # [B, n_atoms, 3, 3]
    t_atom = t_final[:, atom2res]
    return torch.einsum("banj,aj->ban", R_atom, local_atoms) + t_atom
