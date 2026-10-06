#!/usr/bin/env python3
# cryoDRGN-guided AlphaFold3 ensemble, run against an unmodified checkout of
# https://github.com/sai-advaith/guided_alphafold (set up as its README describes). Each map
# decoded by tools/cryodrgn_maps.py guides one batch of structures:
#
#   python tools/af3_cryodrgn_guided.py --guided_alphafold /path/to/guided_alphafold \
#       --maps_dir cdrgn/1a62_A/maps --sim_dir "$CRYO_DATASET_ROOT/1a62_A_sim_output" \
#       --md_pdb data/md/1a62_A/1a62_A.pdb --run_name 1a62_A --out_dir af3_guided/1a62_A
#
# Writes one multi-model backbone PDB per map; score the folder with
# eval/eval_guided_alphafold.py.
#
# The paper's ensembles were generated at guided_alphafold commit 9ef3bf9. Later commits also
# work but changed the guidance step, so they give different samples of the same quality.
import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np

# MD force-field atom names -> PDB chemical component names
TERMINAL_OXYGEN_RENAME = {"OT1": "O", "OT2": "OXT", "O1": "O", "O2": "OXT"}
RESIDUE_ATOM_RENAME = {("ILE", "CD"): "CD1"}

# Settings that differ from guided_alphafold's example config (cryo_baseline.yaml), for a
# single chain in a 64^3 box at 1.2 A/pix.
OVERRIDES = {
    "protein.residue_range": [[1, 2]],
    # without it the density gradient snaps the backbone of many samples
    "loss_function.bond_length_loss_weight": 1.0,
    "loss_function.cryoesp_loss_function.alignment_strategy": "global_rmsd_to_gt",
    "loss_function.cryoesp_loss_function.evaluate_only_resolved": False,
    "loss_function.cryoesp_loss_function.reduced_D": None,
    "loss_function.cryoesp_loss_function.rmax_for_mask": 4.0,
    "loss_function.cryoesp_loss_function.sinkhorn.blur": 0.05,
    "loss_function.cryoesp_loss_function.sinkhorn.guide_multimer_by_chains": False,
    "loss_function.cryoesp_loss_function.sinkhorn.scaling": 0.9,
    "loss_function.cryoesp_loss_function.sinkhorn.turn_off_after": 100,
    "loss_function.cryoesp_loss_function.symmetry.symmetry_type": None,
    "loss_function.cryoesp_loss_function.symmetry.reapply_symmetry_every": False,
}

# The particle images are simulated from backbone-only coordinates, so the maps hold no
# side-chain density. guided_alphafold renders every heavy atom against the map; these edits
# restrict its cryo-EM forward model to N/CA/C/O. They are applied to its loss module in memory.
BACKBONE_ONLY_EDITS = [
    ("        self.AF3_to_pdb_mask = mask.to(device) # masking which of the AF3 entries are missing in the pdb file to align appropriately\n",
     "        self.AF3_to_pdb_mask = mask.to(device)\n"
     "        self.guidance_atom_mask = create_backbone_atom_mask(full_sequences, self.sequence_types, device=device)\n"
     "        assert self.guidance_atom_mask.shape == self.AF3_to_pdb_mask.shape\n"
     "        self.mask_for_esp = self.AF3_to_pdb_mask & self.guidance_atom_mask\n"),
    ("atom_mask = self.AF3_to_pdb_mask,\n            single_b_fac=300.0, use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_mask,",
     "atom_mask = self.mask_for_esp,\n            single_b_fac=300.0, use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_mask,"),
    ("atom_mask = self.regions_of_interest_mask if self.guide_only_ROI else None,",
     "atom_mask = self.regions_of_interest_mask if self.guide_only_ROI else self.guidance_atom_mask,"),
    ("        mask_for_OT = self.AF3_to_pdb_mask\n",
     "        mask_for_OT = self.mask_for_esp\n"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--guided_alphafold", default=".", help="guided_alphafold checkout (default: cwd)")
    p.add_argument("--maps_dir", required=True, help="esp_*.map files from tools/cryodrgn_maps.py decode")
    p.add_argument("--sim_dir", required=True, help="<protein>_sim_output the maps were trained on")
    p.add_argument("--md_pdb", required=True, help="all-atom MD topology PDB of the target")
    p.add_argument("--run_name", required=True, help="e.g. 1a62_A")
    p.add_argument("--out_dir", required=True, help="folder for the map_*.pdb files")
    p.add_argument("--batch_size", type=int, default=8, help="structures guided jointly by each map")
    p.add_argument("--resolution", type=float, default=3.1, help="resolution of the maps (A)")
    p.add_argument("--diffusion_N", type=int, default=None, help="override diffusion steps")
    p.add_argument("--apix", type=float, default=1.2)
    p.add_argument("--box", type=int, default=64)
    p.add_argument("--max_maps", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1, help="split the maps over several GPUs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def mean_gt_backbone(sim_dir):
    # mean backbone over all simulated frames, each centred on its bounding box (the map frame's origin)
    ds = os.path.join(sim_dir, "dataset")
    names = sorted(f for f in os.listdir(ds) if f.endswith(".npz"))
    coords = np.stack([np.load(os.path.join(ds, n))["coords_3d"].astype(np.float32) for n in names])
    centers = ((coords.min(1) + coords.max(1)) / 2).astype(np.float32)
    return (coords - centers[:, None, :]).astype(np.float64).mean(0)


def kabsch(mobile, target):
    mc, tc = mobile.mean(0), target.mean(0)
    H = (mobile - mc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, tc - R @ mc


def write_reference_pdb(md_pdb, ref_backbone, out_pdb, box, apix):
    # heavy-atom, single-chain copy renumbered 1..L and placed in the map frame: its backbone is
    # superimposed on the mean ground-truth backbone, then shifted to the centre of the box.
    # Returns the one-letter sequence.
    import gemmi
    st = gemmi.read_structure(md_pdb)
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    model = st[0]
    while len(model) > 1:
        del model[len(model) - 1]
    chain = model[0]
    chain.name = "A"
    for i, residue in enumerate(chain):
        residue.seqid.num = i + 1
        for atom in residue:
            atom.name = RESIDUE_ATOM_RENAME.get((residue.name, atom.name), atom.name)
    pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in chain for a in r])
    # before the terminal-oxygen rename, so the C-terminal OT1 is not counted as a backbone O
    bb = np.array([k for k, a in enumerate(a for r in chain for a in r) if a.name in ("N", "CA", "C", "O")])
    for atom in chain[len(chain) - 1]:
        atom.name = TERMINAL_OXYGEN_RENAME.get(atom.name, atom.name)
    assert len(bb) == ref_backbone.shape[0], f"{len(bb)} backbone atoms in {md_pdb}, dataset has {ref_backbone.shape[0]}"
    R, t = kabsch(pos[bb], ref_backbone)
    pos = pos @ R.T + t + box / 2.0 * apix
    k = 0
    for residue in chain:
        for atom in residue:
            atom.pos = gemmi.Position(*pos[k])
            atom.b_iso = 100.0
            k += 1
    os.makedirs(os.path.dirname(out_pdb), exist_ok=True)
    st.write_pdb(out_pdb)
    return gemmi.one_letter_code([r.name for r in chain]).upper()


def install_backbone_only_guidance(root):
    import gemmi
    import torch
    import src.losses
    from src.utils.io import SEQUENCE_TYPE_TO_ATOM_DICTIONARY, SEQUENCE_TYPE_TO_RESIDUE_KIND

    def create_backbone_atom_mask(sequences, sequence_types=None, device=torch.device("cpu")):
        # one mask over all AF3 atoms selecting the N/CA/C/O of every protein residue
        if sequence_types is None:
            sequence_types = ["proteinChain"] * len(sequences)
        mask = []
        for sequence, sequence_type in zip(sequences, sequence_types):
            atoms = SEQUENCE_TYPE_TO_ATOM_DICTIONARY[sequence_type]
            kind = SEQUENCE_TYPE_TO_RESIDUE_KIND[sequence_type]
            per_residue = []
            for code in sequence:
                n_atoms = len(atoms[gemmi.expand_one_letter(code, kind)])
                if sequence_type == "proteinChain":
                    per_residue.append([1, 1, 1, 1][:n_atoms] + [0] * max(0, n_atoms - 4))
                else:
                    per_residue.append([0] * n_atoms)
            # the extra slot guided_alphafold adds: OXT on the last protein residue, OP3 on the first nucleotide
            per_residue[-1 if sequence_type == "proteinChain" else 0].append(0)
            mask += [a for residue_mask in per_residue for a in residue_mask]
        return torch.tensor(mask, dtype=torch.bool, device=device).flatten()

    path = os.path.join(root, "src", "losses", "em_loss_function.py")
    code = open(path).read()
    for old, new in BACKBONE_ONLY_EDITS:
        assert code.count(old) == 1, f"guided_alphafold's {path} changed; cannot apply the backbone-only edit:\n{old}"
        code = code.replace(old, new)
    spec = importlib.util.spec_from_file_location("src.losses.em_loss_function", path)
    module = importlib.util.module_from_spec(spec)
    module.create_backbone_atom_mask = create_backbone_atom_mask
    exec(compile(code, path, "exec"), module.__dict__)
    sys.modules[spec.name] = module
    src.losses.em_loss_function = module
    src.losses.CryoEM_ESP_GuidanceLossFunction = module.CryoEM_ESP_GuidanceLossFunction


def backbone_selection(atom_array):
    # N/CA/C/O per residue, no O on the last one: the layout of the ground-truth ensembles
    names = np.asarray(atom_array.atom_name)
    res = np.asarray(atom_array.res_id)
    residues = sorted(set(res.tolist()))
    out = []
    for r in residues:
        where = np.nonzero(res == r)[0]
        for nm in ("N", "CA", "C", "O"):
            if nm == "O" and r == residues[-1]:
                continue
            hit = where[names[where] == nm]
            assert len(hit) == 1, f"residue {r}: {len(hit)} atoms named {nm}"
            out.append(int(hit[0]))
    return np.array(out)


def write_models(path, coords, atom_array, bb):
    names = np.asarray(atom_array.atom_name)[bb]
    resn = np.asarray(atom_array.res_name)[bb]
    resi = np.asarray(atom_array.res_id)[bb]
    with open(path + ".tmp", "w") as f:
        for m, xyz in enumerate(coords, 1):
            f.write(f"MODEL     {m:4d}\n")
            for k, (x, y, z) in enumerate(xyz):
                f.write(f"ATOM  {k + 1:5d}  {names[k]:<3s} {resn[k]:>3s} A{int(resi[k]):4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {names[k][0]:>2s}\n")
            f.write("ENDMDL\n")
        f.write("END\n")
    os.replace(path + ".tmp", path)


def set_dotted(config, key, value):
    *parents, leaf = key.split(".")
    for p in parents:
        config = getattr(config, p)
    setattr(config, leaf, value)


def main():
    args = parse_args()
    md_pdb, out_dir = os.path.abspath(args.md_pdb), os.path.abspath(args.out_dir)
    maps = sorted(glob.glob(os.path.join(os.path.abspath(args.maps_dir), "esp_*.map")))
    assert maps, f"no esp_*.map under {args.maps_dir}"
    ref_backbone = mean_gt_backbone(os.path.abspath(args.sim_dir))
    root = os.path.abspath(args.guided_alphafold)
    os.chdir(root)                 # guided_alphafold resolves its data and config paths from cwd
    sys.path.insert(0, root)
    install_backbone_only_guidance(root)
    import torch
    from experiment_manager import ExperimentManager
    from src.utils.io import load_config
    from src.utils.process_pipeline_inputs.preprocess_em_inputs import (
        create_em_configuration_file_from_baseline, save_metadata)

    pdb_id = args.run_name.split("_")[0]
    assembly = f"{args.run_name}_cryodrgn"
    inputs, outputs = "pipeline_inputs_cryodrgn", "pipeline_outputs_cryodrgn"
    ref_pdb = os.path.join(inputs, "pdbs", pdb_id, f"{args.run_name}_ref.pdb")
    sequence = write_reference_pdb(md_pdb, ref_backbone, ref_pdb, args.box, args.apix)
    print(f"{args.run_name}: {len(sequence)} residues")

    if args.max_maps:
        maps = maps[:args.max_maps]
    if args.nshards > 1:
        maps = maps[args.shard::args.nshards]
    # the maps are ordered by trajectory time; shuffled so a partial run still covers all of it
    np.random.default_rng(args.shard).shuffle(maps)
    print(f"shard {args.shard}/{args.nshards}: {len(maps)} maps x {args.batch_size} structures")

    save_metadata(pdb_id, assembly, [sequence], [1], ["proteinChain"], inputs)
    config = load_config(create_em_configuration_file_from_baseline(
        pdb_id, [sequence], [1], ["proteinChain"], None, ref_pdb, assembly, inputs, outputs,
        None, None, "pipeline_configurations/cryo_baseline.yaml"))
    for key, value in OVERRIDES.items():
        set_dotted(config, key, value)
    cryoesp = config.loss_function.cryoesp_loss_function
    cryoesp.emdb_resolution = args.resolution
    cryoesp.esp_file = maps[0]
    config.general.name = f"{args.run_name}_cryodrgn_em"
    config.general.batch_size = args.batch_size
    config.model_manager.dump_dir = f"{config.model_manager.dump_dir}_shard{args.shard}"
    if args.diffusion_N:
        config.model_manager.diffusion_N = args.diffusion_N

    ExperimentManager.seed_experiment(
        args.seed if args.seed is not None else config.general.seed + args.shard)
    pipeline = ExperimentManager(config, args.device)
    bb = backbone_selection(pipeline.model_manager.atom_array)
    origin_shift = args.box / 2.0 * args.apix      # map frame -> particle frame
    os.makedirs(out_dir, exist_ok=True)

    t0, done = time.time(), 0
    for k, esp in enumerate(maps):
        idx = int(os.path.basename(esp).split("_")[1].split(".")[0])
        out = os.path.join(out_dir, f"map_{idx:06d}.pdb")
        if os.path.exists(out):
            continue
        cryoesp.esp_file = esp
        pipeline.loss_function = pipeline._get_loss_function()
        structures = pipeline.run_full_diffusion_process(pipeline.get_initial_latents())
        with torch.no_grad():
            aligned, _, _ = pipeline.cryoesp_loss_function.align_structure(
                structures.detach(), pipeline.cryoesp_loss_function.coordinates_gt.unsqueeze(0),
                i=config.model_manager.diffusion_N - 1, is_counted_down=False)
        write_models(out, aligned[:, bb, :].cpu().numpy() - origin_shift,
                     pipeline.model_manager.atom_array, bb)
        done += 1
        el = time.time() - t0
        print(f"[{k + 1}/{len(maps)}] map {idx}: {el / done:.0f}s per map, "
              f"{(len(maps) - (k + 1)) * el / done / 3600:.2f}h left", flush=True)


if __name__ == "__main__":
    main()
