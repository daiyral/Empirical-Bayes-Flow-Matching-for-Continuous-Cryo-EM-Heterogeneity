#!/usr/bin/env python3
# Unguided AlphaFold3 ensemble from MSA subsampling, run against an unmodified checkout of
# https://github.com/sai-advaith/guided_alphafold (set up as its README describes):
#
#   python tools/af3_msa_subsampling.py --guided_alphafold /path/to/guided_alphafold \
#       --md_pdb data/md/1a62_A/1a62_A.pdb --run_name 1a62_A --out_dir af3_msa/1a62_A
#
# Writes one multi-model backbone PDB per MSA subsample; score the folder with
# eval/eval_unguided_alphafold.py.
import argparse
import math
import os
import sys
import time

import numpy as np

MSA_ROW_FEATS = ("msa", "has_deletion", "deletion_value")
# MD force-field atom names -> PDB chemical component names
TERMINAL_OXYGEN_RENAME = {"OT1": "O", "OT2": "OXT", "O1": "O", "O2": "OXT"}
RESIDUE_ATOM_RENAME = {("ILE", "CD"): "CD1"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--guided_alphafold", default=".", help="guided_alphafold checkout (default: cwd)")
    p.add_argument("--md_pdb", required=True, help="all-atom MD topology PDB of the target")
    p.add_argument("--run_name", required=True, help="e.g. 1a62_A")
    p.add_argument("--out_dir", required=True, help="folder for the sub_*.pdb files")
    p.add_argument("--n_structures", type=int, default=10001)
    p.add_argument("--batch_size", type=int, default=8, help="structures drawn per MSA subsample")
    p.add_argument("--msa_depth", type=int, default=64, help="sequences kept per subsample")
    p.add_argument("--diffusion_N", type=int, default=None, help="override diffusion steps")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1, help="split the subsamples over several GPUs")
    p.add_argument("--seed", type=int, default=98710)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def write_reference_pdb(md_pdb, out_pdb):
    # heavy-atom, single-chain copy renumbered 1..L; returns the one-letter sequence
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
            atom.b_iso = 100.0
    for atom in chain[len(chain) - 1]:
        atom.name = TERMINAL_OXYGEN_RENAME.get(atom.name, atom.name)
    os.makedirs(os.path.dirname(out_pdb), exist_ok=True)
    st.write_pdb(out_pdb)
    return gemmi.one_letter_code([r.name for r in chain]).upper()


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


def raw_deletions(deletion_value):
    # invert deletion_value = (2/pi) arctan(d/3)
    import torch
    v = deletion_value.float().clamp(0.0, 1.0 - 1e-6)
    return 3.0 * torch.tan(v * (math.pi / 2))


def subsample_msa_features(feat, depth, seed):
    # random rows of the MSA (query row always kept); per-column summaries recomputed
    import torch
    import torch.nn.functional as F
    n_msa = feat["msa"].shape[0]
    d = min(depth, n_msa)
    g = torch.Generator().manual_seed(seed)
    rest = torch.randperm(n_msa - 1, generator=g)[: d - 1] + 1
    idx = torch.cat([torch.zeros(1, dtype=torch.long), rest.sort().values]).to(feat["msa"].device)
    out = dict(feat)
    for k in MSA_ROW_FEATS:
        out[k] = feat[k].index_select(0, idx)
    out["profile"] = F.one_hot(out["msa"].long(), num_classes=feat["profile"].shape[-1]).float().mean(0)
    out["deletion_mean"] = raw_deletions(out["deletion_value"]).mean(0)
    return out, d


def recompute_trunk(mm, feat, depth, MSA):
    # re-run the pairformer cycles on the subsampled MSA
    import torch
    mm.input_feature_dict = feat
    mm.protenix_model.msa_module.msa_configs["test_cutoff"] = depth
    mm.protenix_model.msa_module.msa_configs["test_lowerb"] = depth
    mixed = mm.pairformer_mixed_precision
    with torch.no_grad():
        mm._setup_s_init_z_init()
        s = torch.zeros_like(mm.s_init)
        z = torch.zeros_like(mm.z_init)
        mm.msa = MSA(mm.s_inputs_init, s, z)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if mixed else torch.float32):
            if mixed:
                s, z = s.to(torch.bfloat16), z.to(torch.bfloat16)
            for _ in range(mm.N_cycle):
                s, z = mm.pairformer_cycle(s, z)
        if mixed:
            s, z = s.float(), z.float()
        mm.msa = MSA(mm.s_inputs_init, s, z)


def main():
    args = parse_args()
    md_pdb, out_dir = os.path.abspath(args.md_pdb), os.path.abspath(args.out_dir)
    root = os.path.abspath(args.guided_alphafold)
    os.chdir(root)                 # guided_alphafold resolves its data and config paths from cwd
    sys.path.insert(0, root)
    from experiment_manager import ExperimentManager
    from src.utils.io import load_config
    from src.utils.non_diffusion_model_manager import MSA
    from src.utils.process_pipeline_inputs.preprocess_em_inputs import (
        create_em_configuration_file_from_baseline, save_metadata)

    pdb_id = args.run_name.split("_")[0]
    assembly = f"{args.run_name}_unguided_msa"
    inputs, outputs = "pipeline_inputs_unguided", "pipeline_outputs_unguided"
    ref_pdb = os.path.join(inputs, "pdbs", pdb_id, f"{args.run_name}_ref.pdb")
    sequence = write_reference_pdb(md_pdb, ref_pdb)
    print(f"{args.run_name}: {len(sequence)} residues")

    save_metadata(pdb_id, assembly, [sequence], [1], ["proteinChain"], inputs)
    config = load_config(create_em_configuration_file_from_baseline(
        pdb_id, [sequence], [1], ["proteinChain"], None, ref_pdb, assembly, inputs, outputs,
        None, None, "pipeline_configurations/cryo_baseline.yaml"))
    config.general.name = assembly
    config.general.batch_size = args.batch_size
    config.protein.residue_range = [[1, 2]]
    # no guidance: nothing but the diffusion model and the subsampled MSA
    config.general.apply_diffusion_guidance = False
    config.loss_function.loss_function_type = []
    config.loss_function.violation_loss_weight = 0
    config.loss_function.bond_length_loss_weight = 0
    config.loss_function.cryoesp_loss_function.log_bond_length_loss = False
    config.model_manager.dump_dir = f"{config.model_manager.dump_dir}_unguided_shard{args.shard}"
    if args.diffusion_N:
        config.model_manager.diffusion_N = args.diffusion_N

    ExperimentManager.seed_experiment(args.seed + args.shard)
    pipeline = ExperimentManager(config, args.device)
    mm = pipeline.model_manager
    pipeline.loss_function = None

    full_feat = dict(mm.input_feature_dict)
    bb = backbone_selection(mm.atom_array)
    n_sub = math.ceil(args.n_structures / args.batch_size)
    todo = list(range(n_sub))[args.shard::args.nshards]
    os.makedirs(out_dir, exist_ok=True)
    print(f"full MSA depth {full_feat['msa'].shape[0]}, using {args.msa_depth}; shard "
          f"{args.shard}/{args.nshards}: {len(todo)} subsamples x {args.batch_size} structures")

    t0, done = time.time(), 0
    for k, sub in enumerate(todo):
        out = os.path.join(out_dir, f"sub_{sub:06d}.pdb")
        if os.path.exists(out):
            continue
        feat, depth = subsample_msa_features(full_feat, args.msa_depth, args.seed + 1000003 * sub)
        recompute_trunk(mm, feat, depth, MSA)
        structures = pipeline.run_full_diffusion_process(pipeline.get_initial_latents())
        write_models(out, structures.detach()[:, bb, :].cpu().numpy(), mm.atom_array, bb)
        done += 1
        el = time.time() - t0
        print(f"[{k + 1}/{len(todo)}] sub {sub}: {el / done:.1f}s per subsample, "
              f"{(len(todo) - (k + 1)) * el / done / 3600:.2f}h left", flush=True)


if __name__ == "__main__":
    main()
