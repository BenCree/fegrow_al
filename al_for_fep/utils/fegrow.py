def fegrow(smiles_list, core_sdf='core.sdf', protein_filename='rec_final.pdb', num_conf=100, minimum_conf_rms=0.5, **kwargs):
    # load the common core
    core = Chem.SDMolSupplier(core_sdf)[0]

    # create RList of molecules from smiles
    rlist = RList()
    params = Chem.SmilesParserParams()
    params.removeHs = False # keep the hydrogens
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, params=params)
        rlist.append(RMol(mol))

    # ensure that the common core is indeed that
    for rmol in rlist:
        # check if the core_mol is a substructure
        if not rmol.HasSubstructMatch(core):
            raise Exception('The core molecule is not a substructure of one of the RMols in the RList, '
                            'according to Mol.HasSubstructMatch')
        rmol._save_template(core)

    # conformers and energies
    all_affinities = []
    for mol_id, rmol in enumerate(rlist):
        rmol.generate_conformers(num_conf=num_conf, minimum_conf_rms=minimum_conf_rms)
        rec_final = prody.parsePDB(protein_filename)
        rmol.remove_clashing_confs(rec_final)

        # continue only if there are any conformers to be optimised
        if rmol.GetNumConformers() > 0:
            energies = rmol.optimise_in_receptor(
                receptor_file=protein_filename,
                ligand_force_field="openff",
                use_ani=True,
                sigma_scale_factor=0.8,
                relative_permittivity=4,
                water_model=None,
                platform_name='CPU',
                **kwargs
            )
            rmol.sort_conformers(energy_range=5)

            affinities = rmol.gnina(receptor_file=protein_filename)
            all_affinities.append(affinities)

            with Chem.SDWriter(f'Rmol{mol_id}_best_conformers.sdf') as SDW:
                SDW.write(rmol)

    return all_affinities