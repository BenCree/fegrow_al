# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for running a single cycle of active learning."""

from typing import Sequence
import prody
from rdkit import Chem
from rdkit.Chem import AllChem
import fegrow
from fegrow import RGroups, RLinkers, RList, RMol
from typing import Sequence
import os
import numpy as np
import pandas as pd
from absl import app
from ml_collections.config_flags import config_flags
import sys
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
#import molecule_comparison

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import argparse
from absl import app
from ml_collections.config_flags import config_flags
import os
import pandas as pd
from al_for_fep import single_cycle_lib
feat = 'cnnaff'
gt_df = pd.read_csv('al_for_fep/data/testdata/gen.csv')

ground_max = gt_df[feat].min()


def create_distance_calculator(fps, metric='Tanimoto'):
    def _calculate_distance(i, j):
        if metric == "Tanimoto":
            distance = 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        elif metric == "Dice":
            distance = 1 - DataStructs.DiceSimilarity(fps[i], fps[j])
        elif metric == "Cosine":
            distance = 1 - DataStructs.CosineSimilarity(fps[i], fps[j])
        elif metric == "Sokal":
            distance = 1 - DataStructs.SokalSimilarity(fps[i], fps[j])
        #elif metric == "MCES":
       #     distance =  #ADD MCES
        else:
            raise ValueError("Invalid metric")
        return distance

    return _calculate_distance


def score(smiles_list, core_sdf='core.sdf', protein_filename='rec_final.pdb', num_conf=50, minimum_conf_rms=0.5,
           **kwargs):
    RMol.set_gnina('/home/c0065492/software/gnina')
    # load the common core

    core = Chem.SDMolSupplier(core_sdf)[0]

    # create RList of molecules from smiles
    rlist = RList()
    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, params=params)
        mol = AllChem.AddHs(mol)
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
        #print('generate conformers')
        rmol.generate_conformers(num_conf=num_conf, minimum_conf_rms=minimum_conf_rms)
        rec_final = prody.parsePDB(protein_filename)
        rmol.remove_clashing_confs(rec_final)
        #print(rmol.GetNumConformers())
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
            #print('about to calculate affinity')
            affinities = rmol.gnina(receptor_file=protein_filename)
            all_affinities.append([affinities, smiles_list[mol_id]])
            #print('fegrow output :', all_affinities)
            with Chem.SDWriter(f'{mol_id}_best_conformers.sdf') as SDW:
                SDW.write(rmol)
            #print(all_affinities)

    return all_affinities


def affinity_to_csv(all_affinities, filename='al_for_fep/data/testdata/initial_training_set_test.csv'):
    affs = []

    for i in range(len(all_affinities)):
        affs.append([all_affinities[i][0].iloc[0]['CNNaffinity'], all_affinities[i][1]])
    df = pd.DataFrame(affs, columns=['cnnaff', 'Smiles'])
    df.to_csv(filename)
    print(filename)
    return df


def sample_mols(filename, n_mols, outdir, smiles='Smiles', strategy='MaxMin', metric='Tanimoto', *args,
                **kwargs):  # metric argument goes in here
    """Samples a specified amount of mols from the dataset, using MaxMin (by default).

    Args:
        n_mols: number of molecules to select.
        filename: name of .csv file containing dataset to sample
        strategy: method of sampling molecules from the dataset
         *args:
         **kwargs: Extra arguments not used by this function. These need to be
      included for this parser to satisfy the generic parser interface.
    Returns:
        a list of mols and their cnnaffinities
    """

    df = pd.read_csv(filename)
    smi = df[smiles][:n_mols]
    ms = [Chem.MolFromSmiles(smiles) for smiles in smi]
    i = 0
    while ms.count(None):
        ms.remove(None)
        i += 1
    print('Number of failed simles : ', i)
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2) for m in ms]


    nfps = len(fps)
    _calculate_distance = create_distance_calculator(fps, metric)
    if strategy == "MaxMin":
        picker = MaxMinPicker()
        pickIndices = picker.LazyPick(_calculate_distance, nfps, nfps,
                                      seed=0xFACADE)  # what is the best way to pass the metric argument to _calculate_distance
    elif strategy == "Random":
        array = np.arange(n_mols)  # Create a sequential list of integers
        np.random.shuffle(array)  # Shuffle the array in place
        pickIndices = array.tolist()

    elif strategy == "Sequential":
        pickIndices = range(n_mols)
    else:
        raise ValueError("Invalid strategy")

    picks = [smi[x] for x in pickIndices]
    cnnaff = [df['cnnaff'][x] for x in pickIndices]
    del args, kwargs  # required by the interface, apparently
    assert len(ms) == nfps
    assert len(picks) == nfps
    out = pd.DataFrame({'Smiles': picks, 'cnnaff': cnnaff})
    output_filename = outdir + filename.rsplit('.', 1)[0] + '_training.csv'
    out.to_csv(output_filename)

    return print('Generated training pool : ', output_filename)

_CYCLE_CONFIG = config_flags.DEFINE_config_file(
    name='cycle_config',
    default=None,
    help_string='Location of the ConfigDict file containing experiment specifications.',
    lock_config=True)

dG_maxs = []
dG_means = []
contains_best = []


def generate_training_set(initial_sel, filename):
	print('generating training set')
	smiles_list = list(pd.read_csv(initial_sel)['Smiles'])
	all_affs = score(smiles_list, core_sdf=cycle_config.core_sdf, protein_filename=cycle_config.receptor)
	affinity_to_csv(all_affs, filename=filename)
	print(all_affs)
	print('filename ', filename)
	
def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    #i = 0
    global cycle_config
    cycle_config = _CYCLE_CONFIG.value
    original_cycle_dir = cycle_config.cycle_dir  # Save the original cycle_dir
    print(cycle_config.n_sample_mols)
    #sample_mols(filename='./al_for_fep/data/testdata/gen.csv', n_mols=int(cycle_config.n_sample_mols), smiles='Smiles', outdir='')


    for i in range(cycle_config.epochs):
        print('Cycle : ', i)
        new_dir = f'{original_cycle_dir}_{i}'
        # while os.path.exists(new_dir):  # Check if the directory already exists
        #
        #     new_dir = f'{original_cycle_dir}_{i}'
        prev_dir = f'{original_cycle_dir}_{i - 1 if i > 0 else 0}'
        cycle_config.cycle_dir = new_dir
        os.makedirs(new_dir, exist_ok=True)
        if i == 0:
            generate_training_set('al_for_fep/data/testdata/smiles.csv', filename='al_for_fep/data/testdata/initial_training_set_test.csv')
            single_cycle_lib.MakitaCycle(cycle_config).run_cycle()
            cycle_config.training_pool = prev_dir + '/selection.csv'
            cycle_config.virtual_library = prev_dir + '/virtual_library_with_predictions.csv'
        else:
            single_cycle_lib.MakitaCycle(cycle_config).run_cycle()
            print('running fegrow and generating training set from : ', prev_dir + '/selection.csv')
            generate_training_set(prev_dir + '/selection.csv', filename= new_dir + '/selection_cnn.csv')
            print('writing selection with cnn to ', new_dir)
            cycle_config.training_pool = new_dir + '/selection_cnn.csv'
            cycle_config.virtual_library = new_dir + '/virtual_library_with_predictions.csv'
            print('training pool & virtual library dir for next cycle is : ', new_dir)
        #sel_df = pd.read_csv(new_dir+'/selection.csv')
        #print(sel_df)
        #dG_means.append(sel_df[feat].astype(float).mean())
        #dG_maxs.append(sel_df[feat].astype(float).max())
        #contains_best.append(sel_df["id"].values)




        #print(prev_dir)
        #print(cycle_config.virtual_library)
        print('dG means : ', dG_means)
        print('dG maxes : ', dG_maxs)
        #print('Ground truth max :', ground_max)


if __name__ == '__main__':
    app.run(main)
