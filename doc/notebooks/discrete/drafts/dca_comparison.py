from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import fem, sys, os, time, Bio.PDB, nglview, warnings
from Bio import BiopythonWarning
data_dir = '../../../../data/msa'
sys.path.append(data_dir)
from parse_pfam import parse_pfam

warnings.simplefilter('ignore', BiopythonWarning)
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
pfam, pdb_refs = parse_pfam(data_dir)
pfam['size'] = pfam['res'] * pfam['seq']
pfam.sort_values(by='size', ascending=False, inplace=True)
pdb_refs['res'] = pdb_refs['pdb_end'] - pdb_refs['pdb_start'] + 1
pdb_refs.sort_values(by='res', ascending=False, inplace=True)
try:
    os.makedirs(os.path.join(data_dir, 'dca_comparison', 'dca'))
    os.makedirs(os.path.join(data_dir, 'dca_comparison', 'fem'))
except:
    pass
cache = False

# seed
# PF00691
# too big
# BPD_transp_1 PF00528
# HATPase PF02518
# LysR_substrate PF03466
# Radical_SAM PF04055
# Response_reg PF00072
# SBP_bac_5  PF00496
# Sigma70_r2 PF04542
# TonB_dep_Rec PF00593

dca_msas = [
    'PF00015', 'PF00034', 'PF00037', 'PF00079', 'PF00126', 'PF00142',
    'PF00151', 'PF00158', 'PF00165', 'PF00239', 'PF00296', 'PF00353',
    'PF00356', 'PF00376', 'PF00384', 'PF00392', 'PF00437', 'PF00440',
    'PF00465', 'PF00480', 'PF00483', 'PF00486', 'PF00497', 'PF00512',
    'PF00529', 'PF00532', 'PF00534', 'PF00535', 'PF00563', 'PF00589',
    'PF00590', 'PF00691', 'PF00805', 'PF00881', 'PF00903', 'PF00905',
    'PF00990', 'PF00994', 'PF01011', 'PF01022', 'PF01032', 'PF01041',
    'PF01047', 'PF01128', 'PF01368', 'PF01380', 'PF01381', 'PF01420',
    'PF01464', 'PF01497', 'PF01547', 'PF01551', 'PF01555', 'PF01568',
    'PF01584', 'PF01609', 'PF01613', 'PF01614', 'PF01627', 'PF01638',
    'PF01656', 'PF01695', 'PF01751', 'PF01872', 'PF01895', 'PF01966',
    'PF01978', 'PF02082', 'PF02195', 'PF02254', 'PF02272', 'PF02310',
    'PF02321', 'PF02384', 'PF02491', 'PF02515', 'PF02525', 'PF02627',
    'PF02811', 'PF02899', 'PF02954', 'PF03358', 'PF03401', 'PF03459',
    'PF03551', 'PF03602', 'PF03721', 'PF03734', 'PF03793', 'PF03989',
    'PF04545', 'PF05016', 'PF07238', 'PF07244', 'PF07676', 'PF07702',
    'PF07715', 'PF07729', 'PF08245', 'PF08279', 'PF08281', 'PF08402',
    'PF08443', 'PF08447', 'PF09084', 'PF09278', 'PF09339', 'PF10531'
]
# + [
#     'PF00528', 'PF02518', 'PF03466', 'PF04055', 'PF00072', 'PF00496',
#     'PF04542', 'PF00593'
# ]

size_ranks = []
for ac in dca_msas:
    fam = pfam.loc[ac]
    rank = pfam['size'].rank(ascending=False)
    size_ranks.append(rank[fam.name].astype(int))

order = np.argsort(size_ranks)[::-1]
dca_msas = np.array(dca_msas)[order][:5]

info = []

for ac in dca_msas:

    print('Pfam ac:', ac)

    fam = pfam.loc[ac]

    fam_dir = os.path.join(data_dir, 'Pfam-A.full', fam.name)
    msa = np.load(os.path.join(fam_dir, 'msa.npy'))
    print('# residues, # sequences:', msa.shape)

    size_rank = pfam['size'].rank(ascending=False)[fam.name].astype(int)
    print('size rank: %i of %i' % (size_rank, pfam.shape[0]))

    aa = np.array([np.unique(s) for s in msa])
    one_aa = np.array([len(a) == 1 for a in aa])
    two_aa = np.array([len(a) == 2 for a in aa])
    missing_aa_res = np.array(['-' in a for a in aa])
    conserved_residues = one_aa | (two_aa & missing_aa_res)

    m = np.array([len(a) for a in aa])
    m = m[~conserved_residues]
    n_residues = m.shape[0]

    # compute direct info
    w_file = os.path.join(data_dir, 'dca_comparison', 'fem',
                          '%s_w.npy' % (ac, ))
    disc_file = os.path.join(data_dir, 'dca_comparison', 'fem',
                             '%s_disc.npy' % (ac, ))
    time_file = os.path.join(data_dir, 'dca_comparison', 'fem',
                             '%s_time.npy' % (ac, ))
    if cache and os.path.exists(w_file) and os.path.exists(disc_file):
        w = np.load(w_file)
        disc = np.load(disc_file)
        start, end = np.load(time_file)
    else:
        start = time.time()
        model = fem.discrete.model()
        svd = 'exact' if size_rank > pfam.shape[0] / 3 else 'approx'
        model.fit(msa[~conserved_residues], iters=10, overfit=False, svd=svd)
        w = np.hstack(model.w.values())
        disc = model.disc
        end = time.time()
        np.save(w_file, w)
        np.save(disc_file, disc)
        np.save(time_file, np.array([start, end]))
    disc_mean = np.mean([d[-1] for d in disc])
    iters = np.mean([len(d) for d in disc])
    print('discrepancy, iters:', disc_mean, iters)
    print('fit time: %.02f sec' % (end - start, ))

    direct_info_file = os.path.join(data_dir, 'dca_comparison', 'fem',
                                    '%s_di.npy' % (ac, ))
    if cache and os.path.exists(direct_info_file):
        direct_info = np.load(direct_info_file)
    else:
        mm = np.insert(m.cumsum(), 0, 0)
        w_idx = np.vstack((mm[:-1], mm[1:])).T
        direct_info = np.zeros((n_residues, n_residues))
        for i, ii in enumerate(w_idx):
            for j, jj in enumerate(w_idx):
                p = np.exp(w[ii[0]:ii[1], jj[0]:jj[1]])
                pi, pj = p.sum(axis=1), p.sum(axis=0)
                p /= p.sum()
                pi /= pi.sum()
                pj /= pj.sum()
                direct_info[i, j] = (p * np.log(p / np.outer(pi, pj))).sum()
        np.save(direct_info_file, direct_info)

    info.append([
        ac, msa.shape[0], msa.shape[1], size_rank, disc_mean, iters,
        end - start
    ])

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for d in disc:
        ax[0].plot(d, 'k-', lw=0.1)
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('discrepancy')
    scale = 1e-1 * np.abs(w).max()
    ax[1].matshow(w, cmap='seismic', vmin=-scale, vmax=scale)
    ax[1].set_title('Pfam: %s' % (fam.name, ))
    scale = 1e-1 * np.abs(direct_info).max()
    ax[2].matshow(direct_info, cmap='seismic', vmin=0, vmax=scale)
    ax[2].set_title('direct info')
    for a in ax[1:]:
        a.axis('off')
    plt.savefig(
        os.path.join(data_dir, 'dca_comparison', 'fem', '%s.png' % (ac, )))
    plt.close()

    refs = pdb_refs[pdb_refs.index.str.contains(fam.name)]

    def contact_map(ref, dist_thresh=8):
        seq = msa[:, ref.seq]
        pdb_file = pdb_list.retrieve_pdb_file(
            ref.pdb_id, pdir=fam_dir, file_format='pdb')
        chain = pdb_parser.get_structure(ref.pdb_id, pdb_file)[0][ref.chain]
        coords = np.array([a.get_coord() for a in chain.get_atoms()])
        coords = coords[ref.pdb_start - 1:ref.pdb_end]
        missing_aa_seq = seq == '-'
        # if coords.shape[0] > conserved_residues.shape[0]:
        if coords.shape[0] == conserved_residues[~missing_aa_seq].shape[0]:
            coords = coords[~conserved_residues[~missing_aa_seq]]
            return distance_matrix(coords, coords) < dist_thresh

    def roc(x, c, k=5):
        mask = np.triu(np.ones(x.shape[0], dtype=bool), k=k)
        order = x[mask].argsort()[::-1]
        c_flat = c[mask][order]
        tp = np.cumsum(c_flat, dtype=float)
        fp = np.cumsum(~c_flat, dtype=float)
        tp /= tp[-1]
        fp /= fp[-1]
        return fp, tp

    dca = pd.read_csv(
        os.path.join(data_dir, 'dca_comparison', 'dca',
                     '%s_full_dca.txt' % (ac, )),
        sep=' ',
        names=['res1', 'res2', 'MI', 'DI'],
        header=None)
    n = dca.res2.max()
    dca.res1 -= 1
    dca.res2 -= 1

    dca_mi = coo_matrix((dca.MI, (dca.res1, dca.res2)), shape=(n, n)).toarray()
    dca_di = coo_matrix((dca.DI, (dca.res1, dca.res2)), shape=(n, n)).toarray()

    print dca_mi.shape, dca_di.shape, direct_info.shape

    for i in range(refs.shape[0]):
        ref = refs.iloc[i]
        seq = msa[:, ref.seq]
        missing_aa_seq = seq == '-'

        idx = np.arange((~conserved_residues).sum())
        idx = idx[~missing_aa_seq[~conserved_residues]]

        contacts = contact_map(ref)
        if contacts is None:
            continue

        for x in [dca_mi, dca_di, direct_info]:
            di = x[np.ix_(idx, idx)]
            fp, tp = roc(di, contacts)
            auc = tp.sum() / tp.shape[0]

            print('auc:', auc)

np.save(os.path.join(data_dir, 'dca_comparison', 'fem', 'info.npy'), info)
