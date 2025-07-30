import os
import json

import numpy as np
from rdkit import Chem
from scipy.linalg import fractional_matrix_power


PARAMETER_DIR = os.path.join(os.path.dirname(__file__), "parameters")

with open(os.path.join(PARAMETER_DIR, "nhb.json"), "rb") as f:
    nhb_param = json.load(f)
with open(os.path.join(PARAMETER_DIR, "oh.json"), "rb") as f:
    oh_param = json.load(f)
with open(os.path.join(PARAMETER_DIR, "ot.json"), "rb") as f:
    ot_param = json.load(f)
with open(os.path.join(PARAMETER_DIR, "vol.json"), "rb") as f:
    vol_param = json.load(f)
with open(os.path.join(PARAMETER_DIR, "exceptions.json"), "rb") as f:
    chemical_exceptions = json.load(f)


_q0 = 79.53  # area normalization parameter [Å**2]
_r0 = 66.69  # volume normalization parameter [Å**3]
_z = 10  # coordination number
_R = 1.987204258e-3  # gas constant [kcal/K/mol]
_AES = 6525.69  # electrostatic constant A [kcal*ang**4/mol/e**2]
_BES = 1.4859e8  # electrostatic constant B [kcal*Å**4*K**2/mol/e**2]
_aeff = 7.25  # effective area [Å**2], number of sigma profiles,
_chb = np.array(
    [[0, 0, 0], [0, 4013.78, 3016.43], [0, 3016.43, 932.31]]
)  # hydrogen bonding parameter [kcal*Å^4/mol/e^2],
_cES = lambda T: _AES + _BES / T / T  # electrostatic parameter [kcal*Å^4/mol/e^2]
subgroups = [
    "[CX4H3]",
    "[CX3H2v4]",
    "[CX2H1v4]",
    "[!R;CX4H2]",
    "[!R;CX4H]",
    "[!R;CX4H0]",
    "[!R;CX3H1v4]",
    "[!R;CX3H0v4]",
    "[!R;CX2H0;$([CX2H0](=*)(=*))]",
    "[!R;CX2H0;$([CX2H0](#*))]",
    "[R;CX4H2]",
    "[R;CX4H]",
    "[R;CX4H0]",
    "[R;CX3H1v4]",
    "[R;CX3H0v4]",
    "[R;CX2H0;$([CX2H0](=*)(=*))]",
    "[R;CX2H0;$([CX2H0](#*))]",
    "[R;cX4h2]",
    "[R;cX4h]",
    "[R;cX3h1v4]",
    "[R;cX3h0v4]",
    "[FX1H0]",
    "[ClX1H0]",
    "[BrX1H0]",
    "[IX1H0]",
    "[OX2H1]",
    "[!R;OX2H0]",
    "[R;OX2H0]",
    "[R;oX2h0]",
    "[OX1H0v2]",
    "[NX3H2v3]",
    "[NX3H1v3;!R]",
    "[NX3H1v3;R]",
    "[nX3h1v3;R]",
    "[NX3H0v3;!R]",
    "[NX3H0v3;R]",
    "[nX3h0v3;R]",
    "[NX2H0v3;!R]",
    "[NX2H0v3;R]",
    "[nX2h0v3;R]",
    "[NX1H0v3]",
    "[SX2H1v2]",
    "[SX2H0v2;!R]",
    "[SX2H0v2;R]",
    "[sX2h0v2;R]",
    "[SX1H0v2]",
    "[OX1H0v1]",
    "[NX3H0v4]",
]


def _get_GCGCN_input(SMILES):
    mol = Chem.MolFromSmiles(SMILES)

    # Get node feature matrix
    nfm = np.zeros((25, len(subgroups)))
    for smarts_index, smarts in enumerate(subgroups):
        pat = Chem.MolFromSmarts(smarts)
        nfm[mol.GetSubstructMatches(pat), smarts_index] = 1

    # Get edge feature matrix
    efm = Chem.GetAdjacencyMatrix(mol).astype("float64")
    np.fill_diagonal(efm, 1)
    diag = np.diag(np.sum(efm, axis=1))
    diag_half = fractional_matrix_power(diag, -0.5)
    efm = np.matmul(np.matmul(diag_half, efm), diag_half)
    n_heavyatom = len(efm)
    pad = 25 - n_heavyatom
    efm = np.pad(efm, ((0, pad), (0, pad)), "constant", constant_values=0.0)

    return nfm, efm


def _GCGCN_model(nfm, efm, param_list):
    x = np.dot(efm, nfm)
    x = np.dot(x, param_list[0]) + param_list[1]
    x = np.where(x > 0, x, 0)

    x = np.dot(efm, x)
    x = np.dot(x, param_list[2]) + param_list[3]
    x = np.where(x > 0, x, 0)

    x = x.reshape(-1)
    x = np.dot(x, np.tile(np.eye(256), (25, 1)))

    x = np.dot(x, param_list[4]) + param_list[5]
    x = np.where(x > 0, x, 0)

    x = np.dot(x, param_list[6]) + param_list[7]
    x = np.where(x > 0, x, 0)

    x = np.dot(x, param_list[8]) + param_list[9]
    x = np.where(x > 0, x, 0)

    x = np.dot(x, param_list[10]) + param_list[11]

    return x


def _cal_DW(T):
    """
    Calculate the exchange energy.

    The exchange energy has the values for each charge density combinations
    and sigma profile type combinations, therefore having the shape of
    (num_sp, num_sp, 51, 51).

    Parameters
    ----------
    T : float
        The system temperature.

    Returns
    -------
    DW : numpy.ndarray of shape=(num_sp, num_sp, 51, 51)
        The exchange energy.
    """
    # Initialize parameters
    sig = np.linspace(-0.025, 0.025, 51)
    sigT = sig.reshape(-1, 1)
    DW = np.zeros((3, 3, 51, 51))

    # Calculate exchange energy for each pair of sigma profile types
    for i in range(3):
        for j in range(i + 1):
            mask = (sig * sigT) < 0
            chb_part = np.where(mask, _chb[i, j] * (sig - sigT) ** 2, 0)

            # Calculate total exchange energy
            DW[i, j] = DW[j, i] = _cES(T) * (sig + sigT) ** 2 - chb_part

    return DW


def _get_atom_type(atom, bond):
    """Get hybridization and sigma profile types for each atom.

    The dispersive natures are as below.
    DSP_WATER : WATER in this code. This indicates water.
    DSP_COOH : COOH in this code. This indicates a molecule with a carboxyl
    group.
    DSP_HB_ONLY_ACCEPTOR : HBOA in this code. The molecule contains any of
    the atoms O,N, or F but no H atoms bonded to any of these O, N, or F.
    DSP_HB_DONOR_ACCEPTOR : HBDA in this code. The molecule contains any of
    the functional groups NH, OH, or FH (but not OH of COOH or water).
    DSP_NHB : NHB in this code. This indicates that the molecule is non-
    hydrogen-bonding.

    The dispersion types are as below.
    C(sp3) : C bonded to 4 others.
    C(sp2) : C bonded to 3 others.
    C(sp) : C bonded to 2 others.
    N(sp3) : N bonded to three others.
    N(sp2) : N bonded to two others.
    N(sp) : N bonded to one other.
    -O- : O(sp3) in this code. O bonded to 2 others.
    =O : O(sp2) in this code. Double-bonded O.
    F : F bonded to one other.
    Cl : Cl bonded to one other.
    H(water) : H in water.
    H(OH) : H-O bond but not water.
    H(NH) : H bonded to N.
    H(other) : H otherwise.
    other : Undifined.

    The hydrogen-bonding types are as below.
    OH : if the atom is O and is bonded to an H, or vice versa.
    OT : if the atom is O and is bonded to an atom other than H, or if the
    atom is H and is bonded to N or F.
    COOH : if the atoms are C, O, H and are in the carboxyl group.
    NHB : otherwise.

    Parameters
    ----------
    atom : numpy.ndarray of shape=(num_atom,)
        Atom symbols sorted by index in the cosmo file.
    bond : numpy.ndarray of shape=(num_atom, num_atom)
        The bond matrix. If two atoms are bonded, their entry is 1, else 0.

    Returns
    -------
    dtype : list of shape=(num_atom,)
        The dispersion type for each atom.
    stype : list of shape=(num_atom,)
        The hydrogen-bonding type for each atom.
    dnatr : {"NHB", "HBOA", "HBDA", "WATER", "COOH"}
        The dispersive nature of the molecule.
    """
    dtype = ["other"] * len(atom)  # hybridization type
    stype = ["NHB"] * len(atom)  # sigma profile type
    dnatr = "NHB"  # dispersive nature of molecule
    dntype = set()  # dispersive nature type of atoms

    # {atom type: {bonded atoms: (dtype, stype, dnatr), ...}, ...}
    # This assumes that all atoms are belong to NHB, OT and H(other).
    atom_prop = {
        "C": {
            2: ("C(sp)", "NHB", "NHB"),
            3: ("C(sp2)", "NHB", "NHB"),
            4: ("C(sp3)", "NHB", "NHB"),
        },
        "O": {
            1: ("O(sp2)", "OT", "HBOA"),
            2: ("O(sp3)", "OT", "HBOA"),
        },
        "N": {
            1: ("N(sp)", "OT", "HBOA"),
            2: ("N(sp2)", "OT", "HBOA"),
            3: ("N(sp3)", "OT", "HBOA"),
        },
        "F": {1: ("F", "OT", "HBOA")},
        "Cl": {1: ("Cl", "NHB", "NHB")},
        "H": {1: ("H(other)", "NHB", "NHB")},
    }

    for i, atom_type in enumerate(atom):
        # Get dictionary of index and atom types bonded with atom i
        ard_i = {j: atom[j] for j in np.flatnonzero(bond[i])}

        # If the atom is in the difined properties
        if atom_type in atom_prop:
            # Get atom types, else get ("Undifined", 0)
            dtype[i], stype[i], dntype_i = atom_prop[atom_type].get(
                len(ard_i), ("other", "NHB", "NHB")
            )
            dntype.add(dntype_i)

        # Find H near N, and renew the types of H
        if atom_type == "H" and "N" in ard_i.values():
            dtype[i] = "H(NH)"
            stype[i] = "OT"
            dntype.add("HBDA")

        # Find H in HF, and renew the types of H
        if atom_type == "H" and "F" in ard_i.values():
            stype[i] = "OT"
            dntype.add("HBDA")

        # Find atom type for -OH, H2O, and COOH
        if atom_type == "H" and "O" in ard_i.values():
            # # Renew the typs of H and O in OH
            # Renew the types of H
            dtype[i] = "H(OH)"
            stype[i] = "OH"

            # Find the atom index of O in OH
            j = list(ard_i.keys())[0]
            ard_j = {k: atom[k] for k in np.flatnonzero(bond[j])}
            # Renew the types of O in -OH
            stype[j] = "OH"
            dntype.add("HBDA")

            # # Further find H-OH and CO-OH
            # if the O in -OH has not two bonds, stop searching
            if len(ard_j) != 2:
                break

            # Find atom index of neighber of O in -OH, but not H in -OH
            k = [k for k in ard_j.keys() if k != i][0]
            ard_k = {m: atom[m] for m in np.flatnonzero(bond[k])}

            # if atom k is H, that is, if the molecule is water, renew the
            # dtype of the Hs in H2O and stop searching
            if atom[k] == "H":
                dtype[i] = "H(water)"
                dtype[k] = "H(water)"
                dntype.add("WATER")
                break

            # # Further find COOH
            # if the atom k is not the C in part of COOH, stop searching
            if not (
                atom[k] == "C"
                and len(ard_k) == 3
                and list(ard_k.values()).count("O") == 2
            ):
                break

            # Find the O, neighber of C in -COH, but not in O in -COH
            m = [m for m in ard_k.keys() if (m != j and ard_k[m] == "O")][0]
            ard_m = {n: atom[n] for n in np.flatnonzero(bond[m])}

            # if the atom m is -O-, not =O, stop searching
            if len(ard_m) != 1:
                break

            # Renew i(H), j(O), k(C) and m(O) as the part of COOH
            dntype.add("COOH")
            stype[i] = "COOH"
            stype[j] = "COOH"
            stype[m] = "COOH"

    # find the dispersive nature of the molecule
    if "HBOA" in dntype:
        dnatr = "HBOA"
    if "HBDA" in dntype:
        dnatr = "HBDA"
    if "WATER" in dntype:
        dnatr = "WATER"
    if "COOH" in dntype:
        dnatr = "COOH"

    return dtype, stype, dnatr


def _get_dsp(dtype):
    """
    Get the dispersive nature of the molecule.

    Parameters
    ----------
    dtype : list of shape=(num_atom,)
        The dispersion type for each atom.

    Returns
    -------
    ek : float
        Dispersive parameter.
    """
    # dispersive parameters
    ddict = {
        "C(sp3)": 115.7023,
        "C(sp2)": 117.4650,
        "C(sp)": 66.0691,
        "N(sp3)": 15.4901,
        "N(sp2)": 84.6268,
        "N(sp)": 109.6621,
        "O(sp3)": 95.6184,  # -O-
        "O(sp2)": -11.0549,  # =O
        "F": 52.9318,
        "Cl": 104.2534,
        "H(water)": 58.3301,
        "H(OH)": 19.3477,
        "H(NH)": 141.1709,
        "H(other)": 0,
    }

    # calculate the dispersive parameter of the molecule
    ek = np.vectorize(ddict.get)(dtype)
    if None in ek:

        return None
    else:
        ek = np.sum(ek) / np.count_nonzero(ek)

    return ek


def calculate_sigma_profile(SMILES: str) -> dict:
    if SMILES in chemical_exceptions:
        area = np.float64(chemical_exceptions[SMILES]["area"])
        volume = np.float64(chemical_exceptions[SMILES]["volume"])
        sigma_profiles = np.array(chemical_exceptions[SMILES]["sigma_profiles"])
    else:
        nfm, efm = _get_GCGCN_input(SMILES)

        volume = 562 * _GCGCN_model(nfm, efm, vol_param)[0]

        sigma_profiles = np.zeros((3, 51))
        sigma_profiles[0] = 145 * _GCGCN_model(nfm, efm, nhb_param)
        sigma_profiles[1] = 7 * _GCGCN_model(nfm, efm, oh_param)
        sigma_profiles[2] = 16 * _GCGCN_model(nfm, efm, ot_param)
        sigma_profiles = np.where(sigma_profiles < 0, 0, sigma_profiles)
        sigma_profiles = sigma_profiles.reshape(1, 3, 51)

        area = np.sum(sigma_profiles)

    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bonds = Chem.GetAdjacencyMatrix(mol)
    bond_type, _, natr = _get_atom_type(atoms, bonds)
    ek = _get_dsp(bond_type)

    return {
        "area": area,
        "volume": volume,
        "sigma_profiles": sigma_profiles,
        "ek": ek,
        "natr": natr,
    }


def cal_ln_gam_comb(A, V, x):
    """
    Calculate log of combinatory activity coefficients.

    Parameters
    ----------
    None.

    Returns
    -------
    ln_gam_comb : numpy.ndarray of shape=(num_comp,)
        Combinatory activity coefficients of components.
    """
    # calculate normalized areas and volumes
    q = A / _q0
    r = V / _r0
    L = (_z / 2) * (r - q) - (r - 1)

    theta = q / np.sum(x * q)
    phi = r / np.sum(x * r)

    # calcualte combinatory activity coefficients
    ln_gam_comb = (
        np.log(phi) + _z * q * np.log(theta / phi) / 2 + L - phi * np.sum(x * L)
    )

    return ln_gam_comb


def cal_ln_gam_res(A, psigA, x, T):
    """
    Calculate residual activity coefficients.

    Parameters
    ----------
    None.

    Returns
    -------
    ln_gam_res : numpy.ndarray of shape=(num_comp,)
        Residual activity coefficients of components.
    """
    # calculate intermediate terms
    psig = np.einsum("itm,i->itm", psigA, 1 / A)
    psig_mix = np.einsum("i,itm->tm", x, psigA) / np.sum(x * A)

    exp_DW = np.exp(-_cal_DW(T) / _R / T)

    A_plus = np.einsum("stmn,isn->istmn", exp_DW, psig)  # A^(+)
    A_plus_mix = np.einsum("stmn,sn->stmn", exp_DW, psig_mix)  # A^(+)_mix

    # calculate the segment activity coefficients
    Gam = np.ones(np.shape(psig))
    Gam_mix = np.ones(np.shape(psig_mix))
    diff = 1

    for _ in range(500):
        Gam_old = np.array(Gam)
        Gam_mix_old = np.array(Gam_mix)

        # Update Gam element-wise
        for i in range(Gam.shape[0]):
            for t in range(Gam.shape[1]):
                for m in range(Gam.shape[2]):
                    Gam[i, t, m] = 1 / np.einsum(
                        "sn,sn->", A_plus[i, :, t, m, :], Gam[i, :, :]
                    )

        # Update Gam_mix element-wise
        for t in range(Gam_mix.shape[0]):
            for m in range(Gam_mix.shape[1]):
                Gam_mix[t, m] = 1 / np.einsum(
                    "sn,sn->", A_plus_mix[:, t, m, :], Gam_mix[:, :]
                )

        # check convergence
        diff = np.sum((Gam - Gam_old) ** 2)
        diff_mix = np.sum((Gam_mix - Gam_mix_old) ** 2)

        if diff <= 1e-6 and diff_mix <= 1e-6:
            break
    else:
        raise Exception("Converge failed")

    # calculate residual activity coefficients
    Gam_part = np.log(Gam_mix) - np.log(Gam)
    ln_gam_res = np.einsum("itm,itm->i", psigA, Gam_part) / _aeff

    return ln_gam_res


def cal_ln_gam_dsp(x, ek, dnatr):
    """
    Calculate dispersive activity coefficients.

    Parameters
    ----------
    None.

    Returns
    -------
    ln_gam_dsp : numpy.ndarray of shape=(num_comp,)
        Dispersive activity coefficients of components.
    """
    num_mol = len(x)

    if None in ek:

        return np.zeros(num_mol)
    elif True in np.isnan(ek):

        return np.zeros(num_mol)

    ekT = ek.reshape(-1, 1)

    # check if dispersion activity coefficients are applicable
    if None in ek or None in dnatr:
        ln_gam_dsp = np.array([0] * num_mol)

        return ln_gam_dsp
    elif True in np.isnan(ek):
        ln_gam_dsp = np.array([0] * num_mol)

        return ln_gam_dsp

    # calculate interaction parameters
    w = np.ones((num_mol, num_mol)) * 0.27027
    wpair = [
        {"WATER", "HBOA"},
        {"COOH", "NHB"},
        {"COOH", "HBDA"},
        {"WATER", "COOH"},
    ]
    for i in range(num_mol):
        for j in range(i):
            if {dnatr[i], dnatr[j]} in wpair:
                w[i][j] = w[j][i] = -0.27027

    A = w * (0.5 * (ek + ekT) - np.sqrt(ek * ekT))  # not area

    # calculate dispersive activity coefficients
    ln_gam_dsp = np.zeros(num_mol)
    for i in range(num_mol):
        for j in range(num_mol):
            if i != j:
                ln_gam_dsp[i] = ln_gam_dsp[i] + x[j] * A[i, j]
            if j > i:
                ln_gam_dsp[i] = ln_gam_dsp[i] - x[i] * x[j] * A[i, j]

    return ln_gam_dsp


def calculate_gamma(chemical_profiles: list, x: list, T: float) -> list:
    """
    Calculate COSMO-SAC activity coefficients.

    Parameters
    ----------
    None.

    Returns
    -------
    gam : list of shape=(num_comp,)
        Activity coefficients of components.
    """
    areas = np.array([])
    volumes = np.array([])
    psigA = np.array([]).reshape(0, 3, 51)
    eks = np.array([])
    natrs = []
    for chemical_profile in chemical_profiles:
        areas = np.append(areas, chemical_profile["area"])
        volumes = np.append(volumes, chemical_profile["volume"])
        psigA = np.vstack((psigA, chemical_profile["sigma_profiles"]))
        eks = np.append(eks, chemical_profile["ek"])
        natrs.append(chemical_profile["natr"])

    ln_gam_comb = cal_ln_gam_comb(areas, volumes, x)
    ln_gam_res = cal_ln_gam_res(areas, psigA, x, T)
    ln_gam_dsp = cal_ln_gam_dsp(x, eks, natrs)

    ln_gam = ln_gam_comb + ln_gam_res + ln_gam_dsp
    gam: np.ndarray = np.exp(ln_gam)

    return gam.tolist()
