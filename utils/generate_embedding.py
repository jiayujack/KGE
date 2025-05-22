import os
import sys
import math
import time
import json
import zlib
import logging
import requests
import numpy as np
import subprocess 
from rdkit import Chem
from rdkit.Chem import AllChem
import re

# Ensure required packages are installed (UniMol and Biopython)
try:
    import unimol_tools
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "unimol_tools", "huggingface_hub", "biopython"], check=True)
    import unimol_tools
from unimol_tools import UniMolRepr
from Bio.PDB import MMCIFParser  # for parsing mmCIF if needed

# Configure logging for debug information
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Constants and configuration
UNIPROT_BATCH_SIZE = 100000   # Max IDs per UniProt mapping API request (100k)
POCKET_EMBED_BATCH = 16       # Batch size for pocket embedding
MOLECULE_EMBED_BATCH = 64     # Batch size for molecule embedding
STRUCTURE_TIMEOUT = 30        # Timeout (seconds) for structure downloads
CHEBI_TIMEOUT = 5             # Timeout (seconds) for ChEBI requests
OUTPUT_DIR = "embeddings"

# Prepare output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize a single session for HTTP requests with retry logic
session = requests.Session()
# Set up retry strategy for transient errors (HTTP 5xx)
adapter = requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(
    total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]
))
session.mount("http://", adapter)
session.mount("https://", adapter)


def map_protein_names_to_uniprot(protein_names):
    """
    Map a list of protein entry names to UniProt accessions using UniProt ID Mapping API.
    Returns a dict: {protein_name: uniprot_id} and a list of names that failed to map.
    """
    mapping_result = {}
    failed_names = []

    # UniProt ID Mapping API endpoints
    API_URL = "https://rest.uniprot.org"
    submit_url = f"{API_URL}/idmapping/run"
    status_url = f"{API_URL}/idmapping/status"
    details_url = f"{API_URL}/idmapping/details"

    def check_response(response):
        """Raise an exception for HTTP errors (logging the error message)."""
        try:
            response.raise_for_status()
        except requests.HTTPError as err:
            logging.error(f"HTTP error: {err} - {response.text}")
            raise

    # Functions for submitting a mapping job and polling for results
    def submit_id_mapping(ids_batch):
        # Submit a mapping job for a batch of IDs (from UniProt ID/AC to UniProtKB)
        data = {
            "from": "UniProtKB_AC-ID",
            "to": "UniProtKB",
            "ids": ",".join(ids_batch)
        }
        response = session.post(submit_url, data=data)
        check_response(response)
        job_id = response.json().get("jobId")
        return job_id

    def wait_for_job(job_id):
        # Poll the status until the job is finished
        while True:
            response = session.get(f"{status_url}/{job_id}")
            check_response(response)
            status = response.json()
            if status.get("jobStatus") and status["jobStatus"] != "FINISHED":
                if status["jobStatus"] == "RUNNING":
                    time.sleep(3)
                    continue
                else:
                    raise Exception(f"ID mapping job failed with status: {status['jobStatus']}")
            # If no "jobStatus" key, it means job is finished and results are ready (or there are failedIds)
            return

    def retrieve_results(job_id):
        # Get the redirect URL for results, then download all results (in JSON format)
        response = session.get(f"{details_url}/{job_id}")
        check_response(response)
        result_url = response.json().get("redirectURL")
        if not result_url:
            return {"results": [], "failedIds": []}
        # We will retrieve in JSON (default) and handle pagination if needed
        all_results = {"results": [], "failedIds": []}
        # Fetch first batch
        response = session.get(result_url)
        check_response(response)
        data = response.json()
        # Combine first batch
        all_results["results"].extend(data.get("results", []))
        all_results["failedIds"].extend(data.get("failedIds", []))
        # Check for pagination via 'Link: next' header
        next_link = None
        if "Link" in response.headers:
            match = re.match(r'<(.+)>; rel="next"', response.headers["Link"])
            if match:
                next_link = match.group(1)
        # Fetch subsequent batches if any
        while next_link:
            response = session.get(next_link)
            check_response(response)
            data = response.json()
            all_results["results"].extend(data.get("results", []))
            all_results["failedIds"].extend(data.get("failedIds", []))
            # Update next_link
            next_link = None
            if "Link" in response.headers:
                match = re.match(r'<(.+)>; rel="next"', response.headers["Link"])
                if match:
                    next_link = match.group(1)
        return all_results

    # Process in batches to respect API limits
    proteins_list = list(protein_names)
    total = len(proteins_list)
    if total == 0:
        return mapping_result, failed_names

    num_batches = math.ceil(total / UNIPROT_BATCH_SIZE)
    logging.info(f"Mapping {total} protein names to UniProt IDs in {num_batches} batch(es)...")

    for i in range(num_batches):
        batch_ids = proteins_list[i * UNIPROT_BATCH_SIZE : (i + 1) * UNIPROT_BATCH_SIZE]
        logging.info(f"Submitting UniProt mapping batch {i+1}/{num_batches} (size={len(batch_ids)})")
        job_id = submit_id_mapping(batch_ids)
        wait_for_job(job_id)
        results_data = retrieve_results(job_id)
        # Parse results
        for item in results_data.get("results", []):
            frm = item.get("from")    # original identifier (entry name)
            to = item.get("to")      # mapped UniProt accession
            if frm and to:
                mapping_result[frm] = to
        # Track any failed mappings
        for fid in results_data.get("failedIds", []):
            failed_names.append(fid)
    if failed_names:
        logging.warning(f"{len(failed_names)} protein names could not be mapped to UniProt.")
    return mapping_result, failed_names


def get_smiles_for_chebi_ids(chebi_ids):
    """
    Retrieve SMILES strings for a list of CHEBI identifiers using the ChEBI web service.
    Returns a dict {chebi_id: smiles} and a list of CHEBI IDs that failed or had no SMILES.
    """
    chebi_to_smiles = {}
    failed_chebis = []
    base_url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity"

    for idx, chebi_id in enumerate(chebi_ids, start=1):
        # CHEBI IDs in the file might or might not have the "CHEBI:" prefix; ensure it is present
        if not chebi_id.startswith("CHEBI:"):
            chebi_id = "CHEBI:" + chebi_id
        try:
            response = session.get(base_url, params={"chebiId": chebi_id}, timeout=CHEBI_TIMEOUT)
        except requests.RequestException as e:
            logging.error(f"Request error fetching ChEBI {chebi_id}: {e}")
            failed_chebis.append(chebi_id)
            continue
        if response.status_code != 200:
            logging.warning(f"Failed to retrieve ChEBI {chebi_id} (status {response.status_code})")
            failed_chebis.append(chebi_id)
            continue
        # Parse the XML response to find the SMILES
        smiles = None
        try:
            # The response is XML; attempt to find the <smiles> element
            text = response.text
            start_idx = text.find("<smiles>")
            end_idx = text.find("</smiles>")
            if start_idx != -1 and end_idx != -1:
                smiles = text[start_idx + len("<smiles>"): end_idx].strip()
        except Exception as e:
            logging.error(f"Error parsing SMILES for {chebi_id}: {e}")
        if not smiles:
            # No SMILES found
            logging.warning(f"No SMILES found for {chebi_id}")
            failed_chebis.append(chebi_id)
        else:
            chebi_to_smiles[chebi_id] = smiles
        # Progress logging every 1000 compounds
        if idx % 1000 == 0:
            logging.info(f"Processed {idx} ChEBI IDs for SMILES")
    return chebi_to_smiles, failed_chebis


def download_protein_structure(uniprot_id):
    """
    Download the AlphaFold structure for the given UniProt ID and return its atomic coordinates.
    Returns (atom_symbols_list, coordinates_list) on success, or (None, None) on failure.
    """
    # Alphafold structure file URL patterns
    base_url = "https://alphafold.ebi.ac.uk/files/"
    pdb_url_v4 = f"{base_url}AF-{uniprot_id}-F1-model_v4.pdb"
    cif_url_v4 = f"{base_url}AF-{uniprot_id}-F1-model_v4.cif"
    pdb_url_v3 = f"{base_url}AF-{uniprot_id}-F1-model_v3.pdb"
    cif_url_v3 = f"{base_url}AF-{uniprot_id}-F1-model_v3.cif"

    # Try URLs in order of preference
    urls_to_try = [
        ("pdb_v4", pdb_url_v4),
        ("cif_v4", cif_url_v4),
        ("pdb_v3", pdb_url_v3),
        ("cif_v3", cif_url_v3)
    ]
    content = None
    content_type = None
    for label, url in urls_to_try:
        try:
            resp = session.get(url, timeout=STRUCTURE_TIMEOUT)
        except requests.RequestException as e:
            logging.error(f"Error downloading {url}: {e}")
            continue
        if resp.status_code == 200:
            content = resp.content
            content_type = label.split("_")[0]  # 'pdb' or 'cif'
            break
        elif resp.status_code == 404:
            # Not found, try next
            continue
        else:
            # Other HTTP error
            logging.warning(f"Failed to download {url} (status {resp.status_code})")
            continue

    if content is None:
        # All attempts failed
        logging.warning(f"No AlphaFold structure found for {uniprot_id}")
        return None, None

    # Parse structure depending on format
    atoms = []
    coords = []
    if content_type == "pdb":
        text = content.decode('utf-8', errors='ignore')
        for line in text.splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Columns: [12-15] atom name, [17-19] residue name, [21] chain, [22-25] res seq, [30-37] x, [38-45] y, [46-53] z, [76-77] element
                element = line[76:78].strip()
                if element == "":  # If element column is blank, derive from atom name
                    atom_name = line[12:16].strip()
                    element = atom_name[0:2].strip()  # try first two chars
                    if len(element) == 2 and element[1].islower():
                        # e.g. Ca for calcium would appear as 'CA' in PDB, not 'Ca', so second char likely not lowercase in this case
                        pass
                    # If still not a valid element symbol, fall back to first letter
                    if element not in ["C","N","O","S","P","H","MG","ZN","CA","NA","CL","FE","SE"]:
                        element = atom_name[0]
                # Skip hydrogens (AlphaFold structures typically omit hydrogens)
                if element.upper() == "H":
                    continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                atoms.append(element)
                coords.append([x, y, z])
    elif content_type == "cif":
        # Use Biopython MMCIFParser to parse coordinates
        cif_parser = MMCIFParser(QUIET=True)
        try:
            structure_id = uniprot_id
            # Parse from memory (Biopython can parse file-like objects)
            from io import BytesIO
            structure = cif_parser.get_structure(structure_id, BytesIO(content))
        except Exception as e:
            logging.error(f"Biopython mmCIF parse error for {uniprot_id}: {e}")
            return None, None
        for atom in structure.get_atoms():
            elem = atom.element
            # Biopython may denote unknown element as "" or X, handle that if necessary
            if not elem:
                elem = atom.get_name()[0]  # fallback to first letter of atom name
            if elem.upper() == "H":
                continue
            atoms.append(elem)
            pos = atom.get_coord()
            coords.append([float(pos[0]), float(pos[1]), float(pos[2])])
    else:
        return None, None

    if not atoms:
        logging.warning(f"No heavy atoms parsed for {uniprot_id} (structure might be empty or H-only)")
    return atoms, coords


def generate_3d_conformer(smiles):
    """
    Generate a 3D conformation for a molecule given its SMILES using RDKit.
    Returns (atom_symbols_list, coordinates_list) for heavy atoms, or (None, None) on failure.
    """
    # RDKit molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"RDKit failed to parse SMILES: {smiles}")
        return None, None
    # Add hydrogens (explicit) for geometry
    mol = Chem.AddHs(mol)
    # Embed conformation
    embed_ok = False
    for attempt in range(3):
        # Use ETKDG algorithm with a random seed each attempt
        params = AllChem.ETKDGv3()
        params.randomSeed = int(time.time() % 1e6)  # different seed
        if AllChem.EmbedMolecule(mol, params=params) == 0:
            embed_ok = True
            break
    if not embed_ok:
        # Try a basic distance geometry (ETDG) as a fallback
        if AllChem.EmbedMolecule(mol, AllChem.ETDG()) != 0:
            logging.warning(f"Failed to generate 3D conformation for SMILES: {smiles}")
            return None, None
    # Optimize geometry with force field (MMFF or UFF)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass  # ignore optimization failures
    # Extract coordinates for heavy atoms (exclude hydrogens)
    conf = mol.GetConformer()
    atoms = []
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append(atom.GetSymbol())
            coords.append([pos.x, pos.y, pos.z])
    if not atoms:
        logging.warning(f"No heavy atoms in molecule (SMILES: {smiles}) or only hydrogens present.")
    return atoms, coords


def main():
    # Read unique protein names and CHEBI IDs from train.txt
    protein_names = set()
    chebi_ids = set()
    input_file = "/home/jiayu/KGE/data/data_UNIPROT_CHEBI/train.txt"
    with open(input_file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) != 3:
                continue
            prot_name, relation, chebi_id = cols
            # We expect relation to be "catalytic_activity_CHEBI"
            # Collect unique IDs
            protein_names.add(prot_name)
            chebi_ids.add(chebi_id)
    logging.info(f"Found {len(protein_names)} unique proteins and {len(chebi_ids)} unique CHEBI IDs in the input file.")

    # Step 2: Map protein names to UniProt accessions
    prot_to_uniprot, failed_prots = map_protein_names_to_uniprot(protein_names)
    logging.info(f"Mapped {len(prot_to_uniprot)} protein names to UniProt IDs. Failed mappings: {len(failed_prots)}")

    # Step 3: Map CHEBI IDs to SMILES
    chebi_to_smiles, failed_chebis = get_smiles_for_chebi_ids(chebi_ids)
    logging.info(f"Retrieved SMILES for {len(chebi_to_smiles)} CHEBI IDs. Failed/No SMILES: {len(failed_chebis)}")

    # Step 4: Load pretrained Uni-Mol models for molecule and pocket representations
    logging.info("Loading Uni-Mol pretrained models for molecule and pocket...")
    try:
        pocket_model = UniMolRepr(data_type='protein', model_name='unimolv1')
    except Exception:
        # In some versions, data_type might be 'pocket'
        pocket_model = UniMolRepr(data_type='pocket', model_name='unimolv1')
    molecule_model = UniMolRepr(data_type='molecule', model_name='unimolv1')
    # If a GPU is available, move models to GPU for faster processing
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pocket_model.model.to(device)
        molecule_model.model.to(device)
    except AttributeError:
        pass  # If unimol_tools doesn't expose .model, it will use device automatically if available

    # Step 5 & 6: Convert structures to Uni-Mol input format, get embeddings, and save .npy files
    # Process protein structures in batches
    logging.info("Downloading protein structures and extracting pocket embeddings...")
    batch_atoms = []
    batch_coords = []
    batch_ids = []
    count = 0
    for prot_name, uni_id in prot_to_uniprot.items():
        atoms, coords = download_protein_structure(uni_id)
        if atoms is None or coords is None or len(atoms) == 0:
            continue  # skip if structure not available or no heavy atoms
        batch_atoms.append(atoms)
        batch_coords.append(np.array(coords, dtype=np.float32))
        batch_ids.append(uni_id)
        count += 1
        # If batch is full, run the pocket model
        if len(batch_atoms) >= POCKET_EMBED_BATCH:
            data_batch = {'atoms': batch_atoms, 'coordinates': batch_coords}
            reprs = pocket_model.get_repr(data_batch)
            cls_list = reprs['cls_repr']
            for idx, emb in enumerate(cls_list):
                uni_id_out = batch_ids[idx]
                np.save(os.path.join(OUTPUT_DIR, f"{uni_id_out}.npy"), np.array(emb, dtype=np.float32))
            # Reset batch lists
            batch_atoms.clear()
            batch_coords.clear()
            batch_ids.clear()
        # Progress log
        if count % 100 == 0:
            logging.info(f"Processed {count} protein structures")
    # Process any remaining proteins in the last batch
    if batch_atoms:
        data_batch = {'atoms': batch_atoms, 'coordinates': batch_coords}
        reprs = pocket_model.get_repr(data_batch)
        cls_list = reprs['cls_repr']
        for idx, emb in enumerate(cls_list):
            uni_id_out = batch_ids[idx]
            np.save(os.path.join(OUTPUT_DIR, f"{uni_id_out}.npy"), np.array(emb, dtype=np.float32))
    logging.info(f"Saved CLS embeddings for {count} proteins (UniProt IDs).")

    # Process molecules (SMILES) in batches
    logging.info("Generating 3D conformers and extracting molecule embeddings...")
    batch_atoms = []
    batch_coords = []
    batch_ids = []
    count = 0
    for chebi_id, smiles in chebi_to_smiles.items():
        atoms, coords = generate_3d_conformer(smiles)
        if atoms is None or coords is None or len(atoms) == 0:
            continue  # skip if conformer generation failed or molecule has no heavy atoms
        batch_atoms.append(atoms)
        batch_coords.append(np.array(coords, dtype=np.float32))
        # Clean file name: use CHEBIID without colon for output file
        file_id = chebi_id.replace("CHEBI:", "CHEBI_")
        batch_ids.append(file_id)
        count += 1
        if len(batch_atoms) >= MOLECULE_EMBED_BATCH:
            data_batch = {'atoms': batch_atoms, 'coordinates': batch_coords}
            reprs = molecule_model.get_repr(data_batch)
            cls_list = reprs['cls_repr']
            for idx, emb in enumerate(cls_list):
                np.save(os.path.join(OUTPUT_DIR, f"{batch_ids[idx]}.npy"), np.array(emb, dtype=np.float32))
            batch_atoms.clear()
            batch_coords.clear()
            batch_ids.clear()
        if count % 1000 == 0:
            logging.info(f"Processed {count} molecules")
    if batch_atoms:
        data_batch = {'atoms': batch_atoms, 'coordinates': batch_coords}
        reprs = molecule_model.get_repr(data_batch)
        cls_list = reprs['cls_repr']
        for idx, emb in enumerate(cls_list):
            np.save(os.path.join(OUTPUT_DIR, f"{batch_ids[idx]}.npy"), np.array(emb, dtype=np.float32))
    logging.info(f"Saved CLS embeddings for {count} molecules (ChEBI compounds).")

    logging.info("Embedding extraction complete. Embeddings saved as .npy files.")

if __name__ == "__main__":
    main()
