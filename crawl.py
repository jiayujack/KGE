import requests
import xml.etree.ElementTree as ET
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import time

# 读取输入文件并提取ID
def read_input_file(filename):
    uniprot_ids = set()
    chebi_ids = set()
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            uniprot_id = parts[0]
            chebi_full = parts[2]
            chebi_id = chebi_full.split(':')[1]
            uniprot_ids.add(uniprot_id)
            chebi_ids.add(chebi_id)
    return uniprot_ids, chebi_ids

# 获取UniProt序列
def get_uniprot_sequences(uniprot_ids):
    sequences = {}
    for uid in uniprot_ids:
        url = f'https://www.uniprot.org/uniprot/{uid}.fasta'
        try:
            response = requests.get(url)
            if response.status_code == 200:
                seq = ''.join(response.text.split('\n')[1:])
                sequences[uid] = seq
            else:
                print(f"UniProt {uid}: 错误码 {response.status_code}")
            time.sleep(0.5)  # 防止请求过快
        except Exception as e:
            print(f"UniProt {uid} 请求失败: {str(e)}")
    return sequences

# 获取ChEBI原子信息
def get_chebi_atoms(chebi_ids):
    atom_info = {}
    for cid in chebi_ids:
        url = f'https://www.ebi.ac.uk/webservices/chebi/2.0/getCompleteEntity?chebiId={cid}'
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"ChEBI {cid}: 错误码 {response.status_code}")
                continue
            
            root = ET.fromstring(response.content)
            # 尝试解析不同命名空间情况
            smiles_elem = None
            for elem in root.iter():
                if 'smiles' in elem.tag:
                    smiles_elem = elem
                    break
            
            if not smiles_elem or not smiles_elem.text:
                print(f"ChEBI {cid}: 无SMILES信息")
                continue
                
            smiles = smiles_elem.text
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)  # 添加氢原子
                atom_counts = defaultdict(int)
                for atom in mol.GetAtoms():
                    atom_counts[atom.GetSymbol()] += 1
                atom_info[cid] = dict(atom_counts)
            else:
                print(f"ChEBI {cid}: 无法解析SMILES")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"ChEBI {cid} 处理失败: {str(e)}")
    return atom_info

# 保存结果
def save_results(sequences, atom_info):
    with open('protein_sequences.txt', 'w') as f:
        for uid, seq in sequences.items():
            f.write(f">{uid}\n{seq}\n\n")
    
    with open('molecule_atoms.txt', 'w') as f:
        for cid, counts in atom_info.items():
            counts_str = ', '.join([f"{k}:{v}" for k, v in counts.items()])
            f.write(f"ChEBI:{cid}\t{counts_str}\n")

if __name__ == "__main__":
    uniprot_ids, chebi_ids = read_input_file("/home/jiayu/KGE/data/data_UNIPROT_CHEBI/val.txt")
    
    print("获取UniProt序列...")
    sequences = get_uniprot_sequences(uniprot_ids)
    
    print("\n获取ChEBI原子信息...")
    atoms = get_chebi_atoms(chebi_ids)
    
    print("\n保存结果...")
    save_results(sequences, atoms)
    
    print("完成！结果已保存到 protein_sequences.txt 和 molecule_atoms.txt")