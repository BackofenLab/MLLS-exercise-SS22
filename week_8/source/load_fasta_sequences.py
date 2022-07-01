import os
import re
import pandas as pd


################################################################################

def read_fasta_into_dic(fasta_file,
                        convert_to_rna=False,
                        convert_to_dna=True,
                        all_uc=True,
                        skip_n_seqs=False):
    """
    Read in FASTA sequences from file, return dictionary with mapping:
    sequence_id -> sequence

    convert_to_rna:
        Convert input sequences to RNA.
    all_uc:
        Convert all sequence characters to uppercase.
    skip_n_seqs:
        Skip sequences that contain N letters.

    """

    assert os.path.exists(fasta_file), "Given FASTA file \"%s\" not found" %(fasta_file)

    seqs_dic = {}
    seq_id = ""

    with open(fasta_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                assert seq_id not in seqs_dic, "non-unique FASTA header \"%s\" in \"%s\"" % (seq_id, fasta_file)
                seqs_dic[seq_id] = ""
            elif re.search("[ACGTUN]+", line, re.I):
                m = re.search("([ACGTUN]+)", line, re.I)
                if seq_id in seqs_dic:
                    '''if convert_to_rna:
                        seqs_dic[seq_id] += m.group(1).replace("T","U").replace("t","u")'''
                    if convert_to_dna:
                        seqs_dic[seq_id] += m.group(1).replace("U","T").replace("u","t")
                    else:
                        seqs_dic[seq_id] += m.group(1)
                if all_uc:
                    seqs_dic[seq_id] = seqs_dic[seq_id].upper()
    f.closed

    assert seqs_dic, "no sequences read in (input FASTA file \"%s\" empty or mal-formatted?)" %(fasta_file)

    # If sequences with N nucleotides should be skipped.
    c_skipped_n_ids = 0
    if skip_n_seqs:
        del_ids = []
        for seq_id in seqs_dic:
            seq = seqs_dic[seq_id]
            if re.search("N", seq, re.I):
                print ("WARNING: sequence with seq_id \"%s\" in file \"%s\" contains N nucleotides. Discarding sequence ... " % (seq_id, fasta_file))
                c_skipped_n_ids += 1
                del_ids.append(seq_id)
        for seq_id in del_ids:
            del seqs_dic[seq_id]
        assert seqs_dic, "no sequences remaining after deleting N containing sequences (input FASTA file \"%s\")" %(fasta_file)
        if c_skipped_n_ids:
            print("# of N-containing sequences discarded:  %i" %(c_skipped_n_ids))

    return seqs_dic


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


def read_seqs(seq_dict, label, k=6, columns=["sequence", "label"]):
    seqs_kmer = list()
    class_label = list()
    for item in seq_dict:
        seq = seq_dict[item]
        k_seq = seq2kmer(seq, k)
        seqs_kmer.append(k_seq)
        class_label.append(label)
    dataframe = pd.DataFrame(zip(seqs_kmer, class_label), columns=columns, index=None)
    return dataframe


def merge_pos_neg(pos_seqs, neg_seqs):
    """
    Merge positive and negative sequences into one file
    Divide into train and test
    """
    pos_df = read_seqs(pos_seqs, 1)
    neg_df = read_seqs(neg_seqs, 0)
    merged_df = pd.concat([pos_df, neg_df])
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    merged_df.to_csv("../data/train.tsv", sep="\t", index=None)
    print(merged_df)


################################################################################

if __name__ == '__main__':

    pos_fasta = "../data/positives.fa"
    neg_fasta = "../data/negatives.fa"
    
    # Load FASTA sequences into dictionaries.
    pos_seqs_dic = read_fasta_into_dic(pos_fasta, 
                                       convert_to_rna=True, 
                                       all_uc=True,
                                       skip_n_seqs=True)
    neg_seqs_dic = read_fasta_into_dic(neg_fasta,
                                       convert_to_rna=True, 
                                       all_uc=True,
                                       skip_n_seqs=True)

    print("# positive sequences:  %i" %(len(pos_seqs_dic)))
    print("# negative sequences:  %i" %(len(neg_seqs_dic)))

    merge_pos_neg(pos_seqs_dic, neg_seqs_dic)



