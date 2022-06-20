import subprocess as sp
import glob
import random


"""
extract_graph returns indeces of connected nucleotides in the graph

file_name: path to ss.ps file of graph

"""


def extract_graph(file_name):


	file_ = open(file_name)
	lines = file_.readlines()
	
	start_ind = lines.index("/pairs [\n")
	end_ind = start_ind + lines[start_ind:].index("] def\n")

	connected = lines[start_ind + 1:end_ind]
	connected = [[int(c[:-1].split(" ")[0][1:])-1, int(c[:-1].split(" ")[1][:-1])-1] for c in connected]
	file_.close()



	return connected



"""

RNAfold

rnafold_cmd: "RNAfold" to use RNAfold
fasta_file: fasta file with id, sequence and constraints, e.g.


>DUT-NM_001948.4
ATCGTGCGCTCTCCTCTTCCCCCGGTGGTCTCCTCGCTCGCCTTCTGGCTCTGCC
......x......x.x..xx..x........xx.x.x...xx..x........x.


 
constraint: True if a dms based constraint is to be used


RNAfold("RNAfold",path_to_fasta_file, False)
RNAfold("RNAfold",path_to_fasta_file_with_constraints, True)

"""

def RNAfold(rnafold_cmd, fasta_file, constraint):


    fasta_file_preffix = fasta_file.rsplit('.', 1)[0]
    output_pdf = fasta_file_preffix + '_proteins.fa'
    log_file = fasta_file_preffix + '_RNAfold.log'
    rna_fold_cmd += ' {input_fasta} --filename-full'
    if constraint == True: rna_fold_cmd += " -C" 
        
    
    rna_fold_cmd = rna_fold_cmd.format(prodigal=rna_fold_cmd, input_fasta=fasta_file)


    with open(log_file, 'w') as lf:
        sp.call(rna_fold_cmd.split(), stdout=lf)
        

    return




	
if __name__ == "__main__":


	
