**Description of the files**

**Random Sequences Selection for Reference.ipynb** - After the basecalling of the randomers, we need to align to a reference. The code takes the reads,
searching for the patterns that occur in the designed artificial reads and writes a reference accordingly.

**FocusBedFile.ipynb** - to prepare the data for training we need to supply a focus file which indicates where in the reads to build chunks for training.
This file creates a bed file that indicates the location of the O6MeG in the reads, the non-modified G's that are found in the canonical reads and the
non-modified G's in the modified reads. The same for A's.

**ROC_AUC.ipynb** - receives a validation file from the remora package and builds a ROC graph.

**o6meg_choose_A.ipynb** - takes the aligned reads from a fastq file, looking for the known patterns, trims the reads and their quality scores and writing a new fastq files
that only contain the areas we are interested in. It allows us to enter the data to fastqc and visualize the quality around the modification. During that analyses we
also seperated the reads where the nanopore identified the O6MeG as A and G.
