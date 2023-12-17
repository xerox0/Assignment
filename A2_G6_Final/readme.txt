Quick instructions for use:
	- Replace hpc-lab-assignment-repo/OpenMP/linear-algebra/solvers/durbin with ./durbin
	- Move into the durbin solver directory
	- Compile with 'make clean all [run|profile] EXERCISE=durbin_new_alldevice.cu'. Notable EXT_CXXFLAGS are:
		-DDATA_T=<data type> (float is suggested, double aren't supported anymore, because of atomicAdd)
		-DN=<dataset size> to specify dataset size
		-DBLOCK_SIZE=<block size> to specify block size
	- Testing results and presentation can be found in . 
