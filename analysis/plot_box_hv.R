hv=matrix(nrow=0,ncol=2)
file_names=c("../results/stochastic/hv.csv","../just_mutations/results/hv.csv","../results/deterministic/hv.csv")
alg_names=c("M2","M1","NA")
for (alg in 1:length(file_names)){
	hvd=read.csv(file_names[alg],header=FALSE)
	colnames(hvd)=c("hypervolume")
	hvd["algorithm"]=alg_names[alg]
	hv=rbind(hv,hvd)}
boxplot(hypervolume~algorithm,data=hv,ylab="hypervolume")#,boxwex=0.1)
dev.copy2eps(file="box_hv.eps", onefile=TRUE, width=5, height=4)
