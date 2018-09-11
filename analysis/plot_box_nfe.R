nfes=matrix(nrow=0,ncol=2)
alg_names=c("NA","M1","M2")
file_names=c("../results/deterministic/nfe_n800k50.csv","../just_mutations/results/nfe_n800k50.csv","../results/stochastic/nfe_n800k50.csv")
for (alg in 1:length(file_names)){
	nfe=read.csv(file_names[alg],header=FALSE)
	colnames(nfe)=c("steps")
	nfe["algorithm"]=alg_names[alg]
	nfes=rbind(nfes,nfe)}
boxplot(steps~algorithm,data=nfes,ylab="evaluations")
dev.copy2eps(file="box_nfe.eps", onefile=TRUE, width=5, height=4)
