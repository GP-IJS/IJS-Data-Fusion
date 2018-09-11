mut=read.csv("../mut.csv",header=FALSE)
mut[1]=10-mut[1]
colnames(mut)=c("dk","RSE")
mut[mut["dk"]==0,"dk"]="no mutation"
mut$dk=factor(mut$dk,levels=c("no mutation",1,2,3,4,5,6,7))
boxplot(RSE~dk,data=mut,xlab="number of added colums via mutation",ylab="RSE after gradient descent")#,boxwex=0.1)
dev.copy2eps(file="box_mut.eps",onefile=TRUE,width=8,height=6.4)
