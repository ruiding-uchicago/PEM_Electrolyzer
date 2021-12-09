reg<-function(class,sheetnum,axmin=-1,axmax=1){
  if (class==1){
    setwd('D:/学习/硕士/免疫/2021/imm')
  } else {
    setwd('D:/学习/硕士/免疫/2021/burden')
  }
  op<-read.xlsx('all-rf-op.xlsx',sheetnum)
  v<-which(is.na(op[2]))
  nv<-length(v)
  op[1:(nrow(op)-nv),2]<-op[2][-v,]
  op<-op[1:(nrow(op)-nv),]
  op<-as.data.frame(lapply(op,as.numeric))
  train<-apply(op[3:12],1,function(x) mean(x,na.rm=TRUE))
  test<-apply(op[13:22],1,function(x) mean(x,na.rm=TRUE))
  
  data<-data.frame(Observe=c(op[1:nrow(op),2],op[1:nrow(op),2]),
                   Predict=c(train,test),
                   Type=c(rep('Training set',nrow(op)),rep('Test set',nrow(op)))
  )
  
  rmse<-read.xlsx('all-rf-cor.xlsx',sheetnum)[5]
  rmse_test.mean<-mean(rmse$rmse_test)
  
  p<-ggplot(data=data,aes(x=Observe,y=Predict,color=Type,
                          shape=Type,fill=Type))+
    geom_abline(intercept=0,slope=1,size=1,color='#008B8B')+
    geom_abline(intercept=rmse_test.mean,slope=1,size=0.5,linetype="dashed",color='#008B8B')+
    geom_abline(intercept=0-rmse_test.mean,slope=1,size=0.5,linetype="dashed",color='#008B8B')+
    geom_point(size=1.5)+
    scale_x_continuous(limits=c(axmin,axmax))+
    scale_y_continuous(limits=c(axmin,axmax))+
    labs(title=colnames(op)[2])+
    coord_fixed(ratio=1)+
    scale_shape_manual(name="Data Source",
                       values=c(16,17))+
    scale_colour_manual(name="Data Source",
                        values=c("#f8766d","#00bfc4"))+
    theme_bw()+
    theme(axis.line=element_line(color='black'),
          axis.ticks.length=unit(0.5,'line'))+
          #axis.text = element_blank())+
    xlab(NULL)+
    ylab(NULL)+
    theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
    theme(legend.position="none")
  print(p)
  setwd('D:/学习/硕士/免疫/2021/plot/scatter')
  ggsave(paste0('rf-',colnames(op)[2],'.pdf'),width=3,height=3)
  return(p)
}