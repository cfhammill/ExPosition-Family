## a very simple plot with most of the parameters as pass-throughs to plot.

sp.component_plot <- function(scores, axes=c(1,2), pch=20, col="mediumorchid4", line.col="grey80", lty=2, lwd=2,
                              main="Component scores",
                              xlab=paste0("Component ",axes[1]),
                              ylab=paste0("Component ",axes[2]),
                              xlim=c(-max(abs(scores[,axes])),max(abs(scores[,axes])))*1.2,
                              ylim=c(-max(abs(scores[,axes])),max(abs(scores[,axes])))*1.2,
                              asp=1, pos=3, display_names=T,
                              ...){

  plot(0, type="n", xlim=xlim, ylim=ylim, main=main, xlab=xlab, ylab=ylab, axes=F, asp=asp)
  abline(h=0,lty=2,lwd=2, col="grey60")
  abline(v=0,lty=2,lwd=2, col="grey60")
  points(scores[,axes], col=col, pch=pch, ...)
    ## will try to employ a "repel" later.
  if(display_names){
    text(scores[,axes],labels=rownames(scores),pos=pos,col=col)
  }


}
