args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[1], .libPaths()))
d <- read.csv(args[2], stringsAsFactors=FALSE, na.strings='NA')
boundaries <- as.integer(strsplit(args[4], ',', fixed=TRUE)[[1]])
fit <- mpspline2::mpspline(d, var_name='value', d=boundaries,
                         lam=as.numeric(args[5]), vlow=as.numeric(args[6]), vhigh=as.numeric(args[7]))
if(length(fit)!=1) stop('Exactly one valid profile required')
r <- fit[[1]]
out <- data.frame(upper_cm=head(boundaries,-1), lower_cm=tail(boundaries,-1), value=as.numeric(r$est_dcm))
write.csv(out, args[3], row.names=FALSE, na='NA')
write.csv(data.frame(value=r$est_1cm), paste0(args[3], '.one_cm.csv'), row.names=FALSE)
write.csv(data.frame(value=r$est_icm), paste0(args[3], '.input_intervals.csv'), row.names=FALSE)
