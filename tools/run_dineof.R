# Calls the pinned, unmodified sinkr implementation; no test labels are inputs.
args <- commandArgs(trailingOnly=TRUE)
source(args[1])
x <- as.matrix(read.csv(args[2], header=FALSE, na.strings="NA"))
refs <- scan(args[3], quiet=TRUE)
set.seed(as.integer(args[7]))
fit <- dineof(x, n.max=as.integer(args[5]), ref.pos=refs,
              delta.rms=as.numeric(args[6]), method="svd", verbose=FALSE)
write.table(fit$Xa, args[4], sep=",", row.names=FALSE, col.names=FALSE)
cat(fit$n.eof, "\n")
