.libPaths(c('external/R-library', .libPaths()))
library(aqp)
data(sp2, package='aqp')
out <- 'output/algorithm_library_expansion/v5'
write.csv(sp2, file.path(out, 'aqp_sp2_original.csv'), row.names=FALSE, na='NA')
capture.output(utils:::.getHelpFile(help('sp2', package='aqp')), file=file.path(out, 'aqp_sp2_metadata.txt'))
s <- sp2[sp2$id=='hon-1', c('id', 'top', 'bottom', 'field_ph')]
write.csv(s, file.path(out, 'real_honcut_profile.csv'), row.names=FALSE, na='NA')
print(s)
result <- mpspline2::mpspline(s, var_name='field_ph', d=c(0,5,15,30,60), vlow=0, vhigh=14)
str(result)
upstream <- new.env()
source('output/algorithm_library_expansion/upstream/obrl-soil__mpspline2/R/mpspline.R', local=upstream)
functions <- c('mpspline','mpspline_one','mpspline_est1','mpspline_fit1')
identical_code <- sapply(functions, function(n) identical(deparse(body(get(n, envir=upstream))),
                                                        deparse(body(getFromNamespace(n, 'mpspline2')))))
write.csv(data.frame(function_name=functions, installed_matches_official_source=identical_code),
          file.path(out, 'mpspline_source_match.csv'), row.names=FALSE)
stopifnot(all(identical_code))
