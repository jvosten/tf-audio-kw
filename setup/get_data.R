#!/usr/bin/env Rscript

# ***************************************************************************************************************************** #
# ***************************************************  Script installer  ****************************************************** #
# ********************************************  Do not run in interactive mode  *********************************************** #
# ***************************************************************************************************************************** #

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  # stopifnot(args %in% c("--install"))

  dir.create("data")

  download.file(
    url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
    destfile = "data/speech_commands_v0.01.tar.gz"
  )

  untar("data/speech_commands_v0.01.tar.gz", exdir = "data/speech_commands_v0.01")
}

tryCatch(
  error = function(cnd) {
    print(paste0(conditionMessage(cnd), "Re-running the script; if it fails again you might
                 want to manually download the data at https://www.tensorflow.org/datasets/catalog/speech_commands,
                 as there tend to be certificate problems. "))
    main()
  },
  main()
)










