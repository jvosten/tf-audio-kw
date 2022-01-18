#!/usr/bin/env Rscript

# ***************************************************************************************************************************** #
# ***************************************************  Script installer  ****************************************************** #
# ********************************************  Do not run in interactive mode  *********************************************** #
# ***************************************************************************************************************************** #

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  # stopifnot(args %in% c("--install"))

  installer <- function(pkgs) {
    # Install packages not yet installed
    pkgs <- c(
      "alluvial",
      "qs",
      "fs",
      "fst",
      "keras",
      "remotes",
      "rmarkdown",
      "rprojroot",
      "tidyverse",
      "visNetwork",
      "yardstick",
      "tarchetypes",
      "tfdatasets",
      "tensorflow",
      "targets"
    )

    installed_pkgs <- pkgs %in% rownames(installed.packages())

    if (any(installed_pkgs == FALSE)) {
      install.packages(pkgs[!installed_pkgs], repos = "https://cran.rstudio.com/")
    }
  }
  if (length(args) > 0) {
    installer()
    cat("\n", "All packages installed", "\n")
  }
}

main()

# remotes::install_github("wlandau/targets")
# remotes::install_github("wlandau/tarchetypes")
# reticulate::install_miniconda()
keras::install_keras()


