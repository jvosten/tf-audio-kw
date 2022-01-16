library(targets)
library(tarchetypes)
source("R/functions.R")
source("R/utils.R")
options(tidyverse.quiet = TRUE)

tar_option_set(
  packages = c(
    "keras",
    "stringr",
    "rmarkdown",
    "dplyr",
    "tfdatasets",
    "alluvial",
    "here"
  )
)

# Define the pipeline. A pipeline is just a list of targets.
# list(
  tar_target(
    data_path,
    "data/speech_commands_v0.01/",
    format = "file",
    deployment = "main"
  ),
  tar_target(
    generator_prep,
    prep_tbl(data_file),
    format = "fst_tbl",
    deployment = "main"
  ),
  # Set parameters for spectogram creation
  tar_target(
    window_size_ms,
    c(30),
    deployment = "main"
  ),
  tar_target(
    window_stride_ms,
    c(10),
    deployment = "main"
  ),
  tar_target(
    generator,
    prepare_recipe(data),
    #format = "fst",
    deployment = "main"
  ),
#   tar_target(
#     run,
#     test_model(data, recipe, units1 = units, act1 = act),
#     pattern = cross(units, act),
#     format = "fst_tbl"
#   ),
#   tar_target(
#     best_run,
#     run %>%
#       top_n(1, accuracy) %>%
#       head(1),
#     format = "fst_tbl",
#     deployment = "main"
#   ),
#   tar_target(
#     best_model,
#     train_best_model(best_run, recipe),
#     format = "keras"
#   ),
#   tar_render(report, "report.Rmd")
# )
