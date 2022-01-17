library(targets)
library(tarchetypes)
source("R/functions.R")
source("R/utils.R")
options(tidyverse.quiet = TRUE)
set.seed(6)

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
    format = "qs",
    deployment = "main"
  ),
  # Set all parameters parameters for spectogram creation and neural net
  tar_target(
    window_size_ms,
    c(30)
  ),
  tar_target(
    window_stride_ms,
    c(10)
  ),
  # tar_target(
  #   window_size,
  #   as.integer(16000*window_size_ms/1000)
  # ),
  # tar_target(
  #   stride,
  #   as.integer(16000*window_stride_ms/1000)
  # ),
  # tar_target(
  #   fft_size,
  #   as.integer(2^trunc(log(window_size, 2)) + 1)
  # ),
  # tar_target(
  #   n_chunks,
  #   length(seq(window_size/2, 16000 - window_size/2, stride))
  # ),
  # Determine ratio of train/test set
  tar_target(
    id_train,
    id_sampler(generator_prep),
    format = "qs",
    deployment = "main"
  ),
  # Create a definition for train and test set
  tar_target(
    trainset_validation,
    ds_train(
      df = prep_tbl[id_train,],
      batch_size = 32,
      shuffle = TRUE,
      window_size_ms = window_size_ms,
      window_stride_ms = window_stride_ms
    ),
    format = "qs",
    deployment = "main"
  ),
  tar_target(
    dataset_validation,
    ds_validation(
      df = prep_tbl[-id_train,],
      batch_size = 32,
      shuffle = FALSE,
      window_size_ms = window_size_ms,
      window_stride_ms = window_stride_ms
    ),
    format = "qs",
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
