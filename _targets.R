# _targets.R file
library(targets)
library(tarchetypes)
source("R/functions.R")
options(tidyverse.quiet = TRUE)
tar_option_set(
  packages = c(
    "keras",
    "stringr",
    "rmarkdown",
    "dplyr",
    "tfdatasets",
    "alluvial",
    "here",
    "tensorflow",
    "magrittr",
    "fs"
  )
)
#reticulate::use_condaenv(condaenv = "tf-audio", required = TRUE)

list(
  tar_target(
    data_path,
    "~/src/dev/r/kenwave/data/speech_commands_v0.01/",
    format = "file"
  ),
  # Preparation for the data generator
  tar_target(
    gen_prep,
    prep_tbl(data_path),
    format = "fst_tbl",
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
  # Determine ratio of train/test set
  tar_target(
    id_train,
    id_sampler(df = gen_prep),
    format = "qs",
    deployment = "main"
  ),
  tar_target(
    model,
    train_model(
      df = gen_prep,
      id_train = id_train,
      batch_size = 32,
      window_size_ms = window_size_ms,
      window_stride_ms = window_stride_ms
    ),
    format = "keras",
    deployment = "main"
  ),
  tar_target(
    predictions,
    predicting(
      df = gen_prep,
      id_train = id_train,
      window_size_ms = window_size_ms,
      window_stride_ms = window_stride_ms,
      model = model
    ),
    format = "qs",
    deployment = "main"
  ),
  tar_target(
    alluvial_plot,
    alluv_plot(predictions, path = 'output/alluvial.png')
  ),
  tar_render(report, "report.Rmd")
)
