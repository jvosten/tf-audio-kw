#' @title Generator preparation.
#' @description In this step we will list all audio .wav files into a tibble with 3 columns:
#' * `fname`: the file name;
#' * `class`: the label for each audio file;
#' * `class_id`: a unique integer number starting from zero for each class - used to one-hot encode the classes.
#' This will be useful to the next step when we will create a generator using the tfdatasets package.
#' @return A `tibble`
#' @param data_path Path to Speech Commands data dir.
#' @examples
#' library(tidyverse)
#' prep_tbl("data/speech_commands_v0.01/")
prep_tbl <- function(path) {
  files <- fs::dir_ls(
    path = path,
    recurse = TRUE,
    glob = "*.wav"
  )

  files <- files[!str_detect(files, "background_noise")]

  df <- tibble(
    fname = files,
    class = fname %>% str_extract("1/.*/") %>%
      str_replace_all("1/", "") %>%
      str_replace_all("/", ""),
    class_id = class %>% as.factor() %>% as.integer() - 1L
  )
}

#' @title Dataset generator
#' @description define a function called data_generator that will create the
#' generator depending on specified inputs.
#' @return A function that defines how the TensorFlow graph should read
#' and pre-process data. Not compiled or run yet.
#' @param generator_prep A `tibble` generated from [prep_tbl()]
#' @param batch_size
#' @param window_size_ms
#' @param window_stride_ms
#' @param shuffle
#' @examples
#' library(tfdatasets)
data_generator <- function(df,
                           batch_size = 32,
                           window_size_ms,
                           window_stride_ms,
                           shuffle = TRUE) {

  # assume sampling rate is the same in all samples
  sampling_rate <-
    tf$audio$decode_wav(
      tf$io$read_file(
        tf$reshape(
          df$fname[[1]], list()))) %>%
    .$sample_rate

  samples_per_window <- (sampling_rate * window_size_ms) %/% 1000L
  stride_samples <-  (sampling_rate * window_stride_ms) %/% 1000L

  n_periods <-
    tf$shape(
      tf$range(
        samples_per_window %/% 2L,
        16000L - samples_per_window %/% 2L,
        stride_samples
      )
    )[1] + 1L

  n_fft_coefs <-
    (2 ^ tf$math$ceil(tf$math$log(
      tf$cast(samples_per_window, tf$float32)
    ) / tf$math$log(2)) /
      2 + 1L) %>% tf$cast(tf$int32)

  ds <- tensor_slices_dataset(df)

  if (shuffle == TRUE){
    ds <- ds %>%
      dataset_shuffle(buffer_size = 100)
  }

  ds <- ds %>%
    dataset_map(function(obs) {
      wav <-
        tf$audio$decode_wav(tf$io$read_file(tf$reshape(obs$fname, list())))
      samples <- wav$audio
      samples <- samples %>% tf$transpose(perm = c(1L, 0L))

      stft_out <- tf$signal$stft(samples,
                                 frame_length = samples_per_window,
                                 frame_step = stride_samples)

      magnitude_spectrograms <- tf$abs(stft_out)
      log_magnitude_spectrograms <- tf$math$log(magnitude_spectrograms + 1e-6)

      response <- tf$one_hot(obs$class_id, 30L)

      input <- tf$transpose(log_magnitude_spectrograms, perm = c(1L, 2L, 0L))
      list(input, response)
    })

  ds <- ds %>%
    dataset_repeat()

  ds <- ds %>%
    dataset_padded_batch(
      batch_size = batch_size,
      padded_shapes = list(
        tf$stack(list(
          n_periods,
          n_fft_coefs,-1L)),
        tf$constant(-1L,
                    shape = shape(1L))),
      drop_remainder = TRUE
    )

  ds
}

#' @title Train set sampler
#' @description generates a sample of IDs of 70% of the data set for
#' the data set split
#' @return A vector
#' @param generator_prep A `tibble` generated from [prep_tbl()]
#' @examples
#' tbc
id_sampler <- function(df) {
  sample(nrow(df), size = 0.7*nrow(df))
}
